# Copyright 2025 (authors: see run.sh)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Overlap-aware RTTM post-processing using PyAnnote OverlappedSpeechDetection.

Takes single-speaker RTTM output and augments it with a second speaker label
in regions where PyAnnote detects overlapping speech.

Supports multiprocessing (--nj) for speed.

Requirements:  pip install pyannote.audio
               HuggingFace token with access to pyannote/segmentation-3.0
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
import concurrent.futures
from collections import defaultdict

import numpy as np
import torch
# PyTorch >=2.6 defaults to weights_only=True and lightning_fabric passes it
# explicitly, which breaks pyannote checkpoint loading.  Force weights_only=False.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import torchaudio

from pyannote.audio.pipelines import OverlappedSpeechDetection


def get_args():
    parser = argparse.ArgumentParser(
        description="Overlap-aware RTTM augmentation")
    parser.add_argument("--rttm", required=True,
                        help="input single-speaker RTTM")
    parser.add_argument("--scp-wav", required=True,
                        help="wav.scp for audio files")
    parser.add_argument("--scp-emb", required=True,
                        help="emb.scp (Kaldi) for speaker centroids")
    parser.add_argument("--output", required=True,
                        help="output overlap-augmented RTTM file")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or HF_TOKEN env)")
    parser.add_argument("--min-overlap-dur", type=float, default=0.1,
                        help="minimum overlap segment duration (s)")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--channel", type=int, default=1,
                        help="RTTM channel field (default 1)")
    parser.add_argument("--nj", type=int, default=4,
                        help="number of parallel workers (default 4)")
    return parser.parse_args()


def read_wav_scp(path):
    wav_dict = {}
    for line in open(path):
        parts = line.strip().split(None, 1)
        wav_dict[parts[0]] = parts[1]
    return wav_dict


def read_rttm(path):
    """Returns dict: utt -> list of (begin, dur, spk_label)"""
    utt_rttm = defaultdict(list)
    for line in open(path):
        parts = line.strip().split()
        if parts[0] != "SPEAKER":
            continue
        utt = parts[1]
        begin = float(parts[3])
        dur = float(parts[4])
        spk = parts[7]
        utt_rttm[utt].append((begin, dur, spk))
    return utt_rttm


def compute_speaker_centroids(emb_scp, rttm_dict):
    """Compute per-utterance speaker centroids from embeddings and RTTM."""
    import kaldiio

    utt_embs = defaultdict(list)
    for subseg_id, emb in kaldiio.load_scp_sequential(emb_scp):
        parts = subseg_id.split("-")
        utt = parts[0]
        begin_ms = int(parts[1])
        end_ms = int(parts[2])
        utt_embs[utt].append((begin_ms / 1000.0, end_ms / 1000.0, emb))

    utt_centroids = {}
    for utt, segments in rttm_dict.items():
        if utt not in utt_embs:
            continue
        spk_embs = defaultdict(list)
        for emb_begin, emb_end, emb in utt_embs[utt]:
            best_spk = None
            best_overlap = 0
            for rttm_begin, rttm_dur, spk in segments:
                rttm_end = rttm_begin + rttm_dur
                overlap = max(0, min(emb_end, rttm_end) -
                              max(emb_begin, rttm_begin))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = spk
            if best_spk is not None:
                spk_embs[best_spk].append(emb)

        centroids = {}
        for spk, embs in spk_embs.items():
            c = np.mean(embs, axis=0)
            c = c / max(np.linalg.norm(c), 1e-10)
            centroids[spk] = c
        utt_centroids[utt] = centroids

    return utt_centroids


def find_second_speaker(overlap_begin, overlap_end, rttm_segments, centroids):
    """
    Find primary speaker in the overlap region from RTTM, then pick the
    second-most-likely speaker by centroid cosine similarity.
    """
    primary_spk = None
    best_overlap = 0
    for rttm_begin, rttm_dur, spk in rttm_segments:
        rttm_end = rttm_begin + rttm_dur
        overlap = max(0, min(overlap_end, rttm_end) -
                       max(overlap_begin, rttm_begin))
        if overlap > best_overlap:
            best_overlap = overlap
            primary_spk = spk

    if primary_spk is None or len(centroids) < 2:
        return None, None

    primary_centroid = centroids.get(primary_spk)
    if primary_centroid is None:
        return None, None

    best_spk = None
    best_sim = -np.inf
    for spk, centroid in centroids.items():
        if spk == primary_spk:
            continue
        sim = np.dot(primary_centroid, centroid)
        if sim > best_sim:
            best_sim = sim
            best_spk = spk

    return primary_spk, best_spk


# ── Per-worker state (loaded once per process) ────────────────────────
_worker_pipeline = None


def _init_worker(hf_token, min_overlap_dur, device):
    """Process initializer: load OSD pipeline once per worker."""
    global _worker_pipeline
    torch.set_num_threads(1)
    _worker_pipeline = OverlappedSpeechDetection(
        segmentation="pyannote/segmentation-3.0",
        use_auth_token=hf_token)
    _worker_pipeline.instantiate({
        "min_duration_on": min_overlap_dur,
        "min_duration_off": 0.1,
    })


def _process_one_utt(task):
    """
    Process a single utterance: detect overlaps and find second speakers.
    Returns list of (utt, ov_begin, dur, secondary_spk) tuples.
    """
    global _worker_pipeline
    utt, wav_path, segments, centroids = task

    if len(centroids) < 2:
        return []

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    overlap_ann = _worker_pipeline({"waveform": waveform, "sample_rate": sr})

    results = []
    for segment in overlap_ann.get_timeline().support():
        ov_begin, ov_end = segment.start, segment.end
        primary, secondary = find_second_speaker(
            ov_begin, ov_end, segments, centroids)
        if secondary is not None:
            results.append((utt, ov_begin, ov_end - ov_begin, secondary))
    return results


def main():
    args = get_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)

    wav_dict = read_wav_scp(args.scp_wav)
    rttm_dict = read_rttm(args.rttm)

    print("  Computing speaker centroids ...", file=sys.stderr)
    utt_centroids = compute_speaker_centroids(args.scp_emb, rttm_dict)

    # Build task list
    tasks = []
    for utt, segments in rttm_dict.items():
        if utt not in wav_dict:
            print(f"WARNING: {utt} not in wav.scp, skipping", file=sys.stderr)
            continue
        centroids = utt_centroids.get(utt, {})
        tasks.append((utt, wav_dict[utt], segments, centroids))

    total = len(tasks)
    nj = min(args.nj, total) if total > 0 else 1
    channel = args.channel
    rttm_spec = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>"

    added = 0
    done = 0
    all_extra_lines = []

    def _progress():
        nonlocal done
        done += 1
        pct = done * 100 // total
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  Overlap: [{bar}] {done}/{total} ({pct}%) "
              f"added={added}",
              end="", file=sys.stderr)

    print(f"  Running OSD on {total} files (nj={nj}) ...", file=sys.stderr)

    if nj <= 1:
        _init_worker(hf_token, args.min_overlap_dur, args.device)
        for task in tasks:
            results = _process_one_utt(task)
            for utt, ov_begin, dur, secondary in results:
                all_extra_lines.append(
                    rttm_spec.format(utt, channel, ov_begin, dur, secondary))
                added += 1
            _progress()
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=nj,
            initializer=_init_worker,
            initargs=(hf_token, args.min_overlap_dur, args.device),
        ) as executor:
            future_to_task = {executor.submit(_process_one_utt, t): t
                              for t in tasks}
            for fut in concurrent.futures.as_completed(future_to_task):
                results = fut.result()
                for utt, ov_begin, dur, secondary in results:
                    all_extra_lines.append(
                        rttm_spec.format(
                            utt, channel, ov_begin, dur, secondary))
                    added += 1
                _progress()

    # Write output: original RTTM + extra overlap lines
    with open(args.output, "w") as fout:
        for line in open(args.rttm):
            fout.write(line)
        for line in all_extra_lines:
            fout.write(line + "\n")

    print(f"\n  Overlap detection: {added} additional RTTM segments added",
          file=sys.stderr)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

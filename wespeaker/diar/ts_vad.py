# Copyright 2026 WeSpeaker contributors
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
Profile-based Target-Speaker Voice Activity Detection (TS-VAD).

Refines an initial single-speaker RTTM by:
  1. Computing speaker centroids from embeddings + initial RTTM.
  2. Scoring every sub-segment against each centroid (cosine similarity).
  3. Detecting overlapping speech where multiple speakers have high scores.
  4. Applying temporal smoothing to reduce false alarms.
  5. Outputting a refined RTTM with overlap regions.

This is an embedding-only approach (no external model such as PyAnnote OSD
is required).  It complements VBx or any other clusterer.

Usage:
  python wespeaker/diar/ts_vad.py \\
      --rttm exp/vbx_cluster/test_rttm \\
      --scp-emb exp/test_embedding/emb.scp \\
      --output exp/vbx_cluster/test_rttm_tsvad \\
      --overlap-threshold 0.55 \\
      --min-overlap-dur 0.2 \\
      --smooth-win 3 \\
      --channel 1
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import kaldiio


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_rttm(path):
    """utt -> list[(begin, dur, spk_label)]"""
    utt_rttm = defaultdict(list)
    for line in open(path):
        parts = line.strip().split()
        if parts[0] != "SPEAKER":
            continue
        utt_rttm[parts[1]].append(
            (float(parts[3]), float(parts[4]), parts[7])
        )
    return utt_rttm


def _read_emb_by_utt(scp):
    """Return {utt: [(subseg_id, begin_sec, end_sec, emb), ...]}."""
    utt_embs = OrderedDict()
    for subseg_id, emb in kaldiio.load_scp_sequential(scp):
        parts = subseg_id.split("-")
        utt = parts[0]
        # Sub-segment id: utt-beginMS-endMS-beginFrame-endFrame
        seg_begin_ms = int(parts[1])
        frame_begin = int(parts[3])
        frame_end = int(parts[4])
        frame_shift_ms = 10  # default in VoxConverse recipe
        begin_sec = (seg_begin_ms + frame_begin * frame_shift_ms) / 1000.0
        end_sec = (seg_begin_ms + frame_end * frame_shift_ms) / 1000.0
        if utt not in utt_embs:
            utt_embs[utt] = []
        utt_embs[utt].append((subseg_id, begin_sec, end_sec, emb))
    return utt_embs


# ---------------------------------------------------------------------------
# Speaker centroids from RTTM + embeddings
# ---------------------------------------------------------------------------

def _compute_centroids(utt_embs, rttm_segments):
    """Compute L2-normalised speaker centroids for one utterance.

    Each sub-segment embedding is assigned to the RTTM speaker with maximum
    temporal overlap.  Returns {spk_label: centroid_vector}.
    """
    spk_embs = defaultdict(list)
    for _sid, begin, end, emb in utt_embs:
        best_spk = None
        best_ov = 0.0
        for rttm_begin, rttm_dur, spk in rttm_segments:
            rttm_end = rttm_begin + rttm_dur
            ov = max(0.0, min(end, rttm_end) - max(begin, rttm_begin))
            if ov > best_ov:
                best_ov = ov
                best_spk = spk
        if best_spk is not None:
            spk_embs[best_spk].append(emb)

    centroids = {}
    for spk, embs in spk_embs.items():
        c = np.mean(embs, axis=0)
        nrm = np.linalg.norm(c)
        if nrm > 1e-10:
            c /= nrm
        centroids[spk] = c
    return centroids


# ---------------------------------------------------------------------------
# Per-speaker scoring + overlap detection
# ---------------------------------------------------------------------------

def _l2norm(v):
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-10 else v


def _median_filter_1d(x, win):
    """Apply 1-D median filter (odd window) to boolean array."""
    if win <= 1:
        return x
    hw = win // 2
    out = np.array(x, dtype=bool)
    padded = np.pad(x.astype(np.float32), hw, mode="edge")
    for i in range(len(x)):
        out[i] = np.median(padded[i:i + 2 * hw + 1]) >= 0.5
    return out


def _process_utterance(utt_embs, rttm_segments, centroids,
                       overlap_threshold, gap_threshold,
                       smooth_win, min_overlap_dur):
    """Detect overlapping speech for one utterance.

    Returns list of (begin, dur, secondary_spk) for extra RTTM lines.
    """
    if len(centroids) < 2:
        return []

    T = len(utt_embs)
    spk_list = sorted(centroids.keys())
    K = len(spk_list)
    spk_idx = {s: i for i, s in enumerate(spk_list)}

    # Build embedding matrix and time grid
    emb_mat = np.stack([_l2norm(e[3]) for e in utt_embs])  # (T, D)
    centroid_mat = np.stack([centroids[s] for s in spk_list])  # (K, D)

    # Cosine similarity: (T, K)
    sim = emb_mat @ centroid_mat.T

    # Assign primary speaker from RTTM (not from scores)
    primary = np.full(T, -1, dtype=int)
    for t, (_sid, begin, end, _emb) in enumerate(utt_embs):
        best_spk = None
        best_ov = 0.0
        for rb, rd, rs in rttm_segments:
            re = rb + rd
            ov = max(0.0, min(end, re) - max(begin, rb))
            if ov > best_ov:
                best_ov = ov
                best_spk = rs
        if best_spk is not None and best_spk in spk_idx:
            primary[t] = spk_idx[best_spk]

    # For each non-primary speaker, detect high-similarity frames
    extra_lines = []
    for ki in range(K):
        # Boolean mask: this speaker is active AND not primary
        is_secondary = np.zeros(T, dtype=bool)
        for t in range(T):
            if primary[t] == ki:
                continue  # already assigned as primary
            if primary[t] < 0:
                continue  # no primary assignment
            # Check absolute threshold
            if sim[t, ki] < overlap_threshold:
                continue
            # Check gap to primary
            gap = sim[t, primary[t]] - sim[t, ki]
            if gap > gap_threshold:
                continue
            is_secondary[t] = True

        # Temporal smoothing
        is_secondary = _median_filter_1d(is_secondary, smooth_win)

        # Convert contiguous True regions to segments
        in_region = False
        region_start = 0
        for t in range(T):
            if is_secondary[t] and not in_region:
                in_region = True
                region_start = t
            elif not is_secondary[t] and in_region:
                in_region = False
                _emit_overlap(utt_embs, region_start, t,
                              spk_list[ki], min_overlap_dur, extra_lines)
        if in_region:
            _emit_overlap(utt_embs, region_start, T,
                          spk_list[ki], min_overlap_dur, extra_lines)

    return extra_lines


def _emit_overlap(utt_embs, t_start, t_end, spk_label, min_dur, out):
    """Convert sub-segment index range to an RTTM-style tuple."""
    begin_sec = utt_embs[t_start][1]
    end_sec = utt_embs[t_end - 1][2]
    dur = end_sec - begin_sec
    if dur >= min_dur:
        out.append((begin_sec, dur, spk_label))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Profile-based TS-VAD: refine RTTM with overlap detection "
                    "using speaker embeddings (no external model required)."
    )
    p.add_argument("--rttm", required=True,
                   help="input single-speaker RTTM from clustering stage")
    p.add_argument("--scp-emb", required=True, dest="scp_emb",
                   help="embedding scp (Kaldi ark/scp)")
    p.add_argument("--output", required=True,
                   help="output RTTM file (original + overlap lines)")
    p.add_argument("--overlap-threshold", type=float, default=0.55,
                   dest="overlap_threshold",
                   help="cosine similarity threshold for secondary speaker "
                        "activation (default: 0.55). Lower = more overlaps.")
    p.add_argument("--gap-threshold", type=float, default=0.35,
                   dest="gap_threshold",
                   help="maximum gap between primary and secondary speaker "
                        "cosine scores to accept overlap (default: 0.35). "
                        "Higher = more permissive overlap detection.")
    p.add_argument("--min-overlap-dur", type=float, default=0.2,
                   dest="min_overlap_dur",
                   help="discard overlap segments shorter than this (seconds; "
                        "default: 0.2)")
    p.add_argument("--smooth-win", type=int, default=3, dest="smooth_win",
                   help="median-filter window size for temporal smoothing "
                        "(odd integer; default: 3). 1 = no smoothing.")
    p.add_argument("--channel", type=int, default=1,
                   help="RTTM channel field (default: 1)")
    return p.parse_args()


def main():
    args = get_args()

    rttm_dict = _read_rttm(args.rttm)
    utt_embs_dict = _read_emb_by_utt(args.scp_emb)

    rttm_spec = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>"
    added = 0
    total = len(rttm_dict)
    all_extra = []

    for idx, (utt, segments) in enumerate(rttm_dict.items(), 1):
        if utt not in utt_embs_dict:
            print(f"  TS-VAD: WARNING {utt} not in emb.scp, skipping",
                  file=sys.stderr)
            continue

        centroids = _compute_centroids(utt_embs_dict[utt], segments)
        extras = _process_utterance(
            utt_embs_dict[utt], segments, centroids,
            overlap_threshold=args.overlap_threshold,
            gap_threshold=args.gap_threshold,
            smooth_win=args.smooth_win,
            min_overlap_dur=args.min_overlap_dur,
        )
        for begin, dur, spk in extras:
            all_extra.append(
                rttm_spec.format(utt, args.channel, begin, dur, spk)
            )
            added += 1

        if idx % 20 == 0 or idx == total:
            print(f"\r  TS-VAD: {idx}/{total} utterances, "
                  f"added={added} overlap segments",
                  end="", file=sys.stderr)

    # Write output: original RTTM + extra overlap lines
    with open(args.output, "w") as fout:
        for line in open(args.rttm):
            fout.write(line)
        for line in all_extra:
            fout.write(line + "\n")

    print(f"\n  TS-VAD: {added} overlap segments added -> {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

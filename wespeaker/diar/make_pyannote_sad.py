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
PyAnnote-based Voice Activity Detection.

Replaces Silero VAD with pyannote/segmentation-3.0 for higher-quality
speech boundaries.  Output format matches make_system_sad.py so the rest
of the pipeline works unchanged.

Requirements:  pip install pyannote.audio
               A HuggingFace token with access to pyannote/segmentation-3.0
               (set HF_TOKEN env or --hf-token).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import concurrent.futures
import functools

import torch
# PyTorch >=2.6 defaults to weights_only=True and lightning_fabric passes it
# explicitly, which breaks pyannote checkpoint loading.  Force weights_only=False.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import torchaudio
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from wespeaker.utils.file_utils import read_scp


def get_args():
    parser = argparse.ArgumentParser(
        description="PyAnnote VAD (segmentation-3.0)")
    parser.add_argument("--scp", required=True, help="wav scp")
    parser.add_argument("--min-duration", required=True, type=float,
                        help="min segment duration (seconds)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env)")
    parser.add_argument("--onset", type=float, default=0.5,
                        help="onset threshold (default 0.5)")
    parser.add_argument("--offset", type=float, default=0.5,
                        help="offset threshold (default 0.5)")
    parser.add_argument("--min-duration-on", type=float, default=0.0,
                        help="min speech duration for pyannote (seconds)")
    parser.add_argument("--min-duration-off", type=float, default=0.0,
                        help="min silence duration for pyannote (seconds)")
    parser.add_argument("--device", default="cpu",
                        help="cuda or cpu")
    parser.add_argument("--nj", type=int, default=4,
                        help="number of parallel workers (default 4)")
    return parser.parse_args()


def _build_pipeline(model_name, hf_token, device, onset, offset,
                    min_duration_on, min_duration_off):
    """Build a fresh VoiceActivityDetection pipeline (one per process)."""
    model = Model.from_pretrained(model_name, use_auth_token=hf_token)
    model = model.to(torch.device(device))
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate({
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off,
    })
    pipeline.onset = onset
    pipeline.offset = offset
    return pipeline


# Per-process pipeline cache (avoids reloading the model for every file)
_process_pipeline = None


def _init_worker(model_name, hf_token, device, onset, offset,
                 min_duration_on, min_duration_off):
    """Initializer for each worker process — loads model once."""
    global _process_pipeline
    torch.set_num_threads(1)
    _process_pipeline = _build_pipeline(
        model_name, hf_token, device, onset, offset,
        min_duration_on, min_duration_off)


def _vad_one_file(utt_wav_pair, min_duration):
    """Run VAD on a single file. Returns list of output lines."""
    global _process_pipeline
    utt, wav_path = utt_wav_pair

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    vad_result = _process_pipeline({"waveform": waveform, "sample_rate": sr})

    lines = []
    for segment in vad_result.get_timeline().support():
        begin = segment.start
        end = segment.end
        if end - begin >= min_duration:
            lines.append("{}-{:08d}-{:08d} {} {:.3f} {:.3f}".format(
                utt, int(begin * 1000), int(end * 1000), utt, begin, end))
    return lines


def main():
    args = get_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    utt_wav_list = read_scp(args.scp)
    total = len(utt_wav_list)
    nj = min(args.nj, total) if total > 0 else 1

    run_vad = functools.partial(_vad_one_file, min_duration=args.min_duration)

    # Progress bar on stderr so stdout stays clean for segment output
    completed = 0

    def _progress(n=total):
        nonlocal completed
        completed += 1
        pct = completed * 100 // n
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  PyAnnote VAD: [{bar}] {completed}/{n} ({pct}%)",
              end="", file=sys.stderr)

    if nj <= 1:
        # Single-process: simpler, avoids fork overhead
        _init_worker("pyannote/segmentation-3.0", hf_token, args.device,
                     args.onset, args.offset,
                     args.min_duration_on, args.min_duration_off)
        for pair in utt_wav_list:
            for line in run_vad(pair):
                print(line)
            _progress(total)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=nj,
            initializer=_init_worker,
            initargs=("pyannote/segmentation-3.0", hf_token, args.device,
                      args.onset, args.offset,
                      args.min_duration_on, args.min_duration_off),
        ) as executor:
            futures = {executor.submit(run_vad, pair): pair
                       for pair in utt_wav_list}
            # Collect results in submission order for deterministic output
            ordered_pairs = list(utt_wav_list)
            pair_to_future = {id(pair): fut
                              for fut, pair in futures.items()}
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                pair = futures[fut]
                results[id(pair)] = fut.result()
                _progress(total)
            # Print in original order
            for pair in ordered_pairs:
                for line in results[id(pair)]:
                    print(line)

    print("", file=sys.stderr)  # newline after progress bar
    sys.stdout.flush()


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

# Copyright 2025
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
FunASR FSMN-VAD (e.g. funasr/fsmn-vad on Hugging Face).

Requires: pip install funasr  (and torch; use the WeSpeaker .venv)

Output format matches make_system_sad.py / make_pyannote_sad.py.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
import contextlib
import functools
import logging
import os
import sys
import tempfile

import torch
import torchaudio
from wespeaker.diar.wav_channel import channel_index
from wespeaker.utils.file_utils import read_scp

_process_model = None


@contextlib.contextmanager
def _funasr_stdout_to_stderr():
    """FunASR prints e.g. 'funasr version: …' to stdout; keep stdout for segments only."""
    prev = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = prev


def get_args():
    p = argparse.ArgumentParser(description="FunASR FSMN-VAD -> WeSpeaker segments")
    p.add_argument("--scp", required=True, help="wav.scp")
    p.add_argument("--min-duration", required=True, type=float,
                   help="min segment duration (seconds)")
    p.add_argument("--model", default="fsmn-vad", help="FunASR VAD model id")
    p.add_argument("--model-revision", default="v2.0.4",
                   help="model revision / tag")
    p.add_argument("--hub", default="hf", choices=("hf", "ms"),
                   help="download hub: hf=Hugging Face, ms=ModelScope")
    p.add_argument("--device", default="cpu", help="cuda, cuda:0, cpu, ...")
    p.add_argument("--nj", type=int, default=4,
                   help="parallel workers (use 1 with GPU)")
    return p.parse_args()


def _init_worker(model, revision, hub, device):
    global _process_model
    torch.set_num_threads(1)
    logging.getLogger().setLevel(logging.WARNING)
    with _funasr_stdout_to_stderr():
        from funasr import AutoModel

        _process_model = AutoModel(
            model=model,
            model_revision=revision,
            hub=hub,
            device=device,
            disable_update=True,
            disable_pbar=True,
        )


def _pairs_from_result(res):
    """FunASR returns [{ 'key': ..., 'value': [[start_ms, end_ms], ...] }]."""
    if not res:
        return []
    raw = res[0].get("value") or []
    out = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((int(item[0]), int(item[1])))
    return out


def _vad_one_file(utt_wav_pair, min_duration):
    global _process_model
    utt, wav_path = utt_wav_pair
    wav, sr = torchaudio.load(wav_path)
    path_for_model = wav_path
    tmp_path = None
    if wav.shape[0] > 1:
        ch = channel_index()
        if ch >= wav.shape[0]:
            raise ValueError(
                f"{wav_path}: WESPEAKER_WAV_CHANNEL={ch} but file has "
                f"{wav.shape[0]} channel(s)"
            )
        wav = wav[ch : ch + 1]
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        torchaudio.save(tmp_path, wav, sr)
        path_for_model = tmp_path
    try:
        with _funasr_stdout_to_stderr():
            res = _process_model.generate(input=path_for_model, disable_pbar=True)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    lines = []
    for start_ms, end_ms in _pairs_from_result(res):
        if start_ms < 0 or end_ms < 0:
            continue
        begin = start_ms / 1000.0
        end = end_ms / 1000.0
        if end - begin < min_duration:
            continue
        lines.append("{}-{:08d}-{:08d} {} {:.3f} {:.3f}".format(
            utt, int(begin * 1000), int(end * 1000), utt, begin, end))
    return lines


def main():
    args = get_args()
    utt_wav_list = read_scp(args.scp)
    total = len(utt_wav_list)
    nj = min(args.nj, total) if total > 0 else 1
    run_vad = functools.partial(_vad_one_file, min_duration=args.min_duration)

    completed = 0

    def _progress(n=total):
        nonlocal completed
        completed += 1
        pct = completed * 100 // max(n, 1)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  FunASR VAD: [{bar}] {completed}/{n} ({pct}%)",
              end="", file=sys.stderr)

    if nj <= 1:
        _init_worker(args.model, args.model_revision, args.hub, args.device)
        for pair in utt_wav_list:
            for line in run_vad(pair):
                print(line)
            _progress(total)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=nj,
                initializer=_init_worker,
                initargs=(args.model, args.model_revision, args.hub,
                          args.device),
        ) as executor:
            futures = {executor.submit(run_vad, pair): pair
                       for pair in utt_wav_list}
            ordered_pairs = list(utt_wav_list)
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                pair = futures[fut]
                results[id(pair)] = fut.result()
                _progress(total)
            for pair in ordered_pairs:
                for line in results[id(pair)]:
                    print(line)

    print("", file=sys.stderr)
    sys.stdout.flush()


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

#!/usr/bin/env python3
"""
Given:
  - an RTTM file with diarization results
  - a wav.scp mapping utt IDs to WAV paths

Create per-speaker WAV files for each recording:
  - If only one speaker in the recording:
        <out_dir>/<utt>.wav
  - If multiple speakers:
        <out_dir>/<utt>_1.wav
        <out_dir>/<utt>_2.wav
        ...

where <utt> matches the base name of the original WAV (without .wav).
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split WAVs into per-speaker tracks based on diarization RTTM."
    )
    parser.add_argument(
        "--rttm",
        type=str,
        required=True,
        help="Input RTTM file with diarization results.",
    )
    parser.add_argument(
        "--wav-scp",
        type=str,
        required=True,
        help="wav.scp file: 'utt_id path/to/utt.wav'.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory to write per-speaker WAVs.",
    )
    return parser.parse_args()


def load_wav_scp(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, wav_path = line.split(maxsplit=1)
            mapping[utt] = wav_path
    return mapping


def load_rttm(
    path: str,
) -> Dict[str, OrderedDict[str, List[Tuple[float, float]]]]:
    """
    Returns:
        utt -> OrderedDict[label -> list of (start, end) in seconds]
    """
    utt2label2segs: Dict[str, OrderedDict[str, List[Tuple[float, float]]]] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            utt = parts[1]
            try:
                start = float(parts[3])
                dur = float(parts[4])
            except ValueError:
                continue
            end = start + dur
            label = parts[7]

            if utt not in utt2label2segs:
                utt2label2segs[utt] = OrderedDict()
            label2segs = utt2label2segs[utt]
            if label not in label2segs:
                label2segs[label] = []
            label2segs[label].append((start, end))
    return utt2label2segs


def cut_and_save(
    utt: str,
    wav_path: str,
    label2segs: "OrderedDict[str, List[Tuple[float, float]]]",
    out_dir: str,
):
    if not os.path.isfile(wav_path):
        print(f"[WARN] WAV not found for {utt}: {wav_path}")
        return

    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    base = os.path.splitext(os.path.basename(wav_path))[0]
    labels = list(label2segs.keys())

    def extract_track(segs: List[Tuple[float, float]]) -> torch.Tensor:
        pieces = []
        for (s, e) in segs:
            start_idx = max(0, int(round(s * sr)))
            end_idx = min(wav.size(1), int(round(e * sr)))
            if end_idx > start_idx:
                pieces.append(wav[:, start_idx:end_idx])
        if not pieces:
            return torch.zeros((1, 0), dtype=wav.dtype)
        return torch.cat(pieces, dim=1)

    if len(labels) == 1:
        track = extract_track(label2segs[labels[0]])
        out_path = os.path.join(out_dir, f"{base}.wav")
        if track.numel() > 0:
            torchaudio.save(out_path, track, sr)
    else:
        for idx, label in enumerate(labels, start=1):
            track = extract_track(label2segs[label])
            out_path = os.path.join(out_dir, f"{base}_{idx}.wav")
            if track.numel() > 0:
                torchaudio.save(out_path, track, sr)


def main():
    args = parse_args()

    if not os.path.isfile(args.rttm):
        raise FileNotFoundError(f"RTTM not found: {args.rttm}")
    if not os.path.isfile(args.wav_scp):
        raise FileNotFoundError(f"wav.scp not found: {args.wav_scp}")

    os.makedirs(args.out_dir, exist_ok=True)

    utt2wav = load_wav_scp(args.wav_scp)
    utt2label2segs = load_rttm(args.rttm)

    for utt, label2segs in utt2label2segs.items():
        if utt not in utt2wav:
            print(f"[WARN] utt {utt} not found in wav.scp (skip).")
            continue
        cut_and_save(utt, utt2wav[utt], label2segs, args.out_dir)


if __name__ == "__main__":
    main()


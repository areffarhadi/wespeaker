#!/usr/bin/env python3
"""
Split a diarization RTTM file into per-recording, per-speaker RTTMs.

Behavior:
- For each recording ID (UTT in RTTM),
  - If there is only one unique speaker label, write one RTTM:
        <out_dir>/<utt>.rttm
  - If there are multiple speaker labels, write one RTTM per speaker:
        <out_dir>/<utt>_1.rttm
        <out_dir>/<utt>_2.rttm
        ...
    where indices are assigned in a stable order over labels.

This matches the user's requirement:
- Single-speaker recordings keep the original filename.
- Multi-speaker recordings get an added suffix (_1, _2, ...).
"""

import argparse
import os
from collections import defaultdict, OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split RTTM into per-UTT, per-speaker RTTM files."
    )
    parser.add_argument(
        "--rttm",
        type=str,
        required=True,
        help="Input RTTM file with diarization results.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory to write per-UTT RTTM files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.rttm):
        raise FileNotFoundError(f"RTTM not found: {args.rttm}")

    os.makedirs(args.out_dir, exist_ok=True)

    # utt -> label -> list of lines
    utt2label2lines: dict[str, "OrderedDict[str, list[str]]"] = {}

    with open(args.rttm, "r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                # Not a valid md-eval style SPEAKER RTTM line
                continue
            utt = parts[1]
            label = parts[7]

            if utt not in utt2label2lines:
                utt2label2lines[utt] = OrderedDict()
            label2lines = utt2label2lines[utt]
            if label not in label2lines:
                label2lines[label] = []
            label2lines[label].append(line)

    for utt, label2lines in utt2label2lines.items():
        labels = list(label2lines.keys())
        if len(labels) == 1:
            # Single speaker: keep original filename
            out_path = os.path.join(args.out_dir, f"{utt}.rttm")
            with open(out_path, "w", encoding="utf-8") as fout:
                for line in label2lines[labels[0]]:
                    fout.write(line + "\n")
        else:
            # Multi-speaker: one file per speaker with numeric suffix
            for idx, label in enumerate(labels, start=1):
                out_path = os.path.join(args.out_dir, f"{utt}_{idx}.rttm")
                with open(out_path, "w", encoding="utf-8") as fout:
                    for line in label2lines[label]:
                        fout.write(line + "\n")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Convert Praat TextGrid files to RTTM format.

Each tier is treated as one speaker.
Only intervals with non-empty text are considered speech.

Usage:
    python textgrid_to_rttm.py --textgrid-dir <dir> --out-rttm-dir <dir> [--channel 1]

Output: one <file_id>.rttm per <file_id>.TextGrid under --out-rttm-dir.
"""

import argparse
import os
import re
import sys


def parse_textgrid(tg_path):
    """Parse a Praat TextGrid file.

    Returns list of (tier_name, [(xmin, xmax), ...]) for tiers with speech.
    """
    with open(tg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tiers = []
    tier_name = None
    intervals = []
    seg_xmin = seg_xmax = None
    in_interval = False

    for line in lines:
        stripped = line.strip()

        # New top-level item (tier boundary)
        if re.match(r'item\s*\[\d+\]:', stripped):
            if tier_name is not None and intervals:
                tiers.append((tier_name, intervals))
            tier_name = None
            intervals = []
            seg_xmin = seg_xmax = None
            in_interval = False
            continue

        # Tier name
        m = re.match(r'name\s*=\s*"(.*)"$', stripped)
        if m:
            tier_name = m.group(1)
            continue

        # Interval marker — reset segment bounds
        if re.match(r'intervals\s*\[\d+\]:', stripped):
            seg_xmin = seg_xmax = None
            in_interval = True
            continue

        if not in_interval or tier_name is None:
            continue

        # xmin / xmax inside an interval
        m = re.match(r'xmin\s*=\s*([\d.eE+\-]+)', stripped)
        if m:
            seg_xmin = float(m.group(1))
            continue

        m = re.match(r'xmax\s*=\s*([\d.eE+\-]+)', stripped)
        if m:
            seg_xmax = float(m.group(1))
            continue

        # text line — finalises the interval
        m = re.match(r'text\s*=\s*"(.*)"$', stripped)
        if m and seg_xmin is not None and seg_xmax is not None:
            text = m.group(1).replace('""', '"').strip()
            if text and seg_xmax > seg_xmin:
                intervals.append((seg_xmin, seg_xmax))
            seg_xmin = seg_xmax = None
            continue

    # Flush last tier
    if tier_name is not None and intervals:
        tiers.append((tier_name, intervals))

    return tiers


def textgrid_to_rttm(tg_path, file_id, out_rttm_path, channel=1):
    tiers = parse_textgrid(tg_path)
    lines = []
    for tier_name, intervals in tiers:
        for (xmin, xmax) in intervals:
            dur = xmax - xmin
            lines.append(
                f"SPEAKER {file_id} {channel} {xmin:.3f} {dur:.3f}"
                f" <NA> <NA> {tier_name} <NA> <NA>"
            )
    # Sort by start time
    lines.sort(key=lambda l: float(l.split()[3]))
    with open(out_rttm_path, 'w') as f:
        for l in lines:
            f.write(l + '\n')
    return len(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Convert TextGrid files to per-file RTTM format.')
    parser.add_argument('--textgrid-dir', required=True,
                        help='Directory containing *.TextGrid files')
    parser.add_argument('--out-rttm-dir', required=True,
                        help='Output directory for *.rttm files')
    parser.add_argument('--channel', type=int, default=1,
                        help='RTTM channel field (default: 1)')
    args = parser.parse_args()

    os.makedirs(args.out_rttm_dir, exist_ok=True)

    tg_files = sorted(f for f in os.listdir(args.textgrid_dir)
                      if f.endswith('.TextGrid'))
    if not tg_files:
        print(f"ERROR: no .TextGrid files in {args.textgrid_dir}", file=sys.stderr)
        sys.exit(1)

    total_segs = 0
    for tg_file in tg_files:
        file_id = os.path.splitext(tg_file)[0]
        tg_path = os.path.join(args.textgrid_dir, tg_file)
        out_path = os.path.join(args.out_rttm_dir, f"{file_id}.rttm")
        n = textgrid_to_rttm(tg_path, file_id, out_path, channel=args.channel)
        total_segs += n
        print(f"  {file_id}: {n} speech segments -> {out_path}")

    print(f"Done: {len(tg_files)} files, {total_segs} total segments -> {args.out_rttm_dir}")


if __name__ == '__main__':
    main()

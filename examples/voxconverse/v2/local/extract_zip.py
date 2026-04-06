#!/usr/bin/env python3
# Copyright 2026
#
# Extract .zip archives using only the Python standard library (stdlib zipfile).
# Use when the system `unzip` binary is not installed (common on restricted hosts).

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Extract a .zip into a directory (stdlib only).")
    p.add_argument("zip_path", type=Path, help="Path to the .zip file")
    p.add_argument(
        "dest_dir",
        type=Path,
        help="Directory to extract into (created if missing; same as unzip -d)",
    )
    args = p.parse_args()
    zpath = args.zip_path
    dest = args.dest_dir
    if not zpath.is_file():
        print(f"extract_zip.py: file not found: {zpath}", file=sys.stderr)
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(dest)
    print(f"extract_zip.py: extracted {zpath} -> {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()

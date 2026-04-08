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
RTTM-level fusion via the **DOVER-Lap** algorithm (same backend as
`dover-lap` in `run_updated.sh`). Use this to combine two (or more) system
RTTMs, e.g. clustering outputs from ResNet embeddings vs w2v-BERT embeddings.

Requires the `dover-lap` executable (install `dover-lap` / `dover_lap` in your
environment). If it is not on PATH, set **DOVER_LAP_BIN** to the full path, or
use the WeSpeaker repo `.venv` (auto-detected when present).

Example (two systems):
  python3 wespeaker/diar/fuse_rttm_doverlap.py fused.rttm \\
      exp/ahc_cluster/dev_funasr_fsmn_sad_rttm \\
      exp/ahc_cluster/dev_funasr_fsmn_sad_w2vbert_rttm \\
      --weight-type rank --gaussian-filter-std 0.5 --dover-weight 0.05

Custom weights (three inputs):
  python3 wespeaker/diar/fuse_rttm_doverlap.py out.rttm a.rttm b.rttm c.rttm \\
      --weight-type custom --custom-weight "0.5 0.3 0.2"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _default_dover_lap_candidates() -> list[str]:
    exe = os.environ.get("DOVER_LAP_BIN", "").strip()
    out = []
    if exe:
        out.append(exe)
    w = shutil.which("dover-lap")
    if w:
        out.append(w)
    here = Path(__file__).resolve()
    # wespeaker/wespeaker/diar/thisfile.py -> parents[2] = repo root
    repo_venv = here.parents[2] / ".venv" / "bin" / "dover-lap"
    if repo_venv.is_file():
        out.append(str(repo_venv))
    return out


def resolve_dover_lap_bin() -> str:
    for c in _default_dover_lap_candidates():
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        if c and shutil.which(c):
            return c
    raise FileNotFoundError(
        "Could not find `dover-lap`. Install the dover-lap package, put it on "
        "PATH, or set DOVER_LAP_BIN to the executable path "
        "(e.g. .../wespeaker/.venv/bin/dover-lap)."
    )


def get_args():
    p = argparse.ArgumentParser(
        description="Fuse RTTM files with DOVER-Lap (wrapper around dover-lap CLI)."
    )
    p.add_argument(
        "output_rttm",
        help="path to write fused RTTM (first argument to dover-lap)",
    )
    p.add_argument(
        "input_rttms",
        nargs="+",
        help="one or more input RTTM files",
    )
    p.add_argument(
        "--gaussian-filter-std",
        type=float,
        default=0.5,
        help="Gaussian filter std before voting (default: 0.5)",
    )
    p.add_argument(
        "--dover-weight",
        type=float,
        default=0.1,
        help="DOVER weighting factor (default: 0.1)",
    )
    p.add_argument(
        "--weight-type",
        choices=["rank", "custom", "norm"],
        default="rank",
        help="rank | custom | norm (default: rank)",
    )
    p.add_argument(
        "--custom-weight",
        type=str,
        default="",
        help='space-separated weights, e.g. "0.5 0.5" (only with --weight-type custom)',
    )
    p.add_argument(
        "--channel",
        type=int,
        default=1,
        help="output RTTM channel id (default: 1)",
    )
    p.add_argument(
        "--dover-lap-bin",
        default="",
        help="override path to dover-lap (else PATH or DOVER_LAP_BIN or repo .venv)",
    )
    return p.parse_args()


def main() -> int:
    args = get_args()
    if len(args.input_rttms) < 2:
        print("error: need at least two input RTTMs for fusion.", file=sys.stderr)
        return 2

    for r in args.input_rttms:
        if not os.path.isfile(r):
            print(f"error: missing input RTTM: {r}", file=sys.stderr)
            return 2

    if args.dover_lap_bin:
        bin_path = args.dover_lap_bin
        if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
            print(f"error: not an executable: {bin_path}", file=sys.stderr)
            return 2
    else:
        try:
            bin_path = resolve_dover_lap_bin()
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    out_dir = os.path.dirname(os.path.abspath(args.output_rttm))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = [
        bin_path,
        args.output_rttm,
        *args.input_rttms,
        "--gaussian-filter-std",
        str(args.gaussian_filter_std),
        "--dover-weight",
        str(args.dover_weight),
        "--weight-type",
        args.weight_type,
        "-c",
        str(args.channel),
    ]
    if args.weight_type == "custom":
        if not args.custom_weight.strip():
            print(
                "error: --weight-type custom requires --custom-weight",
                file=sys.stderr,
            )
            return 2
        cmd.extend(["--custom-weight", args.custom_weight.strip()])

    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

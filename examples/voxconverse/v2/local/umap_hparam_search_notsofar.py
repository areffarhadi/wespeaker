#!/usr/bin/env python3
# Copyright 2026
#
# Grid / list hyperparameter search for UMAP + HDBSCAN + PAHC on NOTSOFAR MTG (sc_*).
# Re-runs only pipeline stages 7–9 (cluster → RTTM → DER); embeddings must already exist.
#
# Usage (from examples/voxconverse/v2):
#   python local/umap_hparam_search_notsofar.py --dry-run
#   python local/umap_hparam_search_notsofar.py --output exp/umap_hparam_search/results.csv
#
# Env (optional, passed through to the run script): NOTSOFAR_EVAL_ROOT, sad_type is fixed by run script defaults.

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DER_RE = re.compile(
    r"OVERALL\s+SPEAKER\s+DIARIZATION\s+ERROR\s*=\s*([\d.]+)\s+percent",
    re.IGNORECASE,
)


def parse_der_percent(res_path: Path) -> float | None:
    if not res_path.is_file():
        return None
    text = res_path.read_text(encoding="utf-8", errors="replace")
    m = DER_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    merge_cutoffs = [float(x) for x in args.merge_cutoffs.split()]
    n_neighbors = [int(x) for x in args.n_neighbors.split()]
    min_dists = [float(x) for x in args.min_dists.split()]
    n_components = [int(x) for x in args.n_components.split()]
    hdbscan_sizes = [int(x) for x in args.hdbscan_min_cluster_sizes.split()]
    pahc_sizes = [int(x) for x in args.pahc_min_cluster_sizes.split()]
    pahc_absorbs = [float(x) for x in args.pahc_absorb_cutoffs.split()]

    combos: list[dict[str, Any]] = []
    for tup in itertools.product(
        merge_cutoffs,
        n_neighbors,
        min_dists,
        n_components,
        hdbscan_sizes,
        pahc_sizes,
        pahc_absorbs,
    ):
        mc, nn, md, nc, hs, ps, pa = tup
        combos.append(
            {
                "UMAP_MERGE_CUTOFF": mc,
                "UMAP_N_NEIGHBORS": nn,
                "UMAP_MIN_DIST": md,
                "UMAP_N_COMPONENTS": nc,
                "HDBSCAN_MIN_CLUSTER_SIZE": hs,
                "PAHC_MIN_CLUSTER_SIZE": ps,
                "PAHC_ABSORB_CUTOFF": pa,
            }
        )
    return combos


def load_json_grid(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects, e.g. [{\"UMAP_MERGE_CUTOFF\": 0.2}, ...]")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Item {i} is not an object")
        out.append(row)
    return out


def run_trial(
    v2_dir: Path,
    env_extra: dict[str, str],
    *,
    dry_run: bool,
) -> int:
    run_sh = v2_dir / "run_w2vbert_wpt_mhfa_zl389_notsofar_mtg_sc.sh"
    if not run_sh.is_file():
        print(f"ERROR: missing {run_sh}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.update(env_extra)

    cmd = ["bash", str(run_sh), "--stage", "7", "--stop_stage", "9"]
    if dry_run:
        print("DRY-RUN would run:", " ".join(cmd))
        print("  env:", env_extra)
        return 0

    # Run from v2_dir so relative paths (data/, exp/) resolve
    r = subprocess.run(cmd, cwd=str(v2_dir), env=env)
    return r.returncode


def infer_res_path(
    v2_dir: Path,
    *,
    partition: str,
    sad_type: str,
    cluster_type: str,
    w2v_suffix: str,
) -> Path:
    """Reconstruct exp/.../res path (must match run_w2vbert_wpt_mhfa_zl389_notsofar_mtg_sc.sh)."""
    name = f"{partition}_{sad_type}_sad{w2v_suffix}_res"
    return v2_dir / "exp" / f"{cluster_type}_cluster" / name


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hyperparameter search for UMAP path (stages 7–9) on NOTSOFAR MTG recipe."
    )
    ap.add_argument(
        "--v2-dir",
        type=Path,
        default=None,
        help="examples/voxconverse/v2 directory (default: this script's parent)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("exp/umap_hparam_search/results.csv"),
        help="CSV path (relative to v2-dir unless absolute)",
    )
    ap.add_argument(
        "--json-grid",
        type=Path,
        default=None,
        help="If set, use this JSON list of env dicts instead of Cartesian grid flags",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print trials only; do not run")
    ap.add_argument(
        "--require-emb",
        action="store_true",
        help="Abort if embedding exp/.../emb.scp is missing (checks one path from defaults)",
    )
    ap.add_argument(
        "--partition",
        default="notsofar",
        help="Must match partition= in run script (default notsofar)",
    )
    ap.add_argument(
        "--sad-type",
        default="funasr_fsmn",
        dest="sad_type",
        help="Must match sad_type in run script (default funasr_fsmn)",
    )
    ap.add_argument(
        "--cluster-type",
        default="umap",
        dest="cluster_type",
        help="Must match cluster_type (default umap)",
    )
    ap.add_argument(
        "--w2v-suffix",
        default="_w2vbert_wptmhfa_zl389_ns",
        dest="w2v_suffix",
        help="Must match W2V= in run script",
    )
    # Grid axes (ignored if --json-grid)
    ap.add_argument(
        "--merge-cutoffs",
        default="0.15 0.2 0.25",
        help="Space-separated UMAP_MERGE_CUTOFF values",
    )
    ap.add_argument(
        "--n-neighbors",
        default="8 16 24",
        dest="n_neighbors",
        help="Space-separated UMAP_N_NEIGHBORS",
    )
    ap.add_argument(
        "--min-dists",
        default="0.05",
        help="Space-separated UMAP_MIN_DIST",
    )
    ap.add_argument(
        "--n-components",
        default="-1",
        help="Space-separated UMAP_N_COMPONENTS (-1 = auto)",
    )
    ap.add_argument(
        "--hdbscan-min-cluster-sizes",
        default="4",
        dest="hdbscan_min_cluster_sizes",
        help="Space-separated HDBSCAN_MIN_CLUSTER_SIZE",
    )
    ap.add_argument(
        "--pahc-min-cluster-sizes",
        default="3",
        dest="pahc_min_cluster_sizes",
        help="Space-separated PAHC_MIN_CLUSTER_SIZE",
    )
    ap.add_argument(
        "--pahc-absorb-cutoffs",
        default="0.0",
        dest="pahc_absorb_cutoffs",
        help="Space-separated PAHC_ABSORB_CUTOFF",
    )
    args = ap.parse_args()

    v2_dir = args.v2_dir
    if v2_dir is None:
        v2_dir = Path(__file__).resolve().parent.parent
    v2_dir = v2_dir.resolve()

    if args.json_grid:
        trials = load_json_grid(args.json_grid.resolve())
    else:
        trials = build_grid(args)

    out_csv = args.output
    if not out_csv.is_absolute():
        out_csv = (v2_dir / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"v2_dir={v2_dir}")
    print(f"trials={len(trials)}  output_csv={out_csv}")
    if args.dry_run:
        for i, t in enumerate(trials):
            print(f"  [{i+1}] {t}")
        return 0

    # Optional embedding check (default sad_type / partition / W2V match run script)
    if args.require_emb:
        emb = (
            v2_dir
            / "exp"
            / f"{args.partition}_{args.sad_type}_sad_embedding{args.w2v_suffix}"
            / "emb.scp"
        )
        if not emb.is_file() or emb.stat().st_size == 0:
            print(f"ERROR: missing or empty embeddings: {emb}", file=sys.stderr)
            print("  Run stages 1–6 first (e.g. --stage 1 --stop_stage 6).", file=sys.stderr)
            return 1

    fieldnames = [
        "trial_idx",
        "ok",
        "seconds",
        "der_percent",
        "res_file",
        "UMAP_MERGE_CUTOFF",
        "UMAP_N_NEIGHBORS",
        "UMAP_MIN_DIST",
        "UMAP_N_COMPONENTS",
        "HDBSCAN_MIN_CLUSTER_SIZE",
        "PAHC_MIN_CLUSTER_SIZE",
        "PAHC_ABSORB_CUTOFF",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as fd:
        w = csv.DictWriter(fd, fieldnames=fieldnames)
        w.writeheader()

        for idx, trial in enumerate(trials):
            env_str = {k: str(v) for k, v in trial.items()}
            t0 = time.perf_counter()
            rc = run_trial(v2_dir, env_str, dry_run=False)
            elapsed = time.perf_counter() - t0

            # Map env keys for CSV
            row = {
                "trial_idx": idx + 1,
                "ok": 1 if rc == 0 else 0,
                "seconds": f"{elapsed:.1f}",
                "der_percent": "",
                "res_file": "",
                "UMAP_MERGE_CUTOFF": trial.get("UMAP_MERGE_CUTOFF", ""),
                "UMAP_N_NEIGHBORS": trial.get("UMAP_N_NEIGHBORS", ""),
                "UMAP_MIN_DIST": trial.get("UMAP_MIN_DIST", ""),
                "UMAP_N_COMPONENTS": trial.get("UMAP_N_COMPONENTS", ""),
                "HDBSCAN_MIN_CLUSTER_SIZE": trial.get("HDBSCAN_MIN_CLUSTER_SIZE", ""),
                "PAHC_MIN_CLUSTER_SIZE": trial.get("PAHC_MIN_CLUSTER_SIZE", ""),
                "PAHC_ABSORB_CUTOFF": trial.get("PAHC_ABSORB_CUTOFF", ""),
            }

            if rc == 0:
                res_path = infer_res_path(
                    v2_dir,
                    partition=args.partition,
                    sad_type=args.sad_type,
                    cluster_type=args.cluster_type,
                    w2v_suffix=args.w2v_suffix,
                )
                der = parse_der_percent(res_path)
                row["der_percent"] = "" if der is None else f"{der:.4f}"
                row["res_file"] = str(res_path)

            w.writerow(row)
            print(
                f"[{idx+1}/{len(trials)}] rc={rc} der={row['der_percent'] or '?'} "
                f"trial={env_str}",
                flush=True,
            )

    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

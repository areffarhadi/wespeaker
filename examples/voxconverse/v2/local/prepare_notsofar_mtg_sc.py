#!/usr/bin/env python3
# Copyright 2026
#
# Prepare wav.scp and reference RTTMs for NOTSOFAR MTG benchmark (single-channel sc_* only).
# Ground truth: gt_transcription.json segments use a common meeting timeline (start_time/end_time).

import argparse
import json
from pathlib import Path

# Single-channel distant mics (ch0.wav) — same list as run_w2vbert_wpt_mhfa_zl389_notsofar_mtg_sc.sh
SC_SINGLE_DIRS = (
    "sc_meetup_0",
    "sc_plaza_0",
    "sc_rockfall_0",
    "sc_rockfall_1",
    "sc_rockfall_2",
    "sc_studio_0",
)

RTTM_LINE = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n"


def _speaker_map(participants: list[str], gt: list[dict]) -> dict[str, str]:
    names = list(participants) if participants else []
    seen = set(names)
    for u in gt:
        sid = u.get("speaker_id")
        if sid and sid not in seen:
            names.append(sid)
            seen.add(sid)
    names = sorted(set(names))
    return {n: f"SPEAKER_{i:02d}" for i, n in enumerate(names)}


def _merge_same_spk(segments: list[dict], spk_map: dict[str, str]) -> list[tuple[float, float, str]]:
    """Sort by time, merge adjacent intervals with same mapped speaker label."""
    rows = []
    for u in segments:
        st = float(u["start_time"])
        en = float(u["end_time"])
        if en <= st:
            continue
        spk = spk_map[u["speaker_id"]]
        rows.append((st, en, spk))
    rows.sort(key=lambda x: x[0])
    out: list[tuple[float, float, str]] = []
    for st, en, spk in rows:
        if not out:
            out.append((st, en, spk))
            continue
        ps, pe, pl = out[-1]
        if spk == pl and st <= pe + 1e-3:
            out[-1] = (ps, max(pe, en), pl)
        else:
            out.append((st, en, spk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build wav.scp + ref RTTMs for NOTSOFAR MTG (sc_* ch0 only)."
    )
    ap.add_argument(
        "--eval-root",
        required=True,
        help="Path to eval folder containing MTG/ (e.g. .../240825.1_eval_full_with_GT)",
    )
    ap.add_argument(
        "--out-wav-scp",
        required=True,
        help="Output wav.scp path (e.g. data/notsofar/wav.scp)",
    )
    ap.add_argument(
        "--out-ref-dir",
        required=True,
        help="Directory for reference RTTMs (e.g. data/notsofar_master/notsofar)",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=1,
        help="RTTM channel field (default 1, matches WeSpeaker make_rttm.py)",
    )
    args = ap.parse_args()

    eval_root = Path(args.eval_root).resolve()
    mtg_root = eval_root / "MTG"
    if not mtg_root.is_dir():
        raise SystemExit(f"Missing MTG/ under {eval_root}")

    out_wav = Path(args.out_wav_scp).resolve()
    out_ref = Path(args.out_ref_dir).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_ref.mkdir(parents=True, exist_ok=True)

    wav_rows: list[tuple[str, Path]] = []

    for meeting_dir in sorted(mtg_root.glob("MTG_*")):
        if not meeting_dir.is_dir():
            continue
        mid = meeting_dir.name
        gt_path = meeting_dir / "gt_transcription.json"
        meta_path = meeting_dir / "gt_meeting_metadata.json"
        if not gt_path.is_file():
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        participants: list[str] = []
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            participants = list(meta.get("ParticipantAliases") or [])

        spk_map = _speaker_map(participants, gt)
        merged = _merge_same_spk(gt, spk_map)

        for scene in SC_SINGLE_DIRS:
            wav_path = meeting_dir / scene / "ch0.wav"
            if not wav_path.is_file():
                continue
            utt_id = f"{mid}_{scene}"
            wav_rows.append((utt_id, wav_path.resolve()))

            rttm_path = out_ref / f"{utt_id}.rttm"
            with open(rttm_path, "w", encoding="utf-8") as rf:
                for st, en, spk in merged:
                    dur = en - st
                    if dur <= 0:
                        continue
                    rf.write(RTTM_LINE.format(utt_id, args.channel, st, dur, spk))

    wav_rows.sort(key=lambda x: x[0])
    with open(out_wav, "w", encoding="utf-8") as wf:
        for uid, path in wav_rows:
            wf.write(f"{uid} {path}\n")

    print(f"Wrote {len(wav_rows)} utterances -> {out_wav}")
    print(f"Reference RTTMs -> {out_ref} ({len(list(out_ref.glob('*.rttm')))} files)")


if __name__ == "__main__":
    main()

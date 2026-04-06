#!/usr/bin/env python3
# Copyright (c) 2026 — WeSpeaker VoxConverse w2v-BERT pipeline
#
# Vocal extraction with Demucs (htdemucs) before VAD. Self-contained in wespeaker;
# requires: pip install demucs soundfile torch torchaudio
#
# Demucs 3.x: demucs.api.Separator (separate_audio_file).
# Demucs 4.x: demucs.api was removed — use get_model + load_track + apply_model
# (same path as demucs.separate CLI). Vocals stem, mono 16 kHz for Silero VAD / w2v-BERT.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def read_scp(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, wav_path = line.split(maxsplit=1)
            pairs.append((utt, wav_path))
    return pairs


def resample_to_16k_mono(mono_1d: np.ndarray, orig_sr: int) -> tuple[np.ndarray, int]:
    """mono_1d: [T] float32."""
    t = torch.from_numpy(np.asarray(mono_1d, dtype=np.float32)).unsqueeze(0)
    if orig_sr != 16000:
        t = torchaudio.functional.resample(t, orig_freq=orig_sr, new_freq=16000)
    out = t.squeeze(0).numpy()
    return out, 16000


def _vocals_tensor_to_mono_numpy(vocals_t: torch.Tensor) -> np.ndarray:
    v = vocals_t.detach().cpu().numpy()
    if v.ndim == 1:
        return v
    return v.mean(axis=0)


def run_demucs_one_separator(
    audio_path: Path,
    sep: object,
    out_wav: Path,
) -> None:
    """Demucs 3.x: demucs.api.Separator."""
    assert hasattr(sep, "separate_audio_file")
    _, stems = sep.separate_audio_file(audio_path)
    vocals_t = stems.get("vocals")
    if vocals_t is None:
        raise RuntimeError("demucs did not return 'vocals' stem")

    v = vocals_t.detach().cpu().numpy()
    if v.ndim == 1:
        mono = v
    else:
        mono = v.mean(axis=0)
    sr = int(getattr(sep, "samplerate", 44100))

    mono_16k, sr_out = resample_to_16k_mono(mono, sr)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), mono_16k, sr_out, subtype="PCM_16")


def run_demucs_one_v4(
    audio_path: Path,
    model: object,
    device: str,
    out_wav: Path,
) -> None:
    """Demucs 4.x+: pretrained.get_model + separate.load_track + apply.apply_model."""
    from demucs.apply import apply_model
    from demucs.separate import load_track

    import torch as th

    wav = load_track(audio_path, model.audio_channels, model.samplerate)
    ref = wav.mean(0)
    wav = wav - ref.mean()
    std = ref.std()
    if float(std) < 1e-12:
        std = th.tensor(1.0, device=wav.device, dtype=wav.dtype)
    wav = wav / std

    # progress=False: inner chunk tqdm would flood the terminal; file-level bar is in main().
    sources = apply_model(
        model,
        wav[None],
        device=device,
        shifts=1,
        split=True,
        overlap=0.25,
        progress=False,
        num_workers=0,
        segment=None,
    )[0]
    sources = sources * std
    sources = sources + ref.mean()

    if "vocals" not in model.sources:
        raise RuntimeError(f"model has no 'vocals' stem, got {model.sources}")
    idx = model.sources.index("vocals")
    vocals_t = sources[idx]
    mono = _vocals_tensor_to_mono_numpy(vocals_t)
    sr = int(model.samplerate)

    mono_16k, sr_out = resample_to_16k_mono(mono, sr)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), mono_16k, sr_out, subtype="PCM_16")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract vocals with Demucs (htdemucs) for each utterance in wav.scp"
    )
    p.add_argument("--scp", required=True, help="Input Kaldi wav.scp (utt wav_path)")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory for per-utterance vocals WAV files ({utt}.wav)",
    )
    p.add_argument(
        "--wav-scp-out",
        required=True,
        help="Write new wav.scp pointing to vocals (same utterance ids)",
    )
    p.add_argument(
        "--model",
        default="htdemucs",
        help="Demucs model name (default: htdemucs)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="cuda | cpu (default: cuda)",
    )
    args = p.parse_args()

    pairs = read_scp(args.scp)
    if not pairs:
        print(f"{os.path.basename(__file__)}: empty or missing scp: {args.scp}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import demucs  # noqa: F401
    except ImportError as e:
        print(
            "demucs is not installed. Install with: pip install demucs\n"
            f"ImportError: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)
        device = "cpu"

    Separator = None
    try:
        from demucs.api import Separator  # type: ignore[attr-defined]
    except ImportError:
        Separator = None

    if Separator is not None:
        print(
            f"Loading Demucs Separator(model={args.model!r}, device={device}) ...",
            file=sys.stderr,
        )
        sep = Separator(model=args.model, device=device, progress=False)  # type: ignore[misc]
        run_one = lambda src, dst: run_demucs_one_separator(src, sep, dst)
    else:
        from demucs.pretrained import get_model

        print(
            f"Loading Demucs 4.x model {args.model!r} (device={device}) ...",
            file=sys.stderr,
        )
        model = get_model(args.model)
        model.cpu()
        model.eval()
        run_one = lambda src, dst: run_demucs_one_v4(src, model, device, dst)

    print(f"Demucs vocals: {len(pairs)} files -> {out_dir}", file=sys.stderr)

    out_lines = []
    with tqdm(
        pairs,
        desc="Demucs vocals",
        unit="file",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
    ) as pbar:
        for utt, wav_in in pbar:
            short = utt if len(utt) <= 48 else utt[:45] + "..."
            pbar.set_postfix_str(short, refresh=False)
            src = Path(wav_in.strip()).expanduser().resolve()
            if not src.is_file():
                tqdm.write(f"SKIP missing: {src}", file=sys.stderr)
                continue
            dst = out_dir / f"{utt}.wav"
            try:
                run_one(src, dst)
            except Exception as e:
                tqdm.write(f"ERROR {utt}: {e}", file=sys.stderr)
                raise
            out_lines.append(f"{utt} {dst}\n")

    if not out_lines:
        print("No utterances processed.", file=sys.stderr)
        sys.exit(1)

    scp_out = Path(args.wav_scp_out)
    scp_out.parent.mkdir(parents=True, exist_ok=True)
    with open(scp_out, "w") as f:
        f.writelines(out_lines)
    print(f"Wrote {len(out_lines)} lines -> {scp_out}", file=sys.stderr)


if __name__ == "__main__":
    main()

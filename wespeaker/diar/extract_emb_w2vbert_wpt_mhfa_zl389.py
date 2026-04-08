# Copyright (c) 2022 Xu Xiang
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2026 — WPT + W2V-BERT-2.0 + MHFA (USM_FTcode zl389 train recipe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Diarization embeddings from WPT+MHFA+w2v-BERT (zl389-style) training.
Model code defaults to the vendored copy under
  examples/voxconverse/v2/wpt_mhfa_zl389/
(override with --usm-ftcode or USM_FTCODE).

Checkpoint layout (under --ckpt-dir):
  - args.json          — hyperparameters saved at train start
  - best_model.pt      — dict with model_state_dict (and epoch, eer, ...)

This is NOT compatible with zl389 SpeechBrain-style model_base_*.pth used by
extract_emb_w2vbert.py; use run_w2vbert_wpt_mhfa_zl389.sh instead.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import kaldiio
import numpy as np
import torch
from tqdm import tqdm

from wespeaker.diar.extract_emb_w2vbert import subsegment_waveform
from wespeaker.diar.make_fbank import get_speech_segments, read_scp, read_segments
from wespeaker.utils.utils import validate_path


def _default_wpt_mhfa_backend_dir() -> str:
    """Recipe vendored backend: repo/examples/voxconverse/v2/wpt_mhfa_zl389."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    return str(repo_root / "examples" / "voxconverse" / "v2" / "wpt_mhfa_zl389")


def _load_usm_model_class(usm_ftcode: str):
    root = os.path.abspath(usm_ftcode)
    if root not in sys.path:
        sys.path.insert(0, root)
    mod_name = "main_train_simple_sv_wpt_w2vbert_mhfa_zl389"
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {mod_name} from USM_FTCODE={root}. "
            f"Use a Python that has USM deps (transformers, pytorch_wavelets, etc.). "
            f"Original error: {e}"
        ) from e
    return mod.SimpleSVModelWPTW2VBERTMHFA


def _num_speakers_from_state_dict(sd: dict) -> int:
    if "arcface_loss.weight" in sd:
        return int(sd["arcface_loss.weight"].shape[0])
    if "classifier.weight" in sd:
        return int(sd["classifier.weight"].shape[0])
    raise KeyError(
        "Cannot infer num_speakers: expected arcface_loss.weight or classifier.weight in checkpoint"
    )


def load_wpt_mhfa_model(
    ckpt_dir: str,
    checkpoint_name: str,
    usm_ftcode: str,
    map_location: torch.device,
):
    ckpt_dir = os.path.abspath(ckpt_dir)
    args_path = os.path.join(ckpt_dir, "args.json")
    ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Missing args.json: {args_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    with open(args_path, "r", encoding="utf-8") as f:
        train_args = json.load(f)

    blob = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict) or "model_state_dict" not in blob:
        raise ValueError(f"Expected dict with model_state_dict in {ckpt_path}")
    sd = blob["model_state_dict"]
    num_speakers = _num_speakers_from_state_dict(sd)

    SimpleSVModelWPTW2VBERTMHFA = _load_usm_model_class(usm_ftcode)
    use_arcface = bool(train_args.get("use_arcface", True))

    model = SimpleSVModelWPTW2VBERTMHFA(
        model_dir=train_args["xlsr"],
        num_speakers=num_speakers,
        embedding_dim=int(train_args.get("embedding_dim", 256)),
        num_prompt_tokens=int(train_args.get("num_prompt_tokens", 6)),
        num_wavelet_tokens=int(train_args.get("num_wavelet_tokens", 4)),
        prompt_dropout=float(train_args.get("prompt_dropout", 0.1)),
        num_heads=int(train_args.get("num_heads", 8)),
        compression_dim=int(train_args.get("compression_dim", 128)),
        adapter_bottleneck=int(train_args.get("adapter_bottleneck", 128)),
        head_dropout=float(train_args.get("head_dropout", 0.1)),
        use_arcface=use_arcface,
        arcface_margin=float(train_args.get("arcface_margin", 0.3)),
        arcface_scale=float(train_args.get("arcface_scale", 30.0)),
        # Weights come entirely from best_model.pt; avoid reloading zl389 .pth.
        w2vbert_encoder_ckpt=None,
        w2vbert_encoder_ckpt_lmft=None,
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
        if missing[:8]:
            print(f"  missing (sample): {missing[:8]}", file=sys.stderr)
        if unexpected[:8]:
            print(f"  unexpected (sample): {unexpected[:8]}", file=sys.stderr)

    model.eval()
    return model, train_args, blob


def forward_embedding_batches(
    waveforms: list,
    model,
    device: torch.device,
    *,
    batch_size: int = 8,
    use_amp: bool = True,
    l2_normalize: bool = True,
) -> np.ndarray:
    if not waveforms:
        return np.zeros((0, model.embedding_dim), dtype=np.float32)

    n_batches = (len(waveforms) + batch_size - 1) // batch_size
    print(
        f"Extracting WPT+MHFA embeddings: {len(waveforms)} sub-segments, "
        f"batch_size={batch_size}, {n_batches} batches, device={device}",
        file=sys.stderr,
    )

    all_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(waveforms), batch_size),
            total=n_batches,
            desc="embed",
            unit="batch",
        ):
            batch_w = waveforms[start : start + batch_size]
            max_len = max(len(w) for w in batch_w)
            bt = torch.zeros(len(batch_w), max_len, dtype=torch.float32)
            for i, w in enumerate(batch_w):
                arr = np.asarray(w, dtype=np.float32).reshape(-1)
                bt[i, : arr.shape[0]] = torch.from_numpy(arr)
            bt = bt.to(device)
            if device.type == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    emb = model.extract_embedding(bt, normalize=l2_normalize)
            else:
                emb = model.extract_embedding(bt, normalize=l2_normalize)
            if emb.dim() != 2:
                emb = emb.reshape(emb.shape[0], -1)
            feats = emb.float().detach().cpu().numpy()
            all_rows.append(feats)
    return np.concatenate(all_rows, axis=0).astype(np.float32, copy=False)


def get_args():
    p = argparse.ArgumentParser(
        description="Diarization embeddings via WPT+W2V-BERT+MHFA (USM_FTcode zl389 recipe)"
    )
    p.add_argument("--scp", required=True, help="wav.scp")
    p.add_argument("--segments", required=True, help="VAD segments")
    p.add_argument("--ark-path", required=True, help="output embedding ark")
    p.add_argument(
        "--ckpt-dir",
        required=True,
        help="Training out_fold with args.json + best_model.pt",
    )
    p.add_argument(
        "--checkpoint-name",
        default="best_model.pt",
        help="Checkpoint filename inside ckpt-dir (default: best_model.pt)",
    )
    p.add_argument(
        "--usm-ftcode",
        default=os.environ.get("USM_FTCODE") or _default_wpt_mhfa_backend_dir(),
        help="Dir with main_train_simple_sv_wpt_w2vbert_mhfa_zl389.py + losses.py "
        "(default: examples/voxconverse/v2/wpt_mhfa_zl389; override USM_FTCODE).",
    )
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--frame-shift", type=int, default=10)
    p.add_argument("--window-secs", type=float, default=1.5)
    p.add_argument("--period-secs", type=float, default=0.75)
    p.add_argument(
        "--subseg-cmn",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="Ignored (CLI compatibility with extract_emb.sh)",
    )
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast on CUDA")
    p.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Skip L2 normalization (training extract_embedding uses normalize=True)",
    )
    return p.parse_args()


def main():
    args = get_args()

    window_fs = int(args.window_secs * 1000) // args.frame_shift
    period_fs = int(args.period_secs * 1000) // args.frame_shift

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cuda":
        print("CUDA requested but not available; using CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Loading WPT+MHFA checkpoint dir: {args.ckpt_dir}", file=sys.stderr)
    model, train_args, meta = load_wpt_mhfa_model(
        args.ckpt_dir,
        args.checkpoint_name,
        args.usm_ftcode,
        map_location=device,
    )
    model.to(device)
    eer = meta.get("eer")
    ep = meta.get("epoch")
    print(
        f"Loaded best_model (epoch={ep}, val_eer={eer}); train_args xlsr={train_args.get('xlsr')}",
        file=sys.stderr,
    )

    train_audio_len = int(train_args.get("audio_len", 0))
    window_samples = int(16000 * args.window_secs)
    if train_audio_len and train_audio_len != window_samples:
        print(
            f"WARNING: training audio_len={train_audio_len} samples but "
            f"sub-segment window is ~{window_samples} ({args.window_secs}s @16kHz). "
            "Mismatch may hurt quality; align train_simple_sv... AUDIO_LEN / window_secs.",
            file=sys.stderr,
        )

    utt_to_wav = read_scp(args.scp)
    utt_to_segments = read_segments(args.segments)
    print(f"Loading audio: {len(utt_to_wav)} utterances ...", file=sys.stderr)
    speech_segments_id, speech_segments = get_speech_segments(utt_to_wav, utt_to_segments)
    print(f"  -> {len(speech_segments)} VAD segments", file=sys.stderr)

    subsegs: list = []
    subseg_wavs: list = []
    for seg_id, wav_np in tqdm(
        zip(speech_segments_id, speech_segments),
        total=len(speech_segments_id),
        desc="subseg",
        unit="seg",
    ):
        if hasattr(wav_np, "detach"):
            wav_np = wav_np.detach().cpu().numpy()
        wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
        ss, sw = subsegment_waveform(
            wav_np, seg_id, window_fs, period_fs, args.frame_shift, sr=16000
        )
        subsegs.extend(ss)
        subseg_wavs.extend(sw)
    print(f"  -> {len(subsegs)} sub-segments to embed", file=sys.stderr)

    use_amp = not args.no_amp
    l2_norm = not args.no_l2_normalize
    embeddings = forward_embedding_batches(
        subseg_wavs,
        model,
        device,
        batch_size=args.batch_size,
        use_amp=use_amp,
        l2_normalize=l2_norm,
    )

    validate_path(args.ark_path)
    emb_ark = os.path.abspath(args.ark_path)
    emb_scp = emb_ark[:-3] + "scp"

    with kaldiio.WriteHelper("ark,scp:" + emb_ark + "," + emb_scp) as writer:
        for i, subseg_id in enumerate(tqdm(subsegs, desc="write ark")):
            writer(subseg_id, embeddings[i])


if __name__ == "__main__":
    main()

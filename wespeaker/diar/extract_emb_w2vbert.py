# Copyright (c) 2022 Xu Xiang
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2026 — w2v-BERT-2.0 SV variant (see examples/voxconverse/v2/run_w2vbert.sh)
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
Embedding extraction for diarization using w2v-BERT-2.0 SV (PyTorch checkpoint), mirroring
wespeaker/diar/extract_emb.py sliding windows (window_secs / period_secs / frame_shift)
but on raw 16 kHz waveform instead of fbank + ONNX ResNet.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import kaldiio
import numpy as np
import torch
from tqdm import tqdm

from wespeaker.diar.make_fbank import get_speech_segments, read_scp, read_segments
from wespeaker.utils.utils import validate_path


def _load_infer_module(w2vbert_repo: str):
    """Load recipes/DeepASV/infer_w2v_bert_sv_embedding.py from the USM / w2v-BERT repo."""
    deepasv = os.path.join(w2vbert_repo, "recipes", "DeepASV")
    infer_path = os.path.join(deepasv, "infer_w2v_bert_sv_embedding.py")
    if not os.path.isfile(infer_path):
        raise FileNotFoundError(
            f"Expected infer_w2v_bert_sv_embedding.py at {infer_path}. "
            "Set --w2vbert-repo to the root of w2v-BERT-2.0_SV (contains deeplab/, recipes/)."
        )
    trans = os.path.join(
        w2vbert_repo,
        "deeplab",
        "pretrained",
        "audio2vector",
        "module",
        "transformers",
        "src",
    )
    for p in (w2vbert_repo, deepasv, trans):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location("infer_w2v_bert_sv_embedding", infer_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def subsegment_waveform(
    waveform_np: np.ndarray,
    seg_id: str,
    window_fs: int,
    period_fs: int,
    frame_shift: int,
    sr: int = 16000,
):
    """
    Same frame-based sliding as extract_emb.subsegment(), but on 1D waveform samples.
    """
    samples_per_frame = int(sr * frame_shift / 1000)
    window_samples = window_fs * samples_per_frame
    seg_begin, seg_end = seg_id.split("-")[-2:]
    seg_length_frames = (int(seg_end) - int(seg_begin)) // frame_shift

    wav = np.asarray(waveform_np, dtype=np.float32).reshape(-1)
    subsegs = []
    subseg_wavs = []

    if seg_length_frames <= window_fs:
        subseg = seg_id + "-{:08d}-{:08d}".format(0, seg_length_frames)
        subseg_wav = np.resize(wav, window_samples)
        subsegs.append(subseg)
        subseg_wavs.append(subseg_wav)
    else:
        max_subseg_begin = seg_length_frames - window_fs + period_fs
        for subseg_begin in range(0, max_subseg_begin, period_fs):
            subseg_end = min(subseg_begin + window_fs, seg_length_frames)
            subseg = seg_id + "-{:08d}-{:08d}".format(subseg_begin, subseg_end)
            chunk = wav[subseg_begin * samples_per_frame : subseg_end * samples_per_frame]
            subseg_wav = np.resize(chunk, window_samples)
            subsegs.append(subseg)
            subseg_wavs.append(subseg_wav)

    return subsegs, subseg_wavs


def get_args():
    parser = argparse.ArgumentParser(description="Diarization embeddings via w2v-BERT-2.0 SV")
    parser.add_argument("--scp", required=True, help="wav.scp (utterance -> wav path)")
    parser.add_argument("--segments", required=True, help="VAD segments file (same as make_fbank)")
    parser.add_argument("--ark-path", required=True, help="path to store embedding ark")
    parser.add_argument(
        "--w2vbert-repo",
        required=True,
        help="Root of w2v-BERT-2.0_SV (parent of deeplab/ and recipes/DeepASV/)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="model_base_*.pth with modules['spk_model']",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="forward batch size")
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=10,
        help="frame shift in ms (must match make_fbank / extract_emb)",
    )
    parser.add_argument(
        "--window-secs",
        type=float,
        default=1.5,
        help="sub-segment window (seconds); match run_updated.sh extract_emb / ResNet for fusion",
    )
    parser.add_argument(
        "--period-secs",
        type=float,
        default=0.75,
        help="sub-segment hop (seconds); match run_updated.sh extract_emb / ResNet for fusion",
    )
    parser.add_argument(
        "--subseg-cmn",
        default=True,
        type=lambda x: x.lower() == "true",
        help="Ignored for w2v-BERT (kept for CLI compatibility with extract_emb.sh)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable bfloat16 autocast on CUDA",
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="L2-normalize each embedding (spectral clustering already normalizes)",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    window_fs = int(args.window_secs * 1000) // args.frame_shift
    period_fs = int(args.period_secs * 1000) // args.frame_shift

    infer = _load_infer_module(args.w2vbert_repo)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cuda":
        print("CUDA requested but not available; using CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    model = infer.load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)

    utt_to_wav = read_scp(args.scp)
    utt_to_segments = read_segments(args.segments)

    print(f"Loading audio: {len(utt_to_wav)} utterances ...", file=sys.stderr)
    speech_segments_id, speech_segments = get_speech_segments(utt_to_wav, utt_to_segments)
    print(f"  -> {len(speech_segments)} VAD segments", file=sys.stderr)

    subsegs: list = []
    subseg_wavs: list = []
    for seg_id, wav_np in tqdm(zip(speech_segments_id, speech_segments),
                               total=len(speech_segments_id), desc="subseg", unit="seg"):
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
    embeddings = infer.forward_embedding_batches(
        subseg_wavs,
        model,
        device,
        batch_size=args.batch_size,
        use_amp=use_amp,
        l2_normalize=args.l2_normalize,
    )

    validate_path(args.ark_path)
    emb_ark = os.path.abspath(args.ark_path)
    emb_scp = emb_ark[:-3] + "scp"

    with kaldiio.WriteHelper("ark,scp:" + emb_ark + "," + emb_scp) as writer:
        for i, subseg_id in enumerate(tqdm(subsegs, desc="write ark")):
            writer(subseg_id, embeddings[i])


if __name__ == "__main__":
    main()

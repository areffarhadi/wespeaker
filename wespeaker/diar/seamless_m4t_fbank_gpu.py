# Copyright 2026 — GPU log-mel frontend matching HuggingFace SeamlessM4TFeatureExtractor
# (facebook/w2v-bert-2.0 preprocessor). See feature_extraction_seamless_m4t.py + audio_utils.spectrogram.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


def _num_stft_frames(num_samples: int, frame_length: int = 400, hop_length: int = 160) -> int:
    if num_samples < frame_length:
        return 0
    return 1 + (num_samples - frame_length) // hop_length


class SeamlessM4TLogMelGpu(nn.Module):
    """
    Batched log-mel features + per-bin norm + stride reshape, matching
    `SeamlessM4TFeatureExtractor` (center=False, kaldi mel, povey window).
    Input: float waveform (B, T) in [-1, 1] range (same as HF processor).
    Output: (B, T_frames // stride, num_mel_bins * stride) default (B, T', 160).
    """

    def __init__(
        self,
        *,
        mel_filters_np: np.ndarray,
        window_np: np.ndarray,
        stride: int = 2,
        mel_floor: float = 1.192092955078125e-07,
        preemphasis: float = 0.97,
        frame_length: int = 400,
        hop_length: int = 160,
        fft_length: int = 512,
    ):
        super().__init__()
        self.stride = stride
        self.mel_floor = mel_floor
        self.preemphasis = preemphasis
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fft_length = fft_length

        # mel_filters from HF: shape (num_fft_bins, num_mel) = (257, 80)
        mf = torch.from_numpy(mel_filters_np.astype(np.float32))
        self.register_buffer("mel_filters", mf)

        w = torch.from_numpy(window_np.astype(np.float32))
        if w.numel() != frame_length:
            raise ValueError(f"window length {w.numel()} != frame_length {frame_length}")
        self.register_buffer("window", w)

    @classmethod
    def from_hf_processor(cls, processor) -> "SeamlessM4TLogMelGpu":
        from transformers.audio_utils import mel_filter_bank, window_function

        sr = processor.sampling_rate
        n_mel = processor.num_mel_bins
        stride = processor.stride
        mel_filters = mel_filter_bank(
            num_frequency_bins=257,
            num_mel_filters=n_mel,
            min_frequency=20,
            max_frequency=sr // 2,
            sampling_rate=sr,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )
        window = window_function(400, "povey", periodic=False)
        return cls(mel_filters_np=mel_filters, window_np=window, stride=stride)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, n_samples) float32 on any device; values like HF (typically [-1,1]).
        Returns:
            input_features: (B, T_out, n_mel * stride) float32
        """
        if waveform.dim() != 2:
            raise ValueError(f"Expected (B, T) waveform, got {tuple(waveform.shape)}")
        # Kaldi-style scaling (matches SeamlessM4TFeatureExtractor._extract_fbank_features)
        x = waveform.to(dtype=torch.float32) * (2.0**15)

        bsz, length = x.shape
        fl, hl = self.frame_length, self.hop_length
        nfr = _num_stft_frames(length, fl, hl)
        if nfr <= 0:
            raise ValueError(f"Waveform too short for STFT: length={length}")

        # (B, nfr, fl)
        frames = x.unfold(dimension=1, size=fl, step=hl)
        if frames.shape[1] != nfr:
            frames = frames[:, :nfr, :]

        # remove_dc_offset per frame
        frames = frames - frames.mean(dim=-1, keepdim=True)

        # preemphasis (same order as transformers audio_utils.spectrogram loop)
        p = frames.clone()
        p[:, :, 1:] = p[:, :, 1:] - self.preemphasis * p[:, :, :-1]
        p[:, :, 0] = p[:, :, 0] * (1.0 - self.preemphasis)
        frames = p * self.window.view(1, 1, fl)

        buf = F.pad(frames, (0, self.fft_length - fl))
        spec = torch.fft.rfft(buf, n=self.fft_length, dim=-1)
        power = spec.abs().pow(2)

        # (B, nfr, n_mel)
        mel = torch.matmul(power, self.mel_filters)
        mel = torch.clamp(mel, min=self.mel_floor)
        mel = torch.log(mel)

        # Per-mel normalization over time (ddof=1), matches HF do_normalize_per_mel_bins
        mean = mel.mean(dim=1, keepdim=True)
        var = mel.var(dim=1, unbiased=True, keepdim=True)
        mel = (mel - mean) / torch.sqrt(var + 1e-7)

        # Stride reshape (SeamlessM4TFeatureExtractor.__call__)
        t = mel.shape[1]
        rem = t % self.stride
        if rem != 0:
            mel = mel[:, : t - rem, :]
        t2 = mel.shape[1]
        b, _, c = mel.shape
        mel = mel.reshape(b, t2 // self.stride, c * self.stride)
        return mel


def verify_gpu_matches_hf(
    processor,
    gpu_mod: SeamlessM4TLogMelGpu,
    lengths: tuple[int, ...] = (16000, 24000, 32001),
    device: Optional[torch.device] = None,
) -> None:
    """Sanity check vs HF CPU path (development / optional call)."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_mod = gpu_mod.to(dev)
    for L in lengths:
        rng = np.random.default_rng(42)
        wav = rng.standard_normal(L).astype(np.float32) * 0.1
        hf = processor(wav, sampling_rate=16000, return_tensors="pt", padding=False, truncation=False)
        hf_feat = hf["input_features"]
        if isinstance(hf_feat, torch.Tensor):
            hf_np = hf_feat.float().numpy()
        else:
            hf_np = np.asarray(hf_feat, dtype=np.float32)
        if hf_np.ndim == 2:
            hf_np = hf_np[np.newaxis, ...]

        with torch.no_grad():
            g = gpu_mod(torch.from_numpy(wav).unsqueeze(0).to(dev)).float().cpu().numpy()

        if g.shape != hf_np.shape:
            raise AssertionError(f"length {L}: shape gpu {g.shape} vs hf {hf_np.shape}")
        err = np.abs(g - hf_np).max()
        if err > 0.02:
            raise AssertionError(f"length {L}: max abs err {err} too large")

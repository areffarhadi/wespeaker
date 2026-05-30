# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Multi-channel WAV policy for diarization: use a single channel (default 0).

Set WESPEAKER_WAV_CHANNEL to override (integer, default 0). All stages that
load audio for VAD, fbank, embeddings, overlap, etc. should use these helpers
instead of torchaudio.load + mean(dim=0).
"""

from __future__ import annotations

import os

import torch
import torchaudio


def channel_index() -> int:
    return int(os.environ.get("WESPEAKER_WAV_CHANNEL", "0"))


def load_mono_1d(wav_path: str) -> tuple[torch.Tensor, int]:
    """
    Load WAV as a 1D float tensor [T] and sample rate.
    Multi-channel inputs keep only `channel_index()`; mono [1, T] is squeezed.
    """
    signal, sr = torchaudio.load(wav_path)
    ch = channel_index()
    if signal.ndim > 1 and signal.shape[0] > 1:
        if ch >= signal.shape[0]:
            raise ValueError(
                f"{wav_path}: WESPEAKER_WAV_CHANNEL={ch} but file has "
                f"{signal.shape[0]} channel(s)"
            )
        signal = signal[ch]
    else:
        signal = signal.squeeze(0)
    return signal, sr


def load_waveform_one_ch(wav_path: str) -> tuple[torch.Tensor, int]:
    """
    Load WAV as [1, T] for APIs such as PyAnnote (waveform dict).
    """
    waveform, sr = torchaudio.load(wav_path)
    ch = channel_index()
    if waveform.shape[0] > 1:
        if ch >= waveform.shape[0]:
            raise ValueError(
                f"{wav_path}: WESPEAKER_WAV_CHANNEL={ch} but file has "
                f"{waveform.shape[0]} channel(s)"
            )
        waveform = waveform[ch : ch + 1]
    return waveform, sr

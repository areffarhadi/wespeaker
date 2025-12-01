#!/usr/bin/env python3


# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

import copy
import os
import numpy as np
import fire
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs


def compute_fbank(wav_path, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0):
    """
    Extract fbank features from wav file.
    
    Args:
        wav_path: Path to wav file
        num_mel_bins: Number of mel filterbank bins
        frame_length: Frame length in ms
        frame_shift: Frame shift in ms
        dither: Dithering constant
        
    Returns:
        Fbank features as torch tensor (T, F)
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        sample_frequency=sample_rate,
        window_type='hamming',
        use_energy=False
    )
    return mat


def extract_to_npy(
    config='conf/config.yaml',
    model_path=None,
    wav_scp=None,
    wav_dir=None,
    output_dir=None,
    **kwargs
):
    """
    Extract embeddings and save as numpy files preserving folder structure.
    
    Args:
        config: Path to config.yaml
        model_path: Path to model checkpoint
        wav_scp: Path to wav.scp file (format: utt_id /path/to/wav) - Option 1
        wav_dir: Directory containing wav files (will scan recursively) - Option 2
        output_dir: Root directory to save embeddings
    
    Note: Provide either wav_scp OR wav_dir (not both)
    """
    # Parse configs
    configs = parse_config_or_kwargs(config, **kwargs)
    
    if model_path is None:
        raise ValueError("model_path must be provided")
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Read from wav.scp file
    if wav_scp is not None:
        print(f"Mode: Using wav.scp file")
        wav_dict = {}
        with open(wav_scp, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    utt_id, wav_path = parts
                    wav_dict[utt_id] = wav_path
        print(f"Extracting embeddings for {len(wav_dict)} utterances")
    
    # Option 2: Scan directory for wav files
    elif wav_dir is not None:
        print(f"Mode: Scanning directory for wav files")
        if not os.path.isdir(wav_dir):
            raise ValueError(f"wav_dir does not exist: {wav_dir}")
        
        wav_dict = {}
        print(f"Scanning directory: {wav_dir}")
        for root, dirs, files in os.walk(wav_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.flac', '.mp3')):
                    full_path = os.path.join(root, file)
                    # Use relative path from wav_dir as utterance ID
                    rel_path = os.path.relpath(full_path, wav_dir)
                    # Remove extension for utt_id
                    utt_id = os.path.splitext(rel_path)[0]
                    wav_dict[utt_id] = full_path
        
        if len(wav_dict) == 0:
            raise ValueError(f"No wav files found in {wav_dir}")
        
        print(f"Found {len(wav_dict)} audio files")
    
    else:
        raise ValueError("Either wav_scp or wav_dir must be provided")
    
    # Initialize model
    torch.backends.cudnn.benchmark = False
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    
    print('Loading checkpoint from:', model_path)
    load_checkpoint(model, model_path)
    print('Model loaded successfully!')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    # Get fbank configuration
    test_conf = copy.deepcopy(configs['dataset_args'])
    fbank_args = test_conf.get('fbank_args', {})
    num_mel_bins = fbank_args.get('num_mel_bins', 80)
    frame_length = fbank_args.get('frame_length', 25)
    frame_shift = fbank_args.get('frame_shift', 10)
    dither = 0.0  # No dither for evaluation
    
    # Process each wav file
    processed_count = 0
    
    with torch.no_grad():
        for utt_id, wav_path in tqdm(wav_dict.items()):
            try:
                # Compute fbank features
                feats = compute_fbank(
                    wav_path,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither
                )
                
                # Apply CMVN
                if test_conf.get('cmvn', True):
                    mean = feats.mean(dim=0, keepdim=True)
                    feats = feats - mean
                
                # Add batch dimension and move to device
                feats = feats.unsqueeze(0).float().to(device)  # (1, T, F)
                
                # Forward pass
                outputs = model(feats)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embedding = embeds.cpu().detach().numpy().squeeze()  # (embed_dim,)
                
                # Use utt_id as relative path, change extension to .npy
                rel_path = utt_id + ".npy"
                output_path = os.path.join(output_dir, rel_path)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save embedding
                np.save(output_path, embedding)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {utt_id}: {str(e)}")


if __name__ == '__main__':
    fire.Fire(extract_to_npy)


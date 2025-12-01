#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from tqdm import tqdm

#===========================
# Input paths
WAV_ROOT = "TidyVoiceX_eval"  # Directory containing Eval wav files
TRIAL_FILE = "eval_trials/tidyvoice_eval1.txt"  # trial file

# Output paths
DATA_ROOT = "data"  
OUTPUT_DIR = os.path.join(DATA_ROOT, "tidyvoice_eval1")  # Final output directory

#===========================


# Configuration
CREATE_SHARDS = True  
NUM_UTTS_PER_SHARD = 1000
NUM_THREADS = 16

def get_required_wav_files():
    """Extract all WAV file paths mentioned in the trial file"""
    print("Reading trial file to find required WAV files...")
    
    required_files = set()
    
    with open(TRIAL_FILE, 'r') as f:
        for line in tqdm(f, desc="Parsing trial file"):
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line.strip()}")
                continue
            
            label, enroll_path, test_path = parts
            required_files.add(enroll_path)
            required_files.add(test_path)
    
    print(f"Found {len(required_files)} unique WAV files in trial file")
    return required_files

def create_data_dirs():
    """Create necessary directories"""
    print(f"Creating directories in {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'trials'), exist_ok=True)

def create_wav_scp_and_utt2spk():
    """Create wav.scp and utt2spk files - only for files in trial file"""
    print("Creating wav.scp and utt2spk...")
    
    # First get the required WAV files from trial file
    required_files = get_required_wav_files()
    
    wav_scp_path = os.path.join(OUTPUT_DIR, 'wav.scp')
    utt2spk_path = os.path.join(OUTPUT_DIR, 'utt2spk')
    
    found_files = 0
    missing_files = []
    
    with open(wav_scp_path, 'w') as wav_scp, open(utt2spk_path, 'w') as utt2spk:
        for rel_path in tqdm(sorted(required_files), desc="Processing required WAV files"):
            # Construct full path
            full_path = os.path.join(WAV_ROOT, rel_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                missing_files.append(rel_path)
                continue
            
            # Extract speaker ID and wav name from relative path
            # rel_path format: spk_id/wav_name.wav
            path_parts = rel_path.split('/')
            if len(path_parts) < 2:
                print(f"Warning: Unexpected path format: {rel_path}")
                continue
                
            spk_id = path_parts[-2]  # Second to last part is speaker ID
            wav_name = Path(path_parts[-1]).stem  # Last part is wav file, remove extension
            
            # Create utterance ID: spk_id/wav_name
            utt_id = f"{spk_id}/{wav_name}"
            
            # Write to wav.scp: utt_id wav_path
            wav_scp.write(f"{utt_id} {full_path}\n")
            # Write to utt2spk: utt_id spk_id
            utt2spk.write(f"{utt_id} {spk_id}\n")
            found_files += 1
    
    # Report statistics
    print(f"Processed {found_files} WAV files successfully")
    if missing_files:
        print(f"Warning: {len(missing_files)} files from trial file not found:")
        for missing in missing_files[:10]:  # Show first 10 missing files
            print(f"  {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    # Create spk2utt using Kaldi script
    os.system(f"./tools/utt2spk_to_spk2utt.pl {utt2spk_path} > {os.path.join(OUTPUT_DIR, 'spk2utt')}")
    
    print(f"Created wav.scp and utt2spk in {OUTPUT_DIR}")
    return wav_scp_path, utt2spk_path

def convert_trials_to_kaldi():
    """Convert CANDOR trial file to Kaldi format"""
    print("Converting trials to Kaldi format...")
    
    output_path = os.path.join(OUTPUT_DIR, 'trials', 'candor_trials.kaldi')
    
    with open(TRIAL_FILE, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in, desc="Converting trials"):
            # Parse input line
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line.strip()}")
                continue
            
            label, enroll_path, test_path = parts
            
            # Extract utterance IDs from paths
            try:
                # Format: spk_id/wav_name from path like spk_id/wav_name.wav
                enroll_id = '/'.join(enroll_path.split('/')[-2:]).rsplit('.', 1)[0]
                test_id = '/'.join(test_path.split('/')[-2:]).rsplit('.', 1)[0]
                
                # Convert label (0/1) to target/nontarget
                label_str = "target" if label == "1" else "nontarget"
                
                # Write in Kaldi format: enroll_id test_id label
                f_out.write(f"{enroll_id} {test_id} {label_str}\n")
            except Exception as e:
                print(f"Warning: Error processing line: {line.strip()}")
                print(f"Error: {str(e)}")
    
    print(f"Created Kaldi trial file: {output_path}")
    return output_path

def create_shard_list():
    """Create shard list for the dataset"""
    print("Creating shard list...")
    
    wav_scp = os.path.join(OUTPUT_DIR, 'wav.scp')
    utt2spk = os.path.join(OUTPUT_DIR, 'utt2spk')
    shards_dir = os.path.join(OUTPUT_DIR, 'shards')
    shard_list = os.path.join(OUTPUT_DIR, 'shard.list')
    
    os.makedirs(shards_dir, exist_ok=True)
    
    cmd = f"python tools/make_shard_list.py \
        --num_utts_per_shard {NUM_UTTS_PER_SHARD} \
        --num_threads {NUM_THREADS} \
        --prefix shards \
        --shuffle \
        {wav_scp} {utt2spk} \
        {shards_dir} {shard_list}"
    
    os.system(cmd)
    print(f"Created shard list: {shard_list}")
    return shard_list

def main():
    print("\nStarting CANDOR dataset preparation...")
    print(f"WAV_ROOT: {WAV_ROOT}")
    print(f"TRIAL_FILE: {TRIAL_FILE}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    
    # Create data directories
    create_data_dirs()
    
    # Create wav.scp and utt2spk files (only for files in trial file)
    wav_scp, utt2spk = create_wav_scp_and_utt2spk()
    
    # Convert trials to Kaldi format
    kaldi_trials = convert_trials_to_kaldi()
    
    # Optionally create shard list
    if CREATE_SHARDS:
        shard_list = create_shard_list()
    
    print("\nDataset preparation complete!")
    print("\nTo use with run_vox_34_valid.py, use these settings:")
    print('DATA = "data"')
    print('TRIALS = "candor_trials.kaldi"')
    if CREATE_SHARDS:
        print('DATA_TYPE = "shard"')
    else:
        print('DATA_TYPE = "raw"')
    
    # Print some statistics
    with open(wav_scp, 'r') as f:
        num_utts = sum(1 for _ in f)
    with open(os.path.join(OUTPUT_DIR, 'spk2utt'), 'r') as f:
        num_spks = sum(1 for _ in f)
    with open(kaldi_trials, 'r') as f:
        num_trials = sum(1 for _ in f)
    
    print("\nDataset statistics:")
    print(f"Number of utterances: {num_utts}")
    print(f"Number of speakers: {num_spks}")
    print(f"Number of trials: {num_trials}")

if __name__ == '__main__':
    main() 


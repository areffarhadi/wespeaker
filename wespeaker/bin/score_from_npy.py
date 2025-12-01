#!/usr/bin/env python3


# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

import os
import numpy as np
import fire
from tqdm import tqdm


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_embeddings_from_npy(embedding_dir):
    """
    Load all embeddings from a directory of .npy files
    
    Args:
        embedding_dir: Directory containing .npy embedding files
        
    Returns:
        Dictionary mapping utterance IDs to embeddings
    """
    print(f"Loading embeddings from: {embedding_dir}")
    embeddings = {}
    
    # Walk through directory to find all .npy files
    for root, dirs, files in os.walk(embedding_dir):
        for file in files:
            if file.endswith('.npy'):
                full_path = os.path.join(root, file)
                # Get relative path from embedding_dir and remove .npy extension
                rel_path = os.path.relpath(full_path, embedding_dir)
                utt_id = os.path.splitext(rel_path)[0]
                
                # Load embedding
                try:
                    emb = np.load(full_path)
                    embeddings[utt_id] = emb
                except Exception as e:
                    print(f"Warning: Failed to load {full_path}: {e}")
    
    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def compute_mean_embedding(embeddings):
    """
    Compute mean embedding for mean subtraction
    
    Args:
        embeddings: Dictionary of embeddings
        
    Returns:
        Mean embedding vector
    """
    print("Computing mean from evaluation embeddings")
    all_embs = list(embeddings.values())
    
    if len(all_embs) == 0:
        return None
    
    mean_emb = np.mean(np.stack(all_embs), axis=0)
    print(f"Mean embedding shape: {mean_emb.shape}")
    return mean_emb


def score_from_npy(
    trial_file,
    embedding_dir,
    output_score_file,
    cal_mean=True
):
    """
    Compute cosine similarity scores from numpy embeddings
    
    Args:
        trial_file: Path to trial file (format: enroll_id test_id target/nontarget)
        embedding_dir: Directory containing .npy embedding files
        output_score_file: Path to output score file
        cal_mean: Whether to subtract mean embedding
    """
    print(f"Trial file: {trial_file}")
    print(f"Embedding dir: {embedding_dir}")
    print(f"Subtract mean: {cal_mean}")
    
    # Load embeddings
    embeddings = load_embeddings_from_npy(embedding_dir)
    
    if len(embeddings) == 0:
        raise ValueError(f"No embeddings found in {embedding_dir}")
    
    # Compute and subtract mean if requested
    mean_emb = None
    if cal_mean:
        mean_emb = compute_mean_embedding(embeddings)
        if mean_emb is not None:
            print("Subtracting mean from embeddings...")
            for utt_id in embeddings:
                embeddings[utt_id] = embeddings[utt_id] - mean_emb
    
    # Read trial file and compute scores
    print(f"\nProcessing trial file: {trial_file}")
    
    if not os.path.exists(trial_file):
        raise FileNotFoundError(f"Trial file not found: {trial_file}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_score_file), exist_ok=True)
    
    scores_computed = 0
    scores_failed = 0
    missing_enrolls = set()
    missing_tests = set()
    
    with open(trial_file, 'r') as f_in, open(output_score_file, 'w') as f_out:
        lines = f_in.readlines()
        
        for line in tqdm(lines, desc="Computing scores"):
            parts = line.strip().split()
            
            if len(parts) < 3:
                print(f"Warning: Malformed line: {line.strip()}")
                scores_failed += 1
                continue
            
            enroll_id = parts[0]
            test_id = parts[1]
            label = parts[2]
            
            # Check if embeddings exist
            if enroll_id not in embeddings:
                missing_enrolls.add(enroll_id)
                scores_failed += 1
                continue
            
            if test_id not in embeddings:
                missing_tests.add(test_id)
                scores_failed += 1
                continue
            
            # Compute cosine similarity
            try:
                score = cosine_similarity(embeddings[enroll_id], embeddings[test_id])
                # Write in format: enroll_id test_id score label
                f_out.write(f"{enroll_id} {test_id} {score:.6f} {label}\n")
                scores_computed += 1
            except Exception as e:
                print(f"Error computing score for {enroll_id} vs {test_id}: {e}")
                scores_failed += 1
    
    # Print summary
    print(f"Scores computed: {scores_computed}/{scores_computed + scores_failed}")


if __name__ == '__main__':
    fire.Fire(score_from_npy)


# Copyright 2025 (authors: see run.sh)
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
Agglomerative Hierarchical Clustering (AHC) for speaker diarization.

No training required — works directly on speaker embeddings.
Uses cosine similarity with a stopping threshold.

Algorithm:
  1. Start with each sub-segment as its own cluster.
  2. Compute cosine similarity between all cluster centroids.
  3. Merge the two most similar clusters.
  4. Repeat until the highest similarity falls below --threshold.

This is the clustering method used by most top-performing systems
in VoxSRC / DIHARD challenges.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
import functools
from collections import OrderedDict

import numpy as np
import kaldiio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def get_args():
    parser = argparse.ArgumentParser(
        description='AHC speaker clustering')
    parser.add_argument('--scp', required=True, help='embedding scp')
    parser.add_argument('--output', required=True, help='output label file')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='cosine similarity stopping threshold. '
                             'Lower = more merging (fewer speakers). '
                             'Tune on dev set. Default: 0.15')
    parser.add_argument('--linkage', default='average',
                        choices=['average', 'complete', 'single'],
                        help='linkage criterion (default: average)')
    parser.add_argument('--min_cluster_size', type=int, default=1,
                        help='minimum segments per cluster. Smaller '
                             'clusters get merged into nearest (default: 1)')
    args = parser.parse_args()
    return args


def read_emb(scp):
    emb_dict = OrderedDict()
    for sub_seg_id, emb in kaldiio.load_scp_sequential(scp):
        utt = sub_seg_id.split('-')[0]
        if utt not in emb_dict:
            emb_dict[utt] = {}
            emb_dict[utt]['sub_seg'] = []
            emb_dict[utt]['embs'] = []
        emb_dict[utt]['sub_seg'].append(sub_seg_id)
        emb_dict[utt]['embs'].append(emb)
    subsegs_list = []
    embeddings_list = []
    for utt, utt_emb_dict in emb_dict.items():
        subsegs_list.append(utt_emb_dict['sub_seg'])
        embeddings_list.append(np.stack(utt_emb_dict['embs']))
    return subsegs_list, embeddings_list


def cluster(embeddings, threshold=0.5, linkage_method='average',
            min_cluster_size=1):
    """
    AHC clustering on speaker embeddings.

    Args:
        embeddings: (N, D) numpy array
        threshold: cosine similarity stopping threshold
                   (converted to distance = 1 - similarity)
        linkage_method: 'average', 'complete', or 'single'
        min_cluster_size: absorb clusters smaller than this

    Returns:
        labels: list of int cluster labels
    """
    N = len(embeddings)
    if N <= 1:
        return [0] * N
    if N == 2:
        # Two segments: check if they're similar enough to be same speaker
        emb = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-10)
        sim = np.dot(emb[0], emb[1])
        return [0, 0] if sim >= threshold else [0, 1]

    # L2-normalise embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    emb_norm = embeddings / norms

    # Cosine similarity matrix -> distance matrix
    cos_sim = np.dot(emb_norm, emb_norm.T)
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    cos_dist = 1.0 - cos_sim

    # Ensure diagonal is exactly 0 and matrix is symmetric
    np.fill_diagonal(cos_dist, 0.0)
    cos_dist = (cos_dist + cos_dist.T) / 2.0
    # Fix any tiny negative values from floating point
    np.maximum(cos_dist, 0.0, out=cos_dist)

    # Convert to condensed form for scipy
    condensed = squareform(cos_dist, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed, method=linkage_method)

    # Cut the dendrogram at distance = 1 - threshold
    # (threshold is in similarity space, scipy works in distance space)
    dist_threshold = 1.0 - threshold
    labels = fcluster(Z, t=dist_threshold, criterion='distance')
    # fcluster labels start at 1, convert to 0-based
    labels = labels - 1

    # Absorb small clusters into nearest large cluster
    if min_cluster_size > 1:
        labels = _absorb_small_clusters(
            labels, emb_norm, min_cluster_size)

    return labels.tolist()


def _absorb_small_clusters(labels, emb_norm, min_cluster_size):
    """Merge clusters smaller than min_cluster_size into nearest neighbor."""
    unique, counts = np.unique(labels, return_counts=True)
    small = set(unique[counts < min_cluster_size])
    large = set(unique[counts >= min_cluster_size])

    if len(large) == 0:
        # All clusters are small — just return as-is
        return labels

    # Compute centroids for large clusters
    centroids = {}
    for c in large:
        mask = labels == c
        centroid = emb_norm[mask].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-10)
        centroids[c] = centroid

    # For each small cluster, find nearest large cluster
    for c in small:
        mask = labels == c
        small_centroid = emb_norm[mask].mean(axis=0)
        small_centroid /= max(np.linalg.norm(small_centroid), 1e-10)

        best_target = None
        best_sim = -np.inf
        for lc, lcentroid in centroids.items():
            sim = np.dot(small_centroid, lcentroid)
            if sim > best_sim:
                best_sim = sim
                best_target = lc

        labels[mask] = best_target

    # Relabel to consecutive integers
    unique_new = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique_new)}
    labels = np.array([remap[l] for l in labels])
    return labels


if __name__ == '__main__':
    args = get_args()

    subsegs_list, embeddings_list = read_emb(args.scp)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    run_cluster = functools.partial(
        cluster,
        threshold=args.threshold,
        linkage_method=args.linkage,
        min_cluster_size=args.min_cluster_size)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        with open(args.output, 'w') as fd:
            for (subsegs, labels) in zip(
                    subsegs_list,
                    executor.map(run_cluster, embeddings_list)):
                for subseg, label in zip(subsegs, labels):
                    print(subseg, label, file=fd)

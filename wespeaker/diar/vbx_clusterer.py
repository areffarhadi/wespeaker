# Copyright 2026 WeSpeaker contributors
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
VBx (Variational Bayes HMM x-vector) clustering for speaker diarization.

Based on: Landini et al., "Bayesian HMM clustering of x-vector sequences
(VBx) for speaker diarization" (2022).

Algorithm:
  1. Initialize speaker assignments with AHC.
  2. Model the segment sequence as an HMM with K speaker states.
  3. Use PLDA (or cosine) log-likelihoods as emission probabilities.
  4. Run VB iterations: forward-backward -> update speaker models.
  5. Final assignments from posterior responsibilities.

Supports two scoring backends:
  - Cosine similarity (default, no extra model needed).
  - PLDA (provide --plda-model trained with wespeaker/bin/train_plda.py).
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
import sys
from collections import OrderedDict

import numpy as np
import kaldiio
from scipy.special import logsumexp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# AHC initialization (lightweight, no extra imports)
# ---------------------------------------------------------------------------

def _ahc_init(embeddings, threshold=0.3, linkage_method="average"):
    """Return initial speaker labels from agglomerative clustering."""
    N = len(embeddings)
    if N <= 1:
        return np.zeros(N, dtype=int)
    if N == 2:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb = embeddings / np.maximum(norms, 1e-10)
        sim = np.dot(emb[0], emb[1])
        return np.array([0, 0] if sim >= threshold else [0, 1])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb = embeddings / np.maximum(norms, 1e-10)
    cos_sim = np.dot(emb, emb.T)
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    cos_dist = (cos_dist + cos_dist.T) / 2.0
    np.maximum(cos_dist, 0.0, out=cos_dist)
    condensed = squareform(cos_dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance") - 1
    return labels


# ---------------------------------------------------------------------------
# Forward-backward (log-space, vectorised)
# ---------------------------------------------------------------------------

def _forward_backward(log_emissions, log_trans, log_pi):
    """Standard HMM forward-backward in log-space.

    Args:
        log_emissions: (T, K) log emission probabilities.
        log_trans: (K, K) log transition matrix  [from, to].
        log_pi: (K,) log initial state distribution.

    Returns:
        gamma: (T, K) posterior responsibilities (probability space).
    """
    T, K = log_emissions.shape

    # Forward
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        # (K,1) + (K,K) -> (K,K); logsumexp over axis=0 -> (K,)
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
            + log_emissions[t]
        )

    # Backward
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        # (K,K) + (1,K) + (1,K) -> (K,K); logsumexp over axis=1 -> (K,)
        log_beta[t] = logsumexp(
            log_trans
            + log_emissions[t + 1][None, :]
            + log_beta[t + 1][None, :],
            axis=1,
        )

    # Posterior
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    return gamma


# ---------------------------------------------------------------------------
# VBx core
# ---------------------------------------------------------------------------

def _l2norm_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-10)


def cluster(embeddings, Fa=0.4, Fb=17.0, loopP=0.95,
            init_threshold=0.3, n_iters=10, plda=None):
    """VBx clustering for a single utterance.

    Args:
        embeddings: (T, D) numpy array of speaker embeddings.
        Fa: scaling factor for PLDA/cosine scores.
        Fb: scaling factor for speaker priors.
        loopP: HMM self-loop probability (speaker continuity).
        init_threshold: AHC cosine-similarity stopping threshold.
        n_iters: number of VB re-estimation iterations.
        plda: optional TwoCovPLDA instance (else cosine scoring).

    Returns:
        labels: list[int] of cluster labels (0-based, consecutive).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    T = len(embeddings)

    if T <= 2:
        return [0] * T

    # ── 1. AHC initialisation ────────────────────────────────────────────
    init_labels = _ahc_init(embeddings, threshold=init_threshold)
    K = int(init_labels.max()) + 1

    if K <= 1:
        return [0] * T

    # ── 2. Prepare embeddings ────────────────────────────────────────────
    if plda is not None:
        X = np.stack([plda.transform_embedding(e) for e in embeddings])
    else:
        X = _l2norm_rows(embeddings)

    # ── 3. Initialise speaker models from AHC labels ─────────────────────
    means = np.zeros((K, X.shape[1]))
    priors = np.zeros(K)
    for k in range(K):
        mask = init_labels == k
        count = mask.sum()
        if count > 0:
            means[k] = X[mask].mean(axis=0)
            nrm = np.linalg.norm(means[k])
            if nrm > 1e-10:
                means[k] /= nrm
            priors[k] = count / T

    # ── 4. HMM transition matrix (log-space) ─────────────────────────────
    log_trans = np.full((K, K), np.log(max((1.0 - loopP) / max(K - 1, 1),
                                           1e-20)))
    np.fill_diagonal(log_trans, np.log(loopP))
    log_pi = np.full(K, np.log(1.0 / K))

    # ── 5. VB iterations ─────────────────────────────────────────────────
    gamma = None
    for _it in range(n_iters):
        # Emission scores: (T, K) dot-product (= cosine for L2-normed)
        scores = X @ means.T
        log_emissions = Fa * scores + Fb * np.log(
            np.maximum(priors, 1e-20)
        )

        # Forward-backward
        gamma = _forward_backward(log_emissions, log_trans, log_pi)

        # Update speaker models
        new_means = np.zeros_like(means)
        new_priors = np.zeros(K)
        for k in range(K):
            w = gamma[:, k]
            total = w.sum()
            if total > 1e-10:
                new_means[k] = (w[:, None] * X).sum(axis=0) / total
                nrm = np.linalg.norm(new_means[k])
                if nrm > 1e-10:
                    new_means[k] /= nrm
                new_priors[k] = total / T

        # Eliminate dead speakers (prior ~ 0)
        alive = new_priors > 1e-4
        if alive.sum() < 1:
            break
        means = new_means
        priors = new_priors

    # ── 6. Final assignment ──────────────────────────────────────────────
    if gamma is None:
        labels = init_labels
    else:
        labels = gamma.argmax(axis=1)

    # Relabel to consecutive 0-based integers
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = [remap[l] for l in labels]
    return labels


# ---------------------------------------------------------------------------
# I/O helpers (same pattern as other clusterers)
# ---------------------------------------------------------------------------

def read_emb(scp):
    emb_dict = OrderedDict()
    for sub_seg_id, emb in kaldiio.load_scp_sequential(scp):
        utt = sub_seg_id.split("-")[0]
        if utt not in emb_dict:
            emb_dict[utt] = {"sub_seg": [], "embs": []}
        emb_dict[utt]["sub_seg"].append(sub_seg_id)
        emb_dict[utt]["embs"].append(emb)
    subsegs_list = []
    embeddings_list = []
    for utt, d in emb_dict.items():
        subsegs_list.append(d["sub_seg"])
        embeddings_list.append(np.stack(d["embs"]))
    return subsegs_list, embeddings_list


def get_args():
    p = argparse.ArgumentParser(
        description="VBx (Variational Bayes HMM) speaker clustering"
    )
    p.add_argument("--scp", required=True, help="embedding scp")
    p.add_argument("--output", required=True, help="output label file")
    p.add_argument(
        "--Fa", type=float, default=0.4,
        help="score scaling factor (default: 0.4)",
    )
    p.add_argument(
        "--Fb", type=float, default=17.0,
        help="speaker prior scaling factor (default: 17)",
    )
    p.add_argument(
        "--loopP", type=float, default=0.95,
        help="HMM self-loop probability (default: 0.95)",
    )
    p.add_argument(
        "--init-threshold", type=float, default=0.3, dest="init_threshold",
        help="AHC cosine-similarity threshold for initialisation (default: 0.3)",
    )
    p.add_argument(
        "--n-iters", type=int, default=10, dest="n_iters",
        help="number of VB iterations (default: 10)",
    )
    p.add_argument(
        "--plda-model", default=None, dest="plda_model",
        help="path to trained PLDA model (.h5); if omitted uses cosine scoring",
    )
    return p.parse_args()


def main():
    args = get_args()

    # Optionally load PLDA
    plda = None
    if args.plda_model:
        from wespeaker.utils.plda.two_cov_plda import TwoCovPLDA
        plda = TwoCovPLDA.load_model(args.plda_model)
        print(f"  VBx: loaded PLDA model from {args.plda_model}",
              file=sys.stderr)

    subsegs_list, embeddings_list = read_emb(args.scp)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    run_cluster = functools.partial(
        cluster,
        Fa=args.Fa,
        Fb=args.Fb,
        loopP=args.loopP,
        init_threshold=args.init_threshold,
        n_iters=args.n_iters,
        plda=plda,
    )

    print(f"  VBx: Fa={args.Fa} Fb={args.Fb} loopP={args.loopP} "
          f"init_threshold={args.init_threshold} n_iters={args.n_iters} "
          f"scoring={'plda' if plda else 'cosine'}",
          file=sys.stderr)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        with open(args.output, "w") as fd:
            for subsegs, labels in zip(
                subsegs_list, executor.map(run_cluster, embeddings_list)
            ):
                for subseg, label in zip(subsegs, labels):
                    print(subseg, label, file=fd)

    print(f"  VBx: wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

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
Fuse two embedding extractors at the **cosine similarity matrix** level, then
cluster with AHC or spectral (same post-processing as the single-system
clusterers).

Requires **aligned** sub-segments: same VAD speech segments and the same
sliding-window grid (`window_secs`, `period_secs`, `frame_shift`) in both
extractors. Keys are matched by **parent segment + frame span** (last two
``-XXXXXXXX-XXXXXXXX`` fields), so row **order** in ``emb.scp`` may differ.
Output labels use sub-segment IDs from ``--scp-a`` for ``make_rttm.py``.

**Fusion norms**

- ``linear`` (default): \(S = w_a C_1 + w_b C_2\) on raw cosines \(C_i\in[-1,1]\).
  Use an AHC ``--threshold`` tuned for this fused scale (often **not** the same
  as for a single extractor).
- ``sigmoid_avg``: \(S = w_a\,\sigma(k C_1) + w_b\,\sigma(k C_2)\) with \(\sigma\)
  the logistic function, \(S\in(0,1)\). Retune ``--threshold`` on dev (typical
  values differ from linear fusion).

Example:
  python3 wespeaker/diar/fused_sim_clusterer.py \\
      --scp-a exp/dev_funasr_fsmn_sad_embedding/emb.scp \\
      --scp-b exp/dev_funasr_fsmn_sad_embedding_w2vbert/emb.scp \\
      --output exp/fused_sim_cluster/dev_labels \\
      --clusterer ahc --weight-a 0.5 \\
      --threshold 0.21 --linkage average
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
import re
from collections import OrderedDict

import kaldiio
import numpy as np
import scipy.linalg
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster._kmeans import k_means

from wespeaker.diar.ahc_clusterer import _absorb_small_clusters
from wespeaker.utils.utils import validate_path

# Diarization embedding keys end with "-{8-digit}-{8-digit}" (frame indices
# within the VAD segment, or ms in the top-level segment id — we strip once from
# the right to get parent id + span for pairing).
_TRAILING_SPAN = re.compile(r"-(\d{8})-(\d{8})$")


def parent_and_frame_span(key: str):
    """Return (parent_key, frame_start, frame_end) for a sub-segment id."""
    m = _TRAILING_SPAN.search(key)
    if not m:
        raise ValueError(
            f"cannot parse trailing -XXXXXXXX-XXXXXXXX in key: {key!r}"
        )
    parent = key[: m.start()]
    if not parent:
        raise ValueError(f"empty parent after parsing key: {key!r}")
    return parent, int(m.group(1)), int(m.group(2))


def _mismatch_window_hint(
    span_a: tuple, span_to_emb_b: dict
) -> str:
    """If B uses a different sub-window length than A for the same VAD segment, explain."""
    parent_a, s0, s1 = span_a
    wa = s1 - s0
    wb = None
    for (pb, b0, b1) in span_to_emb_b.keys():
        if pb == parent_a:
            wb = b1 - b0
            break
    if wb is None or wa == wb or wa <= 0 or wb <= 0:
        return ""
    # frame_shift is 10 ms in VoxConverse v2 recipes
    sa_sec = wa * 10 / 1000.0
    sb_sec = wb * 10 / 1000.0
    return (
        f"Likely cause: sub-window length differs ({wa} frames ~{sa_sec:.2f}s in --scp-a vs "
        f"{wb} frames ~{sb_sec:.2f}s in --scp-b). Both run_updated.sh and run_w2vbert.sh "
        f"must use the same --window_secs and --period_secs (default recipe: 1.5 / 0.75). "
        f"Re-run embedding stage 6 for both extractors after aligning settings."
    )


def read_emb_by_utt(scp):
    emb_dict = OrderedDict()
    for sub_seg_id, emb in kaldiio.load_scp_sequential(scp):
        utt = sub_seg_id.split("-")[0]
        if utt not in emb_dict:
            emb_dict[utt] = {"sub_seg": [], "embs": []}
        emb_dict[utt]["sub_seg"].append(sub_seg_id)
        emb_dict[utt]["embs"].append(np.asarray(emb, dtype=np.float64))
    return emb_dict


def pair_embeddings(dict_a, dict_b):
    """Return parallel lists (subsegs, E1, E2) per utterance.

    Rows are paired by (parent_key, frame_start, frame_end), not by full string
    equality, so scp-b may list rows in a different order. Label ids follow scp-a.
    """
    keys_a = list(dict_a.keys())
    keys_b = set(dict_b.keys())
    if set(keys_a) != keys_b:
        missing_a = keys_b - set(keys_a)
        missing_b = set(keys_a) - keys_b
        raise ValueError(
            "Utterance sets differ between --scp-a and --scp-b. "
            f"only in b: {sorted(missing_a)[:5]}... "
            f"only in a: {sorted(missing_b)[:5]}..."
        )

    subsegs_list = []
    e1_list = []
    e2_list = []
    for utt in keys_a:
        sa = dict_a[utt]["sub_seg"]
        ea_raw = dict_a[utt]["embs"]
        sb = dict_b[utt]["sub_seg"]
        eb_raw = dict_b[utt]["embs"]

        span_to_emb_b = {}
        for kb, ebb in zip(sb, eb_raw):
            t = parent_and_frame_span(kb)
            if t in span_to_emb_b:
                raise ValueError(
                    f"{utt}: duplicate (parent,frame) span in --scp-b: {t!r}"
                )
            span_to_emb_b[t] = np.asarray(ebb, dtype=np.float64)

        eb_aligned = []
        spans_from_a = []
        for ka in sa:
            t = parent_and_frame_span(ka)
            spans_from_a.append(t)
            if t not in span_to_emb_b:
                only_a = {parent_and_frame_span(x) for x in sa}
                only_b = set(span_to_emb_b.keys())
                miss_a = sorted(only_a - only_b)[:3]
                miss_b = sorted(only_b - only_a)[:3]
                hint = _mismatch_window_hint(t, span_to_emb_b)
                raise ValueError(
                    f"{utt}: span {t!r} from --scp-a key {ka!r} not found in --scp-b. "
                    "The (speech_segment, frame_start, frame_end) multiset must match; "
                    "both extractors need the same window_secs, period_secs, and frame_shift, "
                    "then re-run embedding stage 6 for both. "
                    f"Example spans only in a: {miss_a}; only in b: {miss_b}. {hint}"
                )
            eb_aligned.append(span_to_emb_b[t])

        only_b = set(span_to_emb_b.keys()) - set(spans_from_a)
        if only_b:
            ex = next(iter(only_b))
            raise ValueError(
                f"{utt}: --scp-b has {len(only_b)} sub-segment span(s) not in --scp-a "
                f"(e.g. {ex!r}). Regenerate both embeddings with matching settings."
            )

        ea = np.stack([np.asarray(x, dtype=np.float64) for x in ea_raw])
        eb = np.stack(eb_aligned)
        if ea.shape[0] != eb.shape[0]:
            raise ValueError(f"{utt}: row count mismatch after span alignment")
        subsegs_list.append(sa)
        e1_list.append(ea)
        e2_list.append(eb)
    return subsegs_list, e1_list, e2_list


def _l2norm_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return x / norms


def _sigmoid(x):
    """Elementwise logistic; clips logits for stability."""
    z = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _zscore_affinity(s):
    """Z-score normalize off-diagonal elements of a similarity matrix per utterance.

    This aligns the score distributions of two embedding spaces before fusion,
    preventing one system's wider cosine range from dominating the weighted sum.
    Diagonal is restored to 1.0 after normalization.
    """
    n = s.shape[0]
    if n <= 2:
        return s
    mask = ~np.eye(n, dtype=bool)
    off = s[mask]
    mu, sigma = off.mean(), off.std()
    if sigma < 1e-8:
        return s
    s_norm = (s - mu) / sigma
    np.fill_diagonal(s_norm, 1.0)
    return s_norm


def fused_cosine_similarity(e1, e2, weight_a, fuse_norm="linear",
                            sigmoid_scale=3.0, zscore_norm=False):
    """Pairwise fused similarity for AHC/spectral.

    fuse_norm:
      linear: weighted average of raw cosines in [-1, 1].
      sigmoid_avg: weighted average of sigmoid(scale * cosine) per matrix, in (0,1).
      zscore: Z-score normalize each system's affinity matrix per utterance before
              fusing, then combine with weighted sum. Retune --threshold on dev
              (typical values are in a standardized score space, e.g. 0.0–1.5).

    zscore_norm: if True, Z-score normalize each affinity matrix before combining
                 (applies on top of any fuse_norm). Equivalent to fuse_norm=zscore.
    """
    w_b = 1.0 - weight_a
    n1 = _l2norm_rows(np.asarray(e1))
    n2 = _l2norm_rows(np.asarray(e2))
    s1 = np.dot(n1, n1.T)
    s2 = np.dot(n2, n2.T)
    if fuse_norm == "zscore":
        a1 = _zscore_affinity(s1)
        a2 = _zscore_affinity(s2)
        s = weight_a * a1 + w_b * a2
        np.fill_diagonal(s, 1.0)
    elif fuse_norm == "sigmoid_avg":
        a1 = _sigmoid(float(sigmoid_scale) * s1)
        a2 = _sigmoid(float(sigmoid_scale) * s2)
        s = weight_a * a1 + w_b * a2
    elif fuse_norm == "linear":
        if zscore_norm:
            s1 = _zscore_affinity(s1)
            s2 = _zscore_affinity(s2)
        s = weight_a * s1 + w_b * s2
        np.clip(s, -1.0, 1.0, out=s)
    else:
        raise ValueError(f"unknown fuse_norm: {fuse_norm!r}")
    np.fill_diagonal(s, 1.0)
    s = (s + s.T) / 2.0
    return s, n1, n2


def symmetric_absorb_embedding(n1, n2):
    """L2-normalized sum of per-system normalized rows (for AHC small-cluster absorb)."""
    t = n1 + n2
    return _l2norm_rows(t)


def cluster_ahc_fused(e1, e2, weight_a, threshold, linkage_method,
                      min_cluster_size, fuse_norm="linear",
                      sigmoid_scale=3.0, zscore_norm=False):
    cos_sim, n1, n2 = fused_cosine_similarity(
        e1, e2, weight_a, fuse_norm=fuse_norm, sigmoid_scale=sigmoid_scale,
        zscore_norm=zscore_norm,
    )
    emb_sym = symmetric_absorb_embedding(n1, n2)
    return _ahc_from_cos_sim(
        cos_sim, emb_sym, threshold, linkage_method, min_cluster_size
    )


def _ahc_from_cos_sim(cos_sim, emb_norm_aux, threshold, linkage_method,
                      min_cluster_size):
    n = cos_sim.shape[0]
    if n <= 1:
        return [0] * n
    if n == 2:
        sim = cos_sim[0, 1]
        return [0, 0] if sim >= threshold else [0, 1]

    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    cos_dist = (cos_dist + cos_dist.T) / 2.0
    np.maximum(cos_dist, 0.0, out=cos_dist)
    condensed = squareform(cos_dist, checks=False)
    z = linkage(condensed, method=linkage_method)
    dist_threshold = 1.0 - threshold
    labels = fcluster(z, t=dist_threshold, criterion="distance") - 1

    if min_cluster_size > 1:
        labels = _absorb_small_clusters(
            labels, emb_norm_aux, min_cluster_size
        )
    return labels.tolist()


def cluster_spectral_fused(e1, e2, weight_a, p, num_spks, min_num_spks,
                           max_num_spks, fuse_norm="linear",
                           sigmoid_scale=3.0, zscore_norm=False):
    cos_sim, _, _ = fused_cosine_similarity(
        e1, e2, weight_a, fuse_norm=fuse_norm, sigmoid_scale=sigmoid_scale,
        zscore_norm=zscore_norm,
    )
    if fuse_norm == "sigmoid_avg":
        similarity_matrix = cos_sim
    else:
        # Match spectral_clusterer: affinity in [0, 1]
        similarity_matrix = 0.5 * (1.0 + cos_sim)
    return _spectral_from_affinity(
        similarity_matrix, p, num_spks, min_num_spks, max_num_spks
    )


def _spectral_from_affinity(similarity_matrix, p, num_spks, min_num_spks,
                            max_num_spks):
    def prune(m, p_val):
        m = np.array(m, copy=True)
        rows = m.shape[0]
        if rows < 1000:
            n_keep = max(rows - 10, 2)
        else:
            n_keep = int((1.0 - p_val) * rows)
        for i in range(rows):
            indexes = np.argsort(m[i, :])
            low_indexes, high_indexes = indexes[0:n_keep], indexes[n_keep:rows]
            m[i, low_indexes] = 0.0
            m[i, high_indexes] = 1.0
        return 0.5 * (m + m.T)

    def laplacian(m):
        m = np.array(m, copy=True)
        m[np.diag_indices(m.shape[0])] = 0.0
        d = np.diag(np.sum(np.abs(m), axis=1))
        return d - m

    def spectral_step(m, n_spk, min_spk, max_spk):
        eig_values, eig_vectors = scipy.linalg.eigh(m)
        n_spk = (
            n_spk
            if n_spk is not None
            else np.argmax(np.diff(eig_values[: max_spk + 1])) + 1
        )
        n_spk = max(n_spk, min_spk)
        return eig_vectors[:, :n_spk]

    sim = np.asarray(similarity_matrix, dtype=np.float64)
    n = sim.shape[0]
    if n <= 2:
        return [0] * n

    pruned = prune(sim, p)
    lap = laplacian(pruned)
    spec_emb = spectral_step(lap, num_spks, min_num_spks, max_num_spks)
    k = spec_emb.shape[1]
    _, labels, _ = k_means(spec_emb, k, random_state=None, n_init=10)
    return labels.tolist()


def get_args():
    p = argparse.ArgumentParser(
        description="Diarization clustering with fused cosine similarity "
        "from two embedding scp files (AHC or spectral)."
    )
    p.add_argument("--scp-a", required=True, help="first embedding.scp")
    p.add_argument("--scp-b", required=True, help="second embedding.scp")
    p.add_argument("--output", required=True, help="output label file")
    p.add_argument(
        "--clusterer",
        choices=["ahc", "spectral"],
        default="ahc",
        help="clustering after similarity fusion (default: ahc)",
    )
    p.add_argument(
        "--weight-a",
        type=float,
        default=0.5,
        help="weight for --scp-a cosine matrix; (1-weight-a) for --scp-b "
        "(default: 0.5 = average)",
    )
    p.add_argument(
        "--fuse-norm",
        choices=["linear", "sigmoid_avg", "zscore"],
        default="linear",
        dest="fuse_norm",
        help="linear: average raw cosines [-1,1]. sigmoid_avg: average "
        "sigmoid(scale*cosine) per model, then combine (retune --threshold). "
        "zscore: Z-score normalize each system's affinity per utterance before "
        "combining — best when systems have very different cosine distributions "
        "(retune --threshold, typical range 0.0–1.5).",
    )
    p.add_argument(
        "--zscore-norm",
        action="store_true",
        default=False,
        dest="zscore_norm",
        help="Apply per-utterance Z-score normalization to each affinity matrix "
        "before fusing (can combine with any --fuse-norm). Helps when ResNet and "
        "w2v-BERT cosine distributions differ significantly.",
    )
    p.add_argument(
        "--sigmoid-scale",
        type=float,
        default=3.0,
        dest="sigmoid_scale",
        help="logit scale k for sigmoid_avg: sigma(k*cosine); default 3.0",
    )
    # AHC
    p.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="AHC cosine similarity threshold (same as ahc_clusterer.py)",
    )
    p.add_argument(
        "--linkage",
        default="average",
        choices=["average", "complete", "single"],
        help="AHC linkage",
    )
    p.add_argument(
        "--min_cluster_size",
        type=int,
        default=1,
        help="AHC: absorb clusters smaller than this",
    )
    # Spectral
    p.add_argument(
        "--spectral-prune-p",
        type=float,
        default=0.01,
        dest="spectral_prune_p",
        help="Spectral: prune fraction p (same role as spectral_clusterer)",
    )
    p.add_argument("--num-spks", type=int, default=None, dest="num_spks")
    p.add_argument("--min-num-spks", type=int, default=1, dest="min_num_spks")
    p.add_argument("--max-num-spks", type=int, default=20, dest="max_num_spks")
    args = p.parse_args()
    if not (0.0 <= args.weight_a <= 1.0):
        p.error("--weight-a must be in [0, 1]")
    return args


def main():
    args = get_args()
    dict_a = read_emb_by_utt(args.scp_a)
    dict_b = read_emb_by_utt(args.scp_b)
    subsegs_list, e1_list, e2_list = pair_embeddings(dict_a, dict_b)

    if args.clusterer == "ahc":
        runner = functools.partial(
            cluster_ahc_fused,
            weight_a=args.weight_a,
            threshold=args.threshold,
            linkage_method=args.linkage,
            min_cluster_size=args.min_cluster_size,
            fuse_norm=args.fuse_norm,
            sigmoid_scale=args.sigmoid_scale,
            zscore_norm=args.zscore_norm,
        )
    else:
        runner = functools.partial(
            cluster_spectral_fused,
            weight_a=args.weight_a,
            p=args.spectral_prune_p,
            num_spks=args.num_spks,
            min_num_spks=args.min_num_spks,
            max_num_spks=args.max_num_spks,
            fuse_norm=args.fuse_norm,
            sigmoid_scale=args.sigmoid_scale,
            zscore_norm=args.zscore_norm,
        )

    validate_path(args.output)

    with concurrent.futures.ProcessPoolExecutor() as ex, open(
        args.output, "w"
    ) as fd:
        for subsegs, labels in zip(
            subsegs_list, ex.map(runner, e1_list, e2_list)
        ):
            for subseg, lab in zip(subsegs, labels):
                print(subseg, lab, file=fd)


if __name__ == "__main__":
    main()

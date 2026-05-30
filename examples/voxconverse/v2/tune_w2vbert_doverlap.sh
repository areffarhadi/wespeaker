#!/bin/bash
# tune_w2vbert_doverlap.sh
#
# Sweeps DOVER-Lap (and individual clusterer) hyperparameters for w2v-BERT embeddings
# on the DEV set. Runs only stages 7–9 (clustering → RTTM → DER).
#
# Phase 1: sweep merge_cutoff for UMAP alone
# Phase 2: sweep ahc_threshold for AHC alone
# Phase 3: grid-search DOVER-Lap (gaussian_std × dover_weight) using best UMAP+AHC params
#
# Usage:  bash tune_w2vbert_doverlap.sh
#         bash tune_w2vbert_doverlap.sh --sad_type system  (default)
#         bash tune_w2vbert_doverlap.sh --sad_type pyannote

. ./path.sh || exit 1

PYTHON=/home/aref.farhadipour/wespeaker/.venv/bin/python3
DOVER_LAP=/home/aref.farhadipour/wespeaker/.venv/bin/dover-lap
export PYTHONPATH=../../../:$PYTHONPATH

sad_type="system"

. tools/parse_options.sh

partition="test"
W2V="_w2vbert"
emb_scp="exp/${partition}_${sad_type}_sad_embedding${W2V}/emb.scp"
ref_dir="data/voxconverse-master"
MD_EVAL="external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl"

# ── sanity checks ────────────────────────────────────────────────────────────
if [ ! -s "${emb_scp}" ]; then
    echo "ERROR: embeddings not found: ${emb_scp}"
    echo "  Run run_w2vbert.sh --stage 4 --stop_stage 6 --partition dev first."
    exit 1
fi
if [ ! -f "${MD_EVAL}" ]; then
    echo "ERROR: md-eval not found: ${MD_EVAL}. Run stage 1 of run_w2vbert.sh."
    exit 1
fi

results_dir="exp/tune_w2vbert_doverlap"
mkdir -p "${results_dir}"
summary="${results_dir}/summary.txt"
> "${summary}"

# ── helper: run md-eval and return DER ──────────────────────────────────────
run_eval() {
    local rttm="$1"
    perl "${MD_EVAL}" -c 0.25 \
        -r <(cat "${ref_dir}/${partition}"/*.rttm) \
        -s "${rttm}" 2>/dev/null \
        | grep 'OVERALL SPEAKER DIARIZATION ERROR' \
        | grep -oP '[\d.]+(?= percent)' \
        || echo "999"
}

# ── helper: make RTTM from labels ───────────────────────────────────────────
make_rttm() {
    local labels="$1" rttm="$2"
    $PYTHON wespeaker/diar/make_rttm.py --labels "${labels}" --channel 1 > "${rttm}"
}

echo "============================================================"
echo " W2V-BERT DOVER-Lap hyper-parameter sweep (partition=dev)"
echo " sad_type=${sad_type}   emb=${emb_scp}"
echo "============================================================"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# Phase 1: UMAP  — sweep merge_cutoff
# ────────────────────────────────────────────────────────────────────────────
echo "── Phase 1: UMAP merge_cutoff sweep ──────────────────────────"
printf "%-15s  %s\n" "merge_cutoff" "DER(%)"
printf "%-15s  %s\n" "------------" "------"

best_umap_mc=0.3
best_umap_der=999

mkdir -p exp/umap_cluster

for mc in 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45; do
    tag="umap_mc${mc}"
    labels="${results_dir}/${tag}_labels"
    rttm="${results_dir}/${tag}_rttm"

    $PYTHON wespeaker/diar/umap_clusterer.py \
        --scp "${emb_scp}" \
        --output "${labels}" \
        --merge_cutoff "${mc}" 2>/dev/null

    make_rttm "${labels}" "${rttm}"
    der=$(run_eval "${rttm}")

    printf "%-15s  %s\n" "${mc}" "${der}"
    echo "umap  merge_cutoff=${mc}  DER=${der}" >> "${summary}"

    if (( $(echo "${der} < ${best_umap_der}" | bc -l) )); then
        best_umap_der="${der}"
        best_umap_mc="${mc}"
    fi
done
echo "  → Best UMAP: merge_cutoff=${best_umap_mc}  DER=${best_umap_der}%"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# Phase 2: AHC  — sweep ahc_threshold
# ────────────────────────────────────────────────────────────────────────────
echo "── Phase 2: AHC threshold sweep ──────────────────────────────"
printf "%-15s  %s\n" "ahc_threshold" "DER(%)"
printf "%-15s  %s\n" "-------------" "------"

best_ahc_thr=0.3
best_ahc_der=999

mkdir -p exp/ahc_cluster

for thr in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
    tag="ahc_thr${thr}"
    labels="${results_dir}/${tag}_labels"
    rttm="${results_dir}/${tag}_rttm"

    $PYTHON wespeaker/diar/ahc_clusterer.py \
        --scp "${emb_scp}" \
        --output "${labels}" \
        --threshold "${thr}" \
        --linkage average 2>/dev/null

    make_rttm "${labels}" "${rttm}"
    der=$(run_eval "${rttm}")

    printf "%-15s  %s\n" "${thr}" "${der}"
    echo "ahc   threshold=${thr}  DER=${der}" >> "${summary}"

    if (( $(echo "${der} < ${best_ahc_der}" | bc -l) )); then
        best_ahc_der="${der}"
        best_ahc_thr="${thr}"
    fi
done
echo "  → Best AHC: threshold=${best_ahc_thr}  DER=${best_ahc_der}%"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# Phase 3: DOVER-Lap grid search  (gaussian_std × dover_weight)
# Using best UMAP and AHC params found above; spectral uses its defaults.
# ────────────────────────────────────────────────────────────────────────────
echo "── Phase 3: DOVER-Lap grid search ────────────────────────────"
echo "  UMAP merge_cutoff=${best_umap_mc}  AHC threshold=${best_ahc_thr}"
echo ""

# Pre-compute the three base RTTMs once (they don't change across the grid)
labels_suffix="${partition}_${sad_type}_sad${W2V}_labels"
rttm_suffix="${partition}_${sad_type}_sad${W2V}_rttm"

mkdir -p exp/umap_cluster exp/ahc_cluster exp/spectral_cluster exp/doverlap_cluster

echo "  Recomputing UMAP (mc=${best_umap_mc}) ..."
$PYTHON wespeaker/diar/umap_clusterer.py \
    --scp "${emb_scp}" \
    --output "exp/umap_cluster/${labels_suffix}" \
    --merge_cutoff "${best_umap_mc}" 2>/dev/null
make_rttm "exp/umap_cluster/${labels_suffix}" "exp/umap_cluster/${rttm_suffix}"

echo "  Recomputing AHC (thr=${best_ahc_thr}) ..."
$PYTHON wespeaker/diar/ahc_clusterer.py \
    --scp "${emb_scp}" \
    --output "exp/ahc_cluster/${labels_suffix}" \
    --threshold "${best_ahc_thr}" --linkage average 2>/dev/null
make_rttm "exp/ahc_cluster/${labels_suffix}" "exp/ahc_cluster/${rttm_suffix}"

echo "  Recomputing Spectral ..."
$PYTHON wespeaker/diar/spectral_clusterer.py \
    --scp "${emb_scp}" \
    --output "exp/spectral_cluster/${labels_suffix}" 2>/dev/null
make_rttm "exp/spectral_cluster/${labels_suffix}" "exp/spectral_cluster/${rttm_suffix}"

echo ""
printf "%-10s  %-12s  %s\n" "gstd" "dover_weight" "DER(%)"
printf "%-10s  %-12s  %s\n" "----" "------------" "------"

best_gstd=0.5
best_dw=0.05
best_dl_der=999

for gstd in 0.01 0.1 0.3 0.5 1.0; do
    for dw in 0.01 0.05 0.1 0.2 0.3; do
        tag="dl_gstd${gstd}_dw${dw}"
        dl_rttm="${results_dir}/${tag}_rttm"

        $DOVER_LAP \
            "${dl_rttm}" \
            "exp/umap_cluster/${rttm_suffix}" \
            "exp/ahc_cluster/${rttm_suffix}" \
            "exp/spectral_cluster/${rttm_suffix}" \
            --weight-type rank \
            --gaussian-filter-std "${gstd}" \
            --dover-weight "${dw}" \
            2>/dev/null

        der=$(run_eval "${dl_rttm}")

        printf "%-10s  %-12s  %s\n" "${gstd}" "${dw}" "${der}"
        echo "doverlap  gstd=${gstd}  dover_weight=${dw}  DER=${der}" >> "${summary}"

        if (( $(echo "${der} < ${best_dl_der}" | bc -l) )); then
            best_dl_der="${der}"
            best_gstd="${gstd}"
            best_dw="${dw}"
        fi
    done
done

echo ""
echo "  → Best DOVER-Lap: gstd=${best_gstd}  dover_weight=${best_dw}  DER=${best_dl_der}%"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# Final summary
# ────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " SWEEP COMPLETE"
echo "------------------------------------------------------------"
echo " Best UMAP alone  :  merge_cutoff=${best_umap_mc}   DER=${best_umap_der}%"
echo " Best AHC alone   :  threshold=${best_ahc_thr}       DER=${best_ahc_der}%"
echo " Best DOVER-Lap   :  mc=${best_umap_mc}  thr=${best_ahc_thr}  gstd=${best_gstd}  dw=${best_dw}   DER=${best_dl_der}%"
echo "------------------------------------------------------------"
echo " To apply best DOVER-Lap params in run_w2vbert.sh, set:"
echo "   cluster_type=doverlap"
echo "   merge_cutoff=${best_umap_mc}"
echo "   ahc_threshold=${best_ahc_thr}"
echo "   doverlap_gaussian_std=${best_gstd}"
echo "   doverlap_dover_weight=${best_dw}"
echo "============================================================"

echo "" >> "${summary}"
echo "BEST UMAP:      merge_cutoff=${best_umap_mc}  DER=${best_umap_der}" >> "${summary}"
echo "BEST AHC:       threshold=${best_ahc_thr}  DER=${best_ahc_der}" >> "${summary}"
echo "BEST DOVERLAP:  mc=${best_umap_mc} thr=${best_ahc_thr} gstd=${best_gstd} dw=${best_dw}  DER=${best_dl_der}" >> "${summary}"
echo "Full results: ${summary}"

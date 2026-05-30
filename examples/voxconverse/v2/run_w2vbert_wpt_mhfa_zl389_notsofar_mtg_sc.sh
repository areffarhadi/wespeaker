#!/bin/bash
# Copyright (c) 2022-2023 Xu Xiang
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2024 Hongji Wang (jijijiang77@gmail.com)
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
#
# -----------------------------------------------------------------------------
# NOTSOFAR MTG — WPT + w2v-BERT-2.0 + MHFA (USM_FTcode zl389 recipe)
#
# Stage 2 builds data/${partition}/wav.scp from NOTSOFAR eval tree: only single-channel
# distant mics sc_meetup_0, sc_plaza_0, sc_rockfall_{0,1,2}, sc_studio_0 — each ch0.wav.
# Reference RTTMs come from gt_transcription.json (see local/prepare_notsofar_mtg_sc.py).
# Stages 3–9 match run_w2vbert_wpt_mhfa_zl389.sh (Demucs optional, VAD…DER).
#
# Outputs use *_w2vbert_wptmhfa_zl389_ns so they do not clash with VoxConverse runs.
# -----------------------------------------------------------------------------

. ./path.sh || exit 1

# Prefer repo .venv (FunASR, torch); override with PYTHON=/path/to/python.
_ws_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"
if [ -z "${PYTHON}" ] && [ -x "${_ws_root}/.venv/bin/python" ]; then
    PYTHON="${_ws_root}/.venv/bin/python"
    export PATH="${_ws_root}/.venv/bin:${PATH}"
else
    PYTHON="${PYTHON:-python3}"
fi

# NOTSOFAR eval root (must contain MTG/MTG_* / … / sc_*/ch0.wav)
NOTSOFAR_EVAL_ROOT="${NOTSOFAR_EVAL_ROOT:-$HOME/DATASETS/NOTSOFAR/benchmark-datasets/eval_set/240825.1_eval_full_with_GT}"
# Where reference RTTMs are written (stage 2): ${NOTSOFAR_REF_ROOT}/${partition}/*.rttm
NOTSOFAR_REF_ROOT="${NOTSOFAR_REF_ROOT:-data/notsofar_master}"

stage=4
stop_stage=9
partition="notsofar"
subseg_cmn=true
get_each_file_res=1
skip_download_if_present=true

# ── VAD (same as run_updated.sh) ─────────────────────────────────────
sad_type="funasr_fsmn"       # oracle / system (Silero) / pyannote / funasr_fsmn
pyannote_device="${PYANNOTE_DEVICE:-cpu}"
pyannote_onset=0.5
pyannote_offset=0.5
pyannote_nj=16
funasr_hub="${FUNASR_HUB:-hf}"
funasr_revision="${FUNASR_REVISION:-v2.0.4}"
funasr_device="${FUNASR_DEVICE:-cpu}"
funasr_nj="${FUNASR_NJ:-4}"

# ── Demucs (optional) ────────────────────────────────────────────────
use_demucs=false
demucs_device="${DEMUCS_DEVICE:-cuda}"
demucs_model="${DEMUCS_MODEL:-htdemucs}"

# ── WPT + MHFA + w2v-BERT (vendored under ./wpt_mhfa_zl389/) ──
_v2_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# Python import path (main_train_*.py + losses.py). Override only if you use an external copy.
USM_FTCODE="${USM_FTCODE:-$_v2_dir/wpt_mhfa_zl389}"
# Default training --out_fold (h8/c128/fixed24000 from train_simple_sv_wpt_w2vbert_mhfa_zl389.sh); not inside wespeaker.
WPT_MHFA_CKPT_DIR="${WPT_MHFA_CKPT_DIR:-$HOME/Encode-explore/USM_FTcode/ckpt_asv/simple_sv_wpt_w2vbert_mhfa_zl389_h8_c128_fixed24000}"
#WPT_MHFA_CKPT_DIR="${WPT_MHFA_CKPT_DIR:-$HOME/Encode-explore/USM_FTcode/ckpt_asv/simple_sv_wpt_w2vbert_zl389_v2_adim128_fixed24000_gpu_mel}"
WPT_MHFA_CHECKPOINT_NAME="${WPT_MHFA_CHECKPOINT_NAME:-best_model.pt}"
# Embedding extraction uses WeSpeaker PYTHON (above): it has kaldiio. USM enc-env often does not.
# Set only if you need a different interpreter (install kaldiio there: pip install kaldiio).
USM_WPT_MHFA_PYTHON="${USM_WPT_MHFA_PYTHON:-}"
# Stage 6: larger batch uses more VRAM but speeds up W2V-BERT forward (try 24/32 if memory allows).
emb_batch_size="${EMB_BATCH_SIZE:-32}"
# Split wav.scp across nj processes. Each job loads the full model — use nj>1 only with
# multiple GPUs (set CUDA_VISIBLE_DEVICES per worker) or expect OOM on a single GPU.
emb_nj="${EMB_NJ:-1}"
emb_device="${W2VBERT_EMB_DEVICE:-cuda}"
# Sliding window for sub-segments (must match run_updated.sh for fused_sim)
frame_shift=10
window_secs=1.5
period_secs=0.5

# ── Clustering (same as run_updated.sh) ───────────────────────────────
cluster_type="umap"   # spectral / umap / ahc / doverlap
# UMAP + HDBSCAN + PAHC (wespeaker/diar/umap_clusterer.py) — defaults from exp/umap_hparam_search/results.csv best DER
merge_cutoff="${UMAP_MERGE_CUTOFF:-0.15}"
umap_n_neighbors="${UMAP_N_NEIGHBORS:-8}"
umap_min_dist="${UMAP_MIN_DIST:-0.05}"
# -1 = automatic n_components=min(32, N-2) per file
umap_n_components="${UMAP_N_COMPONENTS:--1}"
hdbscan_min_cluster_size="${HDBSCAN_MIN_CLUSTER_SIZE:-4}"
pahc_min_cluster_size="${PAHC_MIN_CLUSTER_SIZE:-3}"
pahc_absorb_cutoff="${PAHC_ABSORB_CUTOFF:-0.0}"
ahc_threshold=0.21
ahc_linkage="average"
doverlap_weight_type="rank"
doverlap_custom_weights=""
doverlap_gaussian_std=0.5
doverlap_dover_weight=0.05

# ── Overlap (same as run_updated.sh) ──────────────────────────────────
use_overlap=false
overlap_device="${OVERLAP_DEVICE:-cpu}"
overlap_min_dur=0.1
overlap_nj=16

# ── Misc ────────────────────────────────────────────────────────────
verbose=true
bash_trace=false

# Suffix for all embedding artifact names (labels/rttm/res)
W2V="_w2vbert_wptmhfa_zl389_ns"

help_message="Usage: $0 [options]
NOTSOFAR MTG — WPT+MHFA+w2v-BERT (USM_FTcode zl389), sc_* single-channel ch0 only.

Stages: 1=SCTK 2=NOTSOFAR wav.scp+refs 3=Demucs 4=VAD 5=fbank 6=embeddings 7=cluster 8=RTTM 9=DER

  --sad_type system|pyannote|oracle|funasr_fsmn   (oracle uses refs from stage 2)
  --cluster_type spectral|umap|ahc|doverlap
  --use_demucs true|false
  --use_overlap true|false
  --verbose true|false
  --bash_trace true|false
  --frame_shift 10            must match run_updated / make_fbank (ms)
  --window_secs 1.5           sub-segment window (match run_updated for fusion)
  --period_secs 0.75          sub-segment hop (seconds)
  --skip_download_if_present true|false  skip stages 1–2 when SCTK/data exist (default: true)

Env: NOTSOFAR_EVAL_ROOT (MTG/ root), NOTSOFAR_REF_ROOT (default data/notsofar_master; RTTMs under \${NOTSOFAR_REF_ROOT}/notsofar/),
     USM_FTCODE, WPT_MHFA_CKPT_DIR, WPT_MHFA_CHECKPOINT_NAME, USM_WPT_MHFA_PYTHON,
     W2VBERT_EMB_DEVICE, EMB_BATCH_SIZE, EMB_NJ, HF_TOKEN (PyAnnote), DEMUCS_DEVICE, PYANNOTE_DEVICE,
     FUNASR_HUB, FUNASR_REVISION, FUNASR_DEVICE, FUNASR_NJ (funasr_fsmn)
UMAP (when --cluster_type umap or doverlap): UMAP_MERGE_CUTOFF (default 0.15), UMAP_N_NEIGHBORS (default 8), UMAP_MIN_DIST, UMAP_N_COMPONENTS (-1=auto),
     HDBSCAN_MIN_CLUSTER_SIZE, PAHC_MIN_CLUSTER_SIZE, PAHC_ABSORB_CUTOFF — see wespeaker/diar/umap_clusterer.py; baked-in defaults from exp/umap_hparam_search/results.csv

Note: Stage 2 runs local/prepare_notsofar_mtg_sc.py. Default partition name is notsofar (data/notsofar/wav.scp)."

. tools/parse_options.sh

if [ "$bash_trace" = true ]; then
    set -x
fi

log_file="run_w2vbert_wpt_mhfa_zl389_notsofar_mtg_sc_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Full log: $log_file"
echo ""
echo "NOTSOFAR_EVAL_ROOT=${NOTSOFAR_EVAL_ROOT}"
echo "NOTSOFAR_REF_ROOT=${NOTSOFAR_REF_ROOT} (RTTMs: \${NOTSOFAR_REF_ROOT}/${partition}/)"
echo "partition=${partition}  |  W2V suffix=${W2V}"
if [ "${cluster_type}" = "umap" ] || [ "${cluster_type}" = "doverlap" ]; then
    echo "UMAP: merge_cutoff=${merge_cutoff} n_neighbors=${umap_n_neighbors} min_dist=${umap_min_dist} n_components=${umap_n_components}"
    echo "      HDBSCAN min_cluster_size=${hdbscan_min_cluster_size} | PAHC min_cluster_size=${pahc_min_cluster_size} absorb=${pahc_absorb_cutoff}"
fi
echo ""

SECONDS=0
declare -A stage_status

stage_banner() {
    echo ""
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
    echo ""
}

stage_done() {
    local snum="$1" msg="$2"
    stage_status[$snum]="$msg"
}

print_summary() {
    echo ""
    echo "================================================================================"
    echo "  PIPELINE SUMMARY (NOTSOFAR MTG sc_*, WPT+MHFA+w2v-BERT zl389)"
    echo "================================================================================"
    for snum in $(printf '%s\n' "${!stage_status[@]}" | sort -n); do
        echo "  Stage ${snum}: ${stage_status[$snum]}"
    done
    echo "--------------------------------------------------------------------------------"
    echo "  Elapsed: ${SECONDS}s | Log: $log_file"
    echo "================================================================================"
}

extract_zip() {
    "${PYTHON}" local/extract_zip.py "$1" "$2" || exit 1
}

resolve_wav_scp() {
    if [ "$use_demucs" = true ]; then
        wav_scp="data/${partition}/wav_demucs.scp"
        if [ ! -s "$wav_scp" ]; then
            echo "$0: expected ${wav_scp} (run stage 3 with --use_demucs true first)." >&2
            exit 1
        fi
    else
        wav_scp="data/${partition}/wav.scp"
    fi
}

# --- Stage 1: SCTK only (no ResNet ONNX — w2v-BERT does not use it) ---
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    stage_banner "Stage 1: SCTK (md-eval)"
    if [ "${skip_download_if_present}" = true ] \
            && [ -f external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl ]; then
        echo "$0: Stage 1: SCTK already present (skip_download_if_present=true)."
        stage_done 1 "SCTK (existing)"
    else
        mkdir -p external_tools
        wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip -O external_tools/SCTK-v2.4.12.zip
        extract_zip external_tools/SCTK-v2.4.12.zip external_tools
        stage_done 1 "SCTK ready"
    fi
fi

# --- Stage 2: NOTSOFAR MTG (sc_* ch0) + reference RTTMs ---
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    stage_banner "Stage 2: NOTSOFAR MTG (single-channel sc_*) + reference RTTMs"
    if [ ! -d "${NOTSOFAR_EVAL_ROOT}/MTG" ]; then
        echo "$0: NOTSOFAR_EVAL_ROOT has no MTG/: ${NOTSOFAR_EVAL_ROOT}" >&2
        exit 1
    fi
    REF_PARTITION_DIR="${NOTSOFAR_REF_ROOT}/${partition}"
    mkdir -p "data/${partition}"
    mkdir -p "${REF_PARTITION_DIR}"
    if [ "${skip_download_if_present}" = true ] \
            && [ -s "data/${partition}/wav.scp" ] \
            && compgen -G "${REF_PARTITION_DIR}/*.rttm" > /dev/null; then
        n_utts=$(wc -l < "data/${partition}/wav.scp")
        echo "$0: Stage 2: NOTSOFAR wav.scp + refs already present (${n_utts} utts, skip)."
        stage_done 2 "NOTSOFAR data (existing, ${n_utts} utts)"
    else
        "${PYTHON}" local/prepare_notsofar_mtg_sc.py \
                --eval-root "${NOTSOFAR_EVAL_ROOT}" \
                --out-wav-scp "data/${partition}/wav.scp" \
                --out-ref-dir "${REF_PARTITION_DIR}" \
                || exit 1
        n_utts=$(wc -l < "data/${partition}/wav.scp")
        if [ "${n_utts}" -eq 0 ]; then
            echo "$0: no utterances prepared (check NOTSOFAR_EVAL_ROOT and sc_* folders)." >&2
            exit 1
        fi
        stage_done 2 "NOTSOFAR prepared (${n_utts} sc_* ch0 utterances)"
    fi
fi

# --- Stage 3: Demucs ---
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$use_demucs" = true ]; then
        stage_banner "Stage 3: Demucs (${demucs_model})"
        demucs_vocals_dir="data/${partition}/demucs_vocals"
        rm -rf "${demucs_vocals_dir}" 2>/dev/null
        mkdir -p "${demucs_vocals_dir}"
        "${PYTHON}" wespeaker/diar/demucs_vocals.py \
                --scp "data/${partition}/wav.scp" \
                --out-dir "${demucs_vocals_dir}" \
                --wav-scp-out "data/${partition}/wav_demucs.scp" \
                --model "${demucs_model}" \
                --device "${demucs_device}" || exit 1
        stage_done 3 "Demucs -> data/${partition}/wav_demucs.scp"
    else
        echo "Stage 3: Demucs skipped."
    fi
fi

# --- Stage 4: VAD ---
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    resolve_wav_scp
    stage_banner "Stage 4: VAD (${sad_type})"
    min_duration=0.255

    if [[ "x${sad_type}" == "xoracle" ]]; then
        while read -r utt wav_path; do
            "${PYTHON}" wespeaker/diar/make_oracle_sad.py \
                    --rttm "${NOTSOFAR_REF_ROOT}/${partition}/${utt}.rttm" \
                    --min-duration $min_duration
        done < "${wav_scp}" > data/${partition}/oracle_sad
        sad_lines=$(wc -l < "data/${partition}/oracle_sad")
    fi

    if [[ "x${sad_type}" == "xsystem" ]]; then
       if [ "$verbose" = true ]; then
           "${PYTHON}" wespeaker/diar/make_system_sad.py \
                   --scp "${wav_scp}" \
                   --min-duration $min_duration | tee "data/${partition}/system_sad"
       else
           "${PYTHON}" wespeaker/diar/make_system_sad.py \
                   --scp "${wav_scp}" \
                   --min-duration $min_duration > "data/${partition}/system_sad" 2>/dev/null
       fi
       sad_lines=$(wc -l < "data/${partition}/system_sad")
       echo "System SAD: ${sad_lines} segments"
       if [ "$sad_lines" -eq 0 ]; then
           echo "$0: system_sad is empty." ; exit 1
       fi
    fi

    if [[ "x${sad_type}" == "xpyannote" ]]; then
       echo "PyAnnote VAD (device=${pyannote_device}, nj=${pyannote_nj}) ..."
       "${PYTHON}" wespeaker/diar/make_pyannote_sad.py \
               --scp "${wav_scp}" \
               --min-duration $min_duration \
               --onset ${pyannote_onset} \
               --offset ${pyannote_offset} \
               --device ${pyannote_device} \
               --nj ${pyannote_nj} > data/${partition}/pyannote_sad || exit 1
       sad_lines=$(wc -l < "data/${partition}/pyannote_sad")
       echo "PyAnnote SAD: ${sad_lines} segments"
       if [ "$sad_lines" -eq 0 ]; then
           echo "$0: pyannote_sad empty — check HF_TOKEN / models." ; exit 1
       fi
    fi

    if [[ "x${sad_type}" == "xfunasr_fsmn" ]]; then
       echo "FunASR FSMN-VAD (hub=${funasr_hub}, device=${funasr_device}, nj=${funasr_nj}) ..."
       "${PYTHON}" wespeaker/diar/make_funasr_fsmn_sad.py \
               --scp "${wav_scp}" \
               --min-duration $min_duration \
               --hub "${funasr_hub}" \
               --model-revision "${funasr_revision}" \
               --device "${funasr_device}" \
               --nj "${funasr_nj}" > "data/${partition}/funasr_fsmn_sad" || exit 1
       sad_lines=$(wc -l < "data/${partition}/funasr_fsmn_sad")
       echo "FunASR FSMN SAD: ${sad_lines} segments"
       if [ "$sad_lines" -eq 0 ]; then
           echo "$0: funasr_fsmn_sad empty — install funasr in WeSpeaker .venv / check wav.scp." ; exit 1
       fi
    fi

    stage_done 4 "VAD (${sad_type}, ${sad_lines:-?} segments)"
fi

# --- Stage 5: Fbank ---
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    resolve_wav_scp
    stage_banner "Stage 5: Fbank"
    [ -d "exp/${partition}_${sad_type}_sad_fbank" ] && rm -r exp/${partition}_${sad_type}_sad_fbank
    bash local/make_fbank.sh \
            --scp "${wav_scp}" \
            --segments data/${partition}/${sad_type}_sad \
            --store_dir exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --verbose ${verbose} \
            --nj 24 || exit 1
    stage_done 5 "Fbank done"
fi

# --- Stage 6: WPT + MHFA + w2v-BERT (USM_FTcode best_model.pt) ---
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    resolve_wav_scp
    stage_banner "Stage 6: WPT+MHFA+w2v-BERT embeddings (USM zl389 recipe)"
    if [ ! -d "${USM_FTCODE}" ]; then
        echo "$0: USM_FTCODE not found: ${USM_FTCODE}" ; exit 1
    fi
    if [ ! -f "${WPT_MHFA_CKPT_DIR}/args.json" ]; then
        echo "$0: missing args.json under WPT_MHFA_CKPT_DIR=${WPT_MHFA_CKPT_DIR}" ; exit 1
    fi
    if [ ! -f "${WPT_MHFA_CKPT_DIR}/${WPT_MHFA_CHECKPOINT_NAME}" ]; then
        echo "$0: missing checkpoint ${WPT_MHFA_CKPT_DIR}/${WPT_MHFA_CHECKPOINT_NAME}" ; exit 1
    fi

    emb_root="exp/${partition}_${sad_type}_sad_embedding${W2V}"
    [ -d "${emb_root}" ] && rm -r "${emb_root}"

    bash local/extract_emb_w2vbert_wpt_mhfa_zl389.sh \
            --scp "${wav_scp}" \
            --segments data/${partition}/${sad_type}_sad \
            --ckpt-dir "${WPT_MHFA_CKPT_DIR}" \
            --checkpoint-name "${WPT_MHFA_CHECKPOINT_NAME}" \
            --usm-ftcode "${USM_FTCODE}" \
            --device ${emb_device} \
            --store_dir "${emb_root}" \
            --batch_size ${emb_batch_size} \
            --frame_shift ${frame_shift} \
            --window_secs ${window_secs} \
            --period_secs ${period_secs} \
            --subseg_cmn ${subseg_cmn} \
            --verbose ${verbose} \
            --nj ${emb_nj} || exit 1

    emb_scp="${emb_root}/emb.scp"
    if [ ! -s "${emb_scp}" ]; then
        echo "$0: ${emb_scp} missing or empty." ; exit 1
    fi
    emb_lines=$(wc -l < "${emb_scp}")
    stage_done 6 "WPT+MHFA embeddings (${emb_lines} lines, ${emb_device})"
fi

# --- Stage 7: Clustering ---
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    emb_scp="exp/${partition}_${sad_type}_sad_embedding${W2V}/emb.scp"
    # Intermediate RTTM/label names include _w2vbert so DOVER-Lap outputs do not clash with ResNet
    labels_suffix="${partition}_${sad_type}_sad${W2V}_labels"
    rttm_suffix="${partition}_${sad_type}_sad${W2V}_rttm"

    stage_banner "Stage 7: ${cluster_type} clustering"

    if [ "${cluster_type}" == "doverlap" ]; then
        echo "DOVER-Lap: umap + ahc + spectral -> fused RTTM (${W2V})"
        mkdir -p exp/umap_cluster exp/ahc_cluster exp/spectral_cluster exp/doverlap_cluster

        echo "  [1/3] UMAP+HDBSCAN (merge_cutoff=${merge_cutoff})"
        "${PYTHON}" wespeaker/diar/umap_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/umap_cluster/${labels_suffix} \
                --merge_cutoff ${merge_cutoff}
        "${PYTHON}" wespeaker/diar/make_rttm.py \
                --labels exp/umap_cluster/${labels_suffix} \
                --channel 1 > exp/umap_cluster/${rttm_suffix}

        echo "  [2/3] AHC (threshold=${ahc_threshold}, ${ahc_linkage})"
        "${PYTHON}" wespeaker/diar/ahc_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/ahc_cluster/${labels_suffix} \
                --threshold ${ahc_threshold} --linkage ${ahc_linkage}
        "${PYTHON}" wespeaker/diar/make_rttm.py \
                --labels exp/ahc_cluster/${labels_suffix} \
                --channel 1 > exp/ahc_cluster/${rttm_suffix}

        echo "  [3/3] Spectral"
        "${PYTHON}" wespeaker/diar/spectral_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/spectral_cluster/${labels_suffix}
        "${PYTHON}" wespeaker/diar/make_rttm.py \
                --labels exp/spectral_cluster/${labels_suffix} \
                --channel 1 > exp/spectral_cluster/${rttm_suffix}

        doverlap_extra_args=""
        if [ -n "${doverlap_custom_weights}" ] && [ "${doverlap_weight_type}" = "custom" ]; then
            doverlap_extra_args="--custom-weight ${doverlap_custom_weights}"
        fi
        echo "  Fusing (${doverlap_weight_type}, gstd=${doverlap_gaussian_std}, dw=${doverlap_dover_weight}) ..."
        dover-lap \
            exp/doverlap_cluster/${rttm_suffix} \
            exp/umap_cluster/${rttm_suffix} \
            exp/ahc_cluster/${rttm_suffix} \
            exp/spectral_cluster/${rttm_suffix} \
            --weight-type ${doverlap_weight_type} \
            --gaussian-filter-std ${doverlap_gaussian_std} \
            --dover-weight ${doverlap_dover_weight} \
            ${doverlap_extra_args} \
            2>&1

        stage_done 7 "DOVER-Lap fusion"
    else
        mkdir -p exp/${cluster_type}_cluster
        [ -f "exp/${cluster_type}_cluster/${labels_suffix}" ] && rm "exp/${cluster_type}_cluster/${labels_suffix}"

        cluster_extra_args=""
        if [ "${cluster_type}" == "umap" ]; then
            cluster_extra_args="--merge_cutoff ${merge_cutoff}"
        elif [ "${cluster_type}" == "ahc" ]; then
            cluster_extra_args="--threshold ${ahc_threshold} --linkage ${ahc_linkage}"
        fi
        "${PYTHON}" wespeaker/diar/${cluster_type}_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/${cluster_type}_cluster/${labels_suffix} \
                ${cluster_extra_args}
        stage_done 7 "${cluster_type} clustering"
    fi
fi

# --- Stage 8: RTTM (non-doverlap only) ---
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && [ "$cluster_type" != "doverlap" ]; then
    stage_banner "Stage 8: labels -> RTTM"
    "${PYTHON}" wespeaker/diar/make_rttm.py \
            --labels exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_labels \
            --channel 1 > exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm
    stage_done 8 "RTTM generated"
fi

# --- Stage 8b: Overlap detection ---
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && [ "$use_overlap" = true ]; then
    resolve_wav_scp
    emb_scp="exp/${partition}_${sad_type}_sad_embedding${W2V}/emb.scp"
    rttm_in="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm"
    rttm_ovl="${rttm_in}_overlap"
    stage_banner "Stage 8: overlap detection"
    echo "Overlap (device=${overlap_device}, nj=${overlap_nj}) ..."
    "${PYTHON}" wespeaker/diar/overlap_detection.py \
            --rttm "${rttm_in}" \
            --scp-wav "${wav_scp}" \
            --scp-emb "${emb_scp}" \
            --output "${rttm_ovl}" \
            --min-overlap-dur ${overlap_min_dur} \
            --device ${overlap_device} \
            --nj ${overlap_nj} \
            --channel 1
    cp "${rttm_ovl}" "${rttm_in}"
    stage_done 8 "RTTM + overlap"
fi

# --- Stage 9: DER ---
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    stage_banner "Stage 9: DER (md-eval)"
    MD_EVAL=external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl
    if [ ! -f "${MD_EVAL}" ]; then
        echo "$0: SCTK not found: ${MD_EVAL}. Run stage 1." ; exit 1
    fi
    ref_dir="${NOTSOFAR_REF_ROOT}"
    sys_rttm="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm"

    if [ ! -d "${ref_dir}/${partition}" ]; then
        echo "$0: missing refs under ${ref_dir}/${partition}/ (run stage 2)" ; exit 1
    fi

    echo "Evaluating DER (WPT+MHFA+w2v-BERT, NOTSOFAR) ..."
    perl "${MD_EVAL}" \
         -c 0.25 \
         -r <(cat "${ref_dir}/${partition}"/*.rttm) \
         -s "${sys_rttm}" 2>&1 | grep -v '^WARNING:' | tee exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_res

    if [ ${get_each_file_res} -eq 1 ]; then
        single_file_res_dir=exp/${cluster_type}_cluster/${partition}_${sad_type}${W2V}_single_file_res
        mkdir -p $single_file_res_dir
        echo "Per-file DER -> ${single_file_res_dir}"
        awk '{print $2}' "${sys_rttm}" | sort -u | while read file_name; do
            perl "${MD_EVAL}" \
                 -c 0.25 \
                 -r <(cat "${ref_dir}/${partition}/${file_name}.rttm") \
                 -s <(grep "${file_name}" "${sys_rttm}") 2>/dev/null > ${single_file_res_dir}/${partition}_${file_name}_res
        done
    fi

    res_file="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_res"
    der_line=$(grep 'OVERALL SPEAKER DIARIZATION ERROR' "$res_file" 2>/dev/null || true)
    der_pct=$(echo "$der_line" | grep -oP '[\d.]+(?= percent)' || echo "?")
    stage_done 9 "DER = ${der_pct}% (${partition}, ${sad_type}, ${cluster_type})"
fi

print_summary

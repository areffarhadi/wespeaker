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
# VoxConverse v2 — WPT + w2v-BERT-2.0 + zl389 Adapter/ASP/Bottleneck (USM v2 GPU mel train)
#
# Same pipeline options as run_w2vbert.sh / run_updated.sh (VAD, clustering, DER).
# Stage 6: imports ``main_train_simple_sv_wpt_w2vbert_mhfa_zl389_v2_gpu_mel.py`` from USM_FTCODE
# (Encode-explore/USM_FTcode by default). Checkpoints: args.json + best_model.pt (sv_head.*).
#
# Outputs use *_w2vbert_wpt_zl389_adapt_gmel so they do not clash with run_w2vbert*.sh.
#
# Reported result (v2, partition=test, md-eval ALL): DER = 5.74% — see examples/voxconverse/README.md
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
export WESPEAKER_ROOT="${WESPEAKER_ROOT:-$_ws_root}"

stage=7
stop_stage=9
partition="test"   # test / dev
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

# ── WPT + zl389 adapter + w2v-BERT (USM: main_train_*_v2_gpu_mel.py) ──
_v2_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# Directory with main_train_simple_sv_wpt_w2vbert_mhfa_zl389_v2_gpu_mel.py, dataset_asv.py, losses.py
USM_FTCODE="${USM_FTCODE:-$HOME/Encode-explore/USM_FTcode}"
# Default: train_simple_sv_wpt_w2vbert_mhfa_zl389_v2_gpu_mel.sh --out_fold
WPT_ZL389_ADAPTER_CKPT_DIR="${WPT_ZL389_ADAPTER_CKPT_DIR:-$HOME/Encode-explore/USM_FTcode/ckpt_asv/simple_sv_wpt_w2vbert_zl389_v2_adim128_fixed24000_gpu_mel}"
WPT_ZL389_ADAPTER_CHECKPOINT_NAME="${WPT_ZL389_ADAPTER_CHECKPOINT_NAME:-best_model.pt}"
# Optional override for embedding Python (else PYTHON). USM enc-env often lacks kaldiio.
USM_WPT_ZL389_ADAPTER_PYTHON="${USM_WPT_ZL389_ADAPTER_PYTHON:-}"
# Stage 6: larger batch uses more VRAM but speeds up W2V-BERT forward (try 24/32 if memory allows).
emb_batch_size="${EMB_BATCH_SIZE:-32}"
# Split wav.scp across nj processes. Each job loads the full model — use nj>1 only with
# multiple GPUs (set CUDA_VISIBLE_DEVICES per worker) or expect OOM on a single GPU.
emb_nj="${EMB_NJ:-1}"
emb_device="${W2VBERT_EMB_DEVICE:-cuda}"
# Sliding window for sub-segments (must match run_updated.sh for fused_sim)
frame_shift=10
window_secs=1.5
period_secs=0.75

# ── Clustering (same as run_updated.sh) ───────────────────────────────
cluster_type="umap"   # spectral / umap / ahc / vbx / doverlap
merge_cutoff=0.2
ahc_threshold=0.21
ahc_linkage="average"
doverlap_weight_type="rank"
doverlap_custom_weights=""
doverlap_gaussian_std=0.5
doverlap_dover_weight=0.05

# ── VBx parameters (cluster_type=vbx) ────────────────────────────────
vbx_Fa=0.4
vbx_Fb=17.0
vbx_loopP=0.95
vbx_init_threshold="${ahc_threshold}"
vbx_n_iters=10
vbx_plda_model=""   # optional: path to trained PLDA .h5 model

# ── Overlap (same as run_updated.sh) ──────────────────────────────────
use_overlap=true
overlap_device="${OVERLAP_DEVICE:-cpu}"
overlap_min_dur=0.1
overlap_nj=16

# ── TS-VAD refinement (post-clustering overlap detection) ─────────────
use_tsvad=false
tsvad_overlap_threshold=0.55
tsvad_gap_threshold=0.35
tsvad_min_overlap_dur=0.2
tsvad_smooth_win=3

# ── Misc ────────────────────────────────────────────────────────────
verbose=true
bash_trace=false

# Suffix for all embedding artifact names (labels/rttm/res)
W2V="_w2vbert_wpt_zl389_adapt_gmel"

help_message="Usage: $0 [options]
VoxConverse v2 — WPT+zl389-adapter+w2v-BERT (USM v2 GPU mel train), same clustering/VAD as run_w2vbert.sh.

Stages: 1=SCTK 2=data 3=Demucs 4=VAD 5=fbank 6=WPT+zl389-adapter embeddings 7=cluster 8=RTTM+overlap+TS-VAD 9=DER

  --sad_type system|pyannote|oracle|funasr_fsmn
  --cluster_type spectral|umap|ahc|vbx|doverlap
  --use_demucs true|false
  --use_overlap true|false
  --use_tsvad true|false      profile-based TS-VAD overlap detection (no external model)
  --verbose true|false
  --bash_trace true|false
  --frame_shift 10            must match run_updated / make_fbank (ms)
  --window_secs 1.5           sub-segment window (match run_updated for fusion)
  --period_secs 0.75          sub-segment hop (seconds)
  --skip_download_if_present true|false  skip stages 1–2 when SCTK/data exist (default: true)

  VBx (cluster_type=vbx):
  --vbx_Fa 0.4               score scaling factor
  --vbx_Fb 17.0              speaker prior scaling factor
  --vbx_loopP 0.95           HMM self-loop probability
  --vbx_n_iters 10           number of VB iterations
  --vbx_plda_model ''        optional PLDA .h5 model path

  TS-VAD (use_tsvad=true):
  --tsvad_overlap_threshold 0.55  cosine threshold for secondary speaker
  --tsvad_gap_threshold 0.35      max primary-secondary score gap
  --tsvad_min_overlap_dur 0.2     minimum overlap segment duration (s)
  --tsvad_smooth_win 3            median filter window size

Env: USM_FTCODE (default: \$HOME/Encode-explore/USM_FTcode), WPT_ZL389_ADAPTER_CKPT_DIR (training --out_fold),
     WPT_ZL389_ADAPTER_CHECKPOINT_NAME (default best_model.pt), USM_WPT_ZL389_ADAPTER_PYTHON (optional; else PYTHON),
     WESPEAKER_ROOT (default: this repo), W2VBERT_EMB_DEVICE, EMB_BATCH_SIZE, EMB_NJ, HF_TOKEN (PyAnnote),
     DEMUCS_DEVICE, PYANNOTE_DEVICE, OVERLAP_DEVICE, FUNASR_HUB, FUNASR_REVISION, FUNASR_DEVICE, FUNASR_NJ (funasr_fsmn)

Note: Stage 6 loads SimpleSVModelWPTW2VBERTZl389Adapter from USM_FTCODE (v2 GPU mel script).
      Tune embedding speed: EMB_BATCH_SIZE (default 32), EMB_NJ (default 1; multi-GPU only).
      VBx: for best results train a PLDA model on VoxCeleb embeddings (wespeaker/bin/train_plda.py).
      TS-VAD: embedding-based overlap detection — no external model needed, complements any clusterer."

. tools/parse_options.sh

if [ "$bash_trace" = true ]; then
    set -x
fi

log_file="run_w2vbert_wpt_zl389_adapter_gmel_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Full log: $log_file"
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
    echo "  PIPELINE SUMMARY (WPT+zl389-adapter+w2v-BERT GPU mel, same options as run_updated.sh)"
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

# --- Stage 2: data ---
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    stage_banner "Stage 2: VoxConverse data + RTTM refs"
    if [ "${skip_download_if_present}" = true ] \
            && [ -d data/voxconverse-master ] \
            && compgen -G "data/dev/audio/*.wav" > /dev/null \
            && compgen -G "data/test/voxconverse_test_wav/*.wav" > /dev/null; then
        echo "$0: Stage 2: VoxConverse data already on disk (skip_download_if_present=true)."
        if [ ! -s data/dev/wav.scp ]; then
            ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp
        fi
        if [ ! -s data/test/wav.scp ]; then
            ls `pwd`/data/test/voxconverse_test_wav/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/test/wav.scp
        fi
        stage_done 2 "VoxConverse data (existing)"
    else
        mkdir -p data
        wget -c https://github.com/joonson/voxconverse/archive/refs/heads/master.zip -O data/voxconverse_master.zip
        extract_zip data/voxconverse_master.zip data

        mkdir -p data/dev
        wget --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip -O data/voxconverse_dev_wav.zip
        extract_zip data/voxconverse_dev_wav.zip data/dev
        if ! compgen -G "data/dev/audio/*.wav" > /dev/null; then
            echo "$0: no dev WAVs after extract." ; exit 1
        fi
        ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp

        mkdir -p data/test
        wget --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip -O data/voxconverse_test_wav.zip
        extract_zip data/voxconverse_test_wav.zip data/test
        if ! compgen -G "data/test/voxconverse_test_wav/*.wav" > /dev/null; then
            echo "$0: no test WAVs after extract." ; exit 1
        fi
        ls `pwd`/data/test/voxconverse_test_wav/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/test/wav.scp
        stage_done 2 "VoxConverse data ready"
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
                    --rttm data/voxconverse-master/${partition}/${utt}.rttm \
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

# --- Stage 6: WPT + zl389 adapter + w2v-BERT (USM v2 GPU mel best_model.pt) ---
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    resolve_wav_scp
    stage_banner "Stage 6: WPT+zl389-adapter+w2v-BERT embeddings (USM v2 GPU mel)"
    if [ ! -d "${USM_FTCODE}" ]; then
        echo "$0: USM_FTCODE not found: ${USM_FTCODE}" ; exit 1
    fi
    if [ ! -f "${USM_FTCODE}/main_train_simple_sv_wpt_w2vbert_mhfa_zl389_v2_gpu_mel.py" ]; then
        echo "$0: expected training script under USM_FTCODE: main_train_simple_sv_wpt_w2vbert_mhfa_zl389_v2_gpu_mel.py" ; exit 1
    fi
    if [ ! -f "${WPT_ZL389_ADAPTER_CKPT_DIR}/args.json" ]; then
        echo "$0: missing args.json under WPT_ZL389_ADAPTER_CKPT_DIR=${WPT_ZL389_ADAPTER_CKPT_DIR}" ; exit 1
    fi
    if [ ! -f "${WPT_ZL389_ADAPTER_CKPT_DIR}/${WPT_ZL389_ADAPTER_CHECKPOINT_NAME}" ]; then
        echo "$0: missing checkpoint ${WPT_ZL389_ADAPTER_CKPT_DIR}/${WPT_ZL389_ADAPTER_CHECKPOINT_NAME}" ; exit 1
    fi

    emb_root="exp/${partition}_${sad_type}_sad_embedding${W2V}"
    [ -d "${emb_root}" ] && rm -r "${emb_root}"

    export USM_WPT_ZL389_ADAPTER_PYTHON="${USM_WPT_ZL389_ADAPTER_PYTHON:-}"
    bash local/extract_emb_w2vbert_wpt_zl389_adapter.sh \
            --scp "${wav_scp}" \
            --segments data/${partition}/${sad_type}_sad \
            --ckpt-dir "${WPT_ZL389_ADAPTER_CKPT_DIR}" \
            --checkpoint-name "${WPT_ZL389_ADAPTER_CHECKPOINT_NAME}" \
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
    stage_done 6 "WPT+zl389-adapter embeddings (${emb_lines} lines, ${emb_device})"
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
        elif [ "${cluster_type}" == "vbx" ]; then
            cluster_extra_args="--Fa ${vbx_Fa} --Fb ${vbx_Fb} --loopP ${vbx_loopP} --init-threshold ${vbx_init_threshold} --n-iters ${vbx_n_iters}"
            if [ -n "${vbx_plda_model}" ]; then
                cluster_extra_args="${cluster_extra_args} --plda-model ${vbx_plda_model}"
            fi
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

# --- Stage 8c: TS-VAD refinement ---
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && [ "$use_tsvad" = true ]; then
    emb_scp="exp/${partition}_${sad_type}_sad_embedding${W2V}/emb.scp"
    rttm_in="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm"
    rttm_tsvad="${rttm_in}_tsvad"
    stage_banner "Stage 8c: TS-VAD refinement"
    echo "TS-VAD (overlap_threshold=${tsvad_overlap_threshold}, gap=${tsvad_gap_threshold}, smooth=${tsvad_smooth_win}) ..."
    "${PYTHON}" wespeaker/diar/ts_vad.py \
            --rttm "${rttm_in}" \
            --scp-emb "${emb_scp}" \
            --output "${rttm_tsvad}" \
            --overlap-threshold ${tsvad_overlap_threshold} \
            --gap-threshold ${tsvad_gap_threshold} \
            --min-overlap-dur ${tsvad_min_overlap_dur} \
            --smooth-win ${tsvad_smooth_win} \
            --channel 1
    cp "${rttm_tsvad}" "${rttm_in}"
    stage_done 8 "RTTM + TS-VAD"
fi

# --- Stage 9: DER ---
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    stage_banner "Stage 9: DER (md-eval)"
    MD_EVAL=external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl
    if [ ! -f "${MD_EVAL}" ]; then
        echo "$0: SCTK not found: ${MD_EVAL}. Run stage 1." ; exit 1
    fi
    ref_dir=data/voxconverse-master
    sys_rttm="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm"

    if [ ! -d "${ref_dir}/${partition}" ]; then
        echo "$0: missing refs under ${ref_dir}/${partition}/" ; exit 1
    fi

    echo "Evaluating DER (WPT+zl389-adapter+w2v-BERT) ..."
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

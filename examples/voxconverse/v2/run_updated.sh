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

. ./path.sh || exit 1

# Prefer repo .venv so FunASR / torch match (override with PYTHON=/path/to/python).
_ws_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"
if [ -z "${PYTHON}" ] && [ -x "${_ws_root}/.venv/bin/python" ]; then
    PYTHON="${_ws_root}/.venv/bin/python"
    # make_fbank.sh / extract_emb.sh call python3 — put venv first on PATH
    export PATH="${_ws_root}/.venv/bin:${PATH}"
else
    PYTHON="${PYTHON:-python3}"
fi

stage=4
stop_stage=9
partition="dev"         # dev/test
subseg_cmn=true
get_each_file_res=1

# ── VAD ──────────────────────────────────────────────────────────────
sad_type="funasr_fsmn"       # oracle / system (Silero) / pyannote / funasr_fsmn
# PyAnnote VAD settings (only when sad_type=pyannote)
pyannote_device="${PYANNOTE_DEVICE:-cpu}"
pyannote_onset=0.5
pyannote_offset=0.5
pyannote_nj=16
# FunASR FSMN-VAD (only when sad_type=funasr_fsmn); needs: pip install funasr
funasr_hub="${FUNASR_HUB:-hf}"              # hf | ms
funasr_revision="${FUNASR_REVISION:-v2.0.4}"
funasr_device="${FUNASR_DEVICE:-cpu}"     # use cpu if nj>1; cuda ok with --funasr_nj 1
funasr_nj="${FUNASR_NJ:-4}"

# ── Demucs (optional vocal separation before VAD) ───────────────────
use_demucs=false
demucs_device="${DEMUCS_DEVICE:-cuda}"
demucs_model="${DEMUCS_MODEL:-htdemucs}"

# ── Clustering ───────────────────────────────────────────────────────
cluster_type="doverlap" # spectral / umap / ahc / doverlap

# UMAP+HDBSCAN params
merge_cutoff=0.2

# AHC params
ahc_threshold=0.21      # cosine similarity stopping threshold (tuned on dev)
ahc_linkage="average"   # average / complete / single

# DOVER-Lap params (only when cluster_type=doverlap)
doverlap_weight_type="rank"      # rank / custom / norm
doverlap_custom_weights=""       # e.g. "0.5 0.3 0.2" — only with weight_type=custom
doverlap_gaussian_std=0.5        # 0.01=no filtering, 0.5=default
doverlap_dover_weight=0.05       # tuned on dev

# ── Overlap detection (optional, after RTTM) ────────────────────────
use_overlap=false
overlap_device="${OVERLAP_DEVICE:-cpu}"
overlap_min_dur=0.1
overlap_nj=16

# ─────────────────────────────────────────────────────────────────────

help_message="Usage: $0 [options]
Stages: 1=SCTK+ResNet 2=data 3=Demucs 4=VAD 5=fbank 6=embed 7=cluster 8=RTTM 9=DER
  --sad_type system|pyannote|oracle|funasr_fsmn   VAD backend
  --cluster_type spectral|umap|ahc|doverlap  (default: doverlap)
  --use_demucs true|false     Demucs vocals before VAD (default: false)
  --use_overlap true|false    PyAnnote overlap detection (default: false)"

# Extract .zip with Python stdlib (no system `unzip` required — see local/extract_zip.py)
extract_zip() {
    "${PYTHON}" local/extract_zip.py "$1" "$2" || exit 1
}

# wav.scp for VAD / fbank: original mix or Demucs vocals
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

. tools/parse_options.sh

log_file="run_resnet_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Full session log: $log_file"
echo ""

SECONDS=0
declare -A stage_status

stage_done() {
    local snum="$1" msg="$2"
    stage_status[$snum]="$msg"
}

print_summary() {
    echo ""
    echo "================================================================================"
    echo "  PIPELINE SUMMARY"
    echo "================================================================================"
    for snum in $(printf '%s\n' "${!stage_status[@]}" | sort -n); do
        echo "  Stage ${snum}: ${stage_status[$snum]}"
    done
    echo "--------------------------------------------------------------------------------"
    echo "  Elapsed: ${SECONDS}s | Log: $log_file"
    echo "================================================================================"
}


# Stage 1: Prerequisites (SCTK + ResNet ONNX)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p external_tools
    wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip -O external_tools/SCTK-v2.4.12.zip
    extract_zip external_tools/SCTK-v2.4.12.zip external_tools

    mkdir -p pretrained_models
    wget -c https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx -O pretrained_models/voxceleb_resnet34_LM.onnx
    stage_done 1 "SCTK + ResNet34 ONNX ready"
fi


# Stage 2: Download VoxConverse data
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p data
    wget -c https://github.com/joonson/voxconverse/archive/refs/heads/master.zip -O data/voxconverse_master.zip
    extract_zip data/voxconverse_master.zip data

    mkdir -p data/dev
    wget --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip -O data/voxconverse_dev_wav.zip
    extract_zip data/voxconverse_dev_wav.zip data/dev
    ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp

    mkdir -p data/test
    wget  --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip -O data/voxconverse_test_wav.zip
    extract_zip data/voxconverse_test_wav.zip data/test
    ls `pwd`/data/test/voxconverse_test_wav/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/test/wav.scp
    stage_done 2 "VoxConverse data ready"
fi


# Stage 3: Demucs (optional)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$use_demucs" = true ]; then
        demucs_vocals_dir="data/${partition}/demucs_vocals"
        rm -rf "${demucs_vocals_dir}" 2>/dev/null
        mkdir -p "${demucs_vocals_dir}"
        "${PYTHON}" wespeaker/diar/demucs_vocals.py \
                --scp "data/${partition}/wav.scp" \
                --out-dir "${demucs_vocals_dir}" \
                --wav-scp-out "data/${partition}/wav_demucs.scp" \
                --model "${demucs_model}" \
                --device "${demucs_device}" || exit 1
        stage_done 3 "Demucs vocals -> data/${partition}/wav_demucs.scp"
    else
        echo "Stage 3: Demucs skipped (use_demucs=false)."
    fi
fi


# Stage 4: VAD
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    resolve_wav_scp
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
       "${PYTHON}" wespeaker/diar/make_system_sad.py \
               --scp "${wav_scp}" \
               --min-duration $min_duration > data/${partition}/system_sad
       sad_lines=$(wc -l < "data/${partition}/system_sad")
       echo "System SAD: ${sad_lines} segments"
    fi

    if [[ "x${sad_type}" == "xpyannote" ]]; then
       echo "Running PyAnnote VAD (device=${pyannote_device}, nj=${pyannote_nj}) ..."
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
           echo "$0: pyannote_sad is empty. Check model / HF_TOKEN / wav.scp."
           exit 1
       fi
    fi

    if [[ "x${sad_type}" == "xfunasr_fsmn" ]]; then
       echo "Running FunASR FSMN-VAD (hub=${funasr_hub}, device=${funasr_device}, nj=${funasr_nj}) ..."
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
           echo "$0: funasr_fsmn_sad is empty. Check funasr install (WeSpeaker .venv) and wav.scp."
           exit 1
       fi
    fi

    stage_done 4 "VAD (${sad_type}, ${sad_lines:-?} segments)"
fi


# Stage 5: Fbank features
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    resolve_wav_scp
    [ -d "exp/${partition}_${sad_type}_sad_fbank" ] && rm -r exp/${partition}_${sad_type}_sad_fbank

    echo "Make Fbank features ..."
    bash local/make_fbank.sh \
            --scp "${wav_scp}" \
            --segments data/${partition}/${sad_type}_sad \
            --store_dir exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --verbose true \
            --nj 24 || exit 1
    stage_done 5 "Fbank done"
fi


# Stage 6: Extract embeddings
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    [ -d "exp/${partition}_${sad_type}_sad_embedding" ] && rm -r exp/${partition}_${sad_type}_sad_embedding

    echo "Extract embeddings ..."
    bash local/extract_emb.sh \
            --scp exp/${partition}_${sad_type}_sad_fbank/fbank.scp \
            --pretrained_model pretrained_models/voxceleb_resnet34_LM.onnx \
            --device cuda \
            --store_dir exp/${partition}_${sad_type}_sad_embedding \
            --batch_size 96 \
            --frame_shift 10 \
            --window_secs 1 \
            --period_secs 0.5 \
            --subseg_cmn ${subseg_cmn} \
            --verbose true \
            --nj 1 || exit 1
    emb_lines=$(wc -l < "exp/${partition}_${sad_type}_sad_embedding/emb.scp")
    stage_done 6 "Embeddings (${emb_lines} sub-segments)"
fi


# Stage 7: Clustering
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    emb_scp="exp/${partition}_${sad_type}_sad_embedding/emb.scp"
    labels_suffix="${partition}_${sad_type}_sad_labels"
    rttm_suffix="${partition}_${sad_type}_sad_rttm"

    if [ "${cluster_type}" == "doverlap" ]; then
        echo "DOVER-Lap: running umap + ahc + spectral ..."

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

        stage_done 7 "DOVER-Lap fusion (umap+ahc+spectral)"
    else
        mkdir -p exp/${cluster_type}_cluster
        [ -f "exp/${cluster_type}_cluster/${labels_suffix}" ] && rm exp/${cluster_type}_cluster/${labels_suffix}

        cluster_extra_args=""
        if [ "${cluster_type}" == "umap" ]; then
            cluster_extra_args="--merge_cutoff ${merge_cutoff}"
        elif [ "${cluster_type}" == "ahc" ]; then
            cluster_extra_args="--threshold ${ahc_threshold} --linkage ${ahc_linkage}"
        fi

        echo "${cluster_type} clustering ..."
        "${PYTHON}" wespeaker/diar/${cluster_type}_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/${cluster_type}_cluster/${labels_suffix} \
                ${cluster_extra_args}
        stage_done 7 "${cluster_type} clustering done"
    fi
fi


# Stage 8: Convert labels to RTTM (skipped for doverlap — already done above)
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && [ "$cluster_type" != "doverlap" ]; then
    "${PYTHON}" wespeaker/diar/make_rttm.py \
            --labels exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_labels \
            --channel 1 > exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm
    stage_done 8 "RTTM generated"
fi


# Stage 8.5: Overlap detection (optional)
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && [ "$use_overlap" = true ]; then
    resolve_wav_scp
    rttm_in="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm"
    rttm_ovl="${rttm_in}_overlap"
    echo "Running overlap detection (device=${overlap_device}, nj=${overlap_nj}) ..."
    "${PYTHON}" wespeaker/diar/overlap_detection.py \
            --rttm "${rttm_in}" \
            --scp-wav "${wav_scp}" \
            --scp-emb exp/${partition}_${sad_type}_sad_embedding/emb.scp \
            --output "${rttm_ovl}" \
            --min-overlap-dur ${overlap_min_dur} \
            --device ${overlap_device} \
            --nj ${overlap_nj} \
            --channel 1
    cp "${rttm_ovl}" "${rttm_in}"
    stage_done 8 "RTTM + overlap detection"
fi


# Stage 9: Evaluate
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    ref_dir=data/voxconverse-master/
    sys_rttm="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm"

    echo -e "Evaluating DER ...\n"
    perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
         -c 0.25 \
         -r <(cat ${ref_dir}/${partition}/*.rttm) \
         -s "${sys_rttm}" 2>&1 | grep -v '^WARNING:' | tee exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_res

    if [ ${get_each_file_res} -eq 1 ]; then
        single_file_res_dir=exp/${cluster_type}_cluster/${partition}_${sad_type}_single_file_res
        mkdir -p $single_file_res_dir
        echo -e "\nPer-file DER -> ${single_file_res_dir}\n"

        awk '{print $2}' "${sys_rttm}" | sort -u | while read file_name; do
            perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
                 -c 0.25 \
                 -r <(cat ${ref_dir}/${partition}/${file_name}.rttm) \
                 -s <(grep "${file_name}" "${sys_rttm}") 2>/dev/null > ${single_file_res_dir}/${partition}_${file_name}_res
        done
    fi

    res_file="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_res"
    der_line=$(grep 'OVERALL SPEAKER DIARIZATION ERROR' "$res_file" 2>/dev/null || true)
    der_pct=$(echo "$der_line" | grep -oP '[\d.]+(?= percent)' || echo "?")
    stage_done 9 "DER = ${der_pct}% (${partition}, ${sad_type}, ${cluster_type})"
fi

print_summary

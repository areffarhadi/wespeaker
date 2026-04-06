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
# VoxConverse v2 diarization using w2v-BERT-2.0 SV embeddings (PyTorch checkpoint)
# instead of the default WeSpeaker ResNet34 ONNX model (run.sh).
#
# Usage: ./run_w2vbert.sh
#   (defaults below; optional overrides: --stage / --stop_stage / --partition / …
#    or env W2VBERT_REPO, W2VBERT_CHECKPOINT, W2VBERT_EMB_DEVICE)
#
# Embedding outputs use *_w2vbert suffixes so run.sh (ResNet) results stay separate.
# -----------------------------------------------------------------------------


# ./run_w2vbert.sh --bash_trace true

. ./path.sh || exit 1

# Full pipeline: SCTK (1) … Demucs vocals (3, optional) … VAD (4) … DER (9).
# Use --stage / --stop_stage to limit. With use_demucs=false, stage 3 is skipped.
stage=4
stop_stage=9

sad_type="system"       # oracle/system
partition="test"         # dev/test
cluster_type="umap" # spectral/umap

subseg_cmn=true
get_each_file_res=1

# Demucs vocal separation before VAD (optional). Requires: pip install demucs
use_demucs=false
demucs_device="${DEMUCS_DEVICE:-cuda}"
demucs_model="${DEMUCS_MODEL:-htdemucs}"

# --- w2v-BERT paths (edit here if your tree differs) ---
HF_MODELS="${HF_MODELS:-$HOME/Encode-explore/USM_FTcode/hf_models}"
w2vbert_repo="${W2VBERT_REPO:-$HF_MODELS/w2v-BERT-2.0_SV}"
checkpoint="${W2VBERT_CHECKPOINT:-$HF_MODELS/zl389_w2v-bert-2.0_SV/model_base_0.23.pth}"

emb_batch_size=8
emb_device="${W2VBERT_EMB_DEVICE:-cuda}"

# Show Python progress on the terminal (tee) instead of hiding it in log/*.log only.
verbose=true
# Print every shell command (very noisy): --bash_trace true
bash_trace=false

help_message="Usage: $0 [options]
Stages: 1=SCTK 2=data 3=Demucs (optional) 4=VAD 5=fbank 6=w2v-BERT 7=cluster 8=RTTM 9=DER
  --use_demucs true|false   Run Demucs vocals before VAD (default: false). Needs: pip install demucs
  --demucs_device cuda|cpu  Device for Demucs (default: cuda, env DEMUCS_DEVICE)
  --demucs_model NAME       e.g. htdemucs (default: htdemucs, env DEMUCS_MODEL)
  --verbose true|false   Show live Python/log output (default: true). Use false for quiet runs.
  --bash_trace true|false  Print each shell command (set -x).
Override examples: --stage 6 --stop_stage 6  (embeddings only)
Env: W2VBERT_REPO, W2VBERT_CHECKPOINT, HF_MODELS, W2VBERT_EMB_DEVICE, DEMUCS_DEVICE"

. tools/parse_options.sh

if [ "$bash_trace" = true ]; then
    set -x
fi

# ---- Logging: mirror ALL terminal output to a timestamped log file ----
log_file="run_w2vbert_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Full log → $log_file"
echo ""

SECONDS=0
declare -A stage_status
declare -A stage_time

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
    stage_time[$snum]="${SECONDS}s"
}

print_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        PIPELINE SUMMARY                                    ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    for snum in $(echo "${!stage_status[@]}" | tr ' ' '\n' | sort -n); do
        printf "║  Stage %s: %-66s ║\n" "$snum" "${stage_status[$snum]}"
    done
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    printf "║  Total elapsed: %-59s ║\n" "${SECONDS}s"
    printf "║  Full log: %-64s ║\n" "$log_file"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
}

# Extract .zip with Python stdlib (no system `unzip` required — see local/extract_zip.py)
extract_zip() {
    python3 local/extract_zip.py "$1" "$2" || exit 1
}

# wav.scp for VAD / fbank / embeddings: original or Demucs vocals
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

# Prerequisite: SCTK for evaluation (same as run.sh). ResNet ONNX is NOT downloaded.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    stage_banner "Stage 1: SCTK (download + extract)"
    mkdir -p external_tools

    wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip -O external_tools/SCTK-v2.4.12.zip
    extract_zip external_tools/SCTK-v2.4.12.zip external_tools
    stage_done 1 "SCTK downloaded + extracted"
fi


# Download VoxConverse dev/test audios and the corresponding annotations
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    stage_banner "Stage 2: VoxConverse data + RTTM refs (download + extract)"
    mkdir -p data

    wget -c https://github.com/joonson/voxconverse/archive/refs/heads/master.zip -O data/voxconverse_master.zip
    extract_zip data/voxconverse_master.zip data

    mkdir -p data/dev

    wget --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip -O data/voxconverse_dev_wav.zip
    extract_zip data/voxconverse_dev_wav.zip data/dev

    if ! compgen -G "data/dev/audio/*.wav" > /dev/null; then
        echo "$0: no files matched data/dev/audio/*.wav after extracting dev zip."
        echo "    Fix stage 2 (download/extract), then re-run from --stage 2."
        exit 1
    fi
    ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp

    mkdir -p data/test

    wget  --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip -O data/voxconverse_test_wav.zip
    extract_zip data/voxconverse_test_wav.zip data/test

    if ! compgen -G "data/test/voxconverse_test_wav/*.wav" > /dev/null; then
        echo "$0: no files matched data/test/voxconverse_test_wav/*.wav after extracting test zip."
        echo "    Fix stage 2, then re-run from --stage 2."
        exit 1
    fi
    ls `pwd`/data/test/voxconverse_test_wav/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/test/wav.scp
    stage_done 2 "VoxConverse data + RTTM refs ready"
fi


# Demucs: vocals-only WAVs before VAD (optional; code lives in wespeaker/diar/demucs_vocals.py)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$use_demucs" = true ]; then
        stage_banner "Stage 3: Demucs (vocals, ${demucs_model})"
        demucs_vocals_dir="data/${partition}/demucs_vocals"
        rm -rf "${demucs_vocals_dir}" 2>/dev/null
        mkdir -p "${demucs_vocals_dir}"
        python3 wespeaker/diar/demucs_vocals.py \
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


# Voice activity detection
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    resolve_wav_scp
    stage_banner "Stage 4: VAD / SAD (${sad_type})"
    min_duration=0.255

    if [[ "x${sad_type}" == "xoracle" ]]; then
        while read -r utt wav_path; do
            python3 wespeaker/diar/make_oracle_sad.py \
                    --rttm data/voxconverse-master/${partition}/${utt}.rttm \
                    --min-duration $min_duration
        done < "${wav_scp}" > data/${partition}/oracle_sad
    fi

    if [[ "x${sad_type}" == "xsystem" ]]; then
       if [ "$verbose" = true ]; then
           # tee only stdout (VAD segments); stderr (Python warnings) goes straight to terminal
           python3 wespeaker/diar/make_system_sad.py \
                   --scp "${wav_scp}" \
                   --min-duration $min_duration | tee "data/${partition}/system_sad"
       else
           python3 wespeaker/diar/make_system_sad.py \
                   --scp "${wav_scp}" \
                   --min-duration $min_duration > "data/${partition}/system_sad" 2>/dev/null
       fi
       sad_lines=$(wc -l < "data/${partition}/system_sad")
       echo "System SAD: ${sad_lines} segments in data/${partition}/system_sad"
       if [ "$sad_lines" -eq 0 ]; then
           echo "$0: system_sad is empty — no speech detected. Check wav.scp / VAD settings."
           exit 1
       fi
    fi
    stage_done 4 "VAD done (${sad_type}, ${sad_lines:-?} segments)"
fi


# Extract fbank features (unchanged vs run.sh; still used if you compare pipelines)
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    resolve_wav_scp
    stage_banner "Stage 5: Fbank (per VAD segment)"

    [ -d "exp/${partition}_${sad_type}_sad_fbank" ] && rm -r exp/${partition}_${sad_type}_sad_fbank

    echo "Make Fbank features and store it under exp/${partition}_${sad_type}_sad_fbank"
    echo "..."
    bash local/make_fbank.sh \
            --scp "${wav_scp}" \
            --segments data/${partition}/${sad_type}_sad \
            --store_dir exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --verbose ${verbose} \
            --nj 24 || exit 1
    stage_done 5 "Fbank features extracted"
fi


# Extract embeddings (w2v-BERT-2.0 SV checkpoint — not ONNX ResNet)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    resolve_wav_scp
    stage_banner "Stage 6: w2v-BERT SV embeddings"
    if [ ! -d "${w2vbert_repo}" ]; then
        echo "$0: w2v-BERT repo not found: ${w2vbert_repo}"
        echo "    Edit HF_MODELS / w2vbert_repo defaults at top of $0, or: export W2VBERT_REPO=..."
        exit 1
    fi
    if [ ! -f "${checkpoint}" ]; then
        echo "$0: checkpoint not found: ${checkpoint}"
        echo "    Edit checkpoint default at top of $0, or: export W2VBERT_CHECKPOINT=..."
        exit 1
    fi

    [ -d "exp/${partition}_${sad_type}_sad_embedding_w2vbert" ] && rm -r exp/${partition}_${sad_type}_sad_embedding_w2vbert

    echo "Extract w2v-BERT SV embeddings -> exp/${partition}_${sad_type}_sad_embedding_w2vbert"
    echo "..."
    bash local/extract_emb_w2vbert.sh \
            --scp "${wav_scp}" \
            --segments data/${partition}/${sad_type}_sad \
            --w2vbert-repo ${w2vbert_repo} \
            --checkpoint ${checkpoint} \
            --device ${emb_device} \
            --store_dir exp/${partition}_${sad_type}_sad_embedding_w2vbert \
            --batch_size ${emb_batch_size} \
            --frame_shift 10 \
            --window_secs 1.5 \
            --period_secs 0.75 \
            --subseg_cmn ${subseg_cmn} \
            --verbose ${verbose} \
            --nj 1 || exit 1

    emb_scp="exp/${partition}_${sad_type}_sad_embedding_w2vbert/emb.scp"
    if [ ! -s "${emb_scp}" ]; then
        echo "$0: ${emb_scp} is missing or empty."
        echo "    Fix embedding stage (see exp/${partition}_${sad_type}_sad_embedding_w2vbert/log/) then re-run from --stage 6."
        exit 1
    fi
    emb_count=$(wc -l < "${emb_scp}")
    stage_done 6 "w2v-BERT embeddings (${emb_count} sub-segments, device=${emb_device})"
fi


# Clustering
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    stage_banner "Stage 7: ${cluster_type} clustering"

    [ -f "exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_labels" ] && rm exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_labels

    echo "Doing ${cluster_type} clustering -> exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_labels"
    echo "..."
    python3 wespeaker/diar/${cluster_type}_clusterer.py \
            --scp exp/${partition}_${sad_type}_sad_embedding_w2vbert/emb.scp \
            --output exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_labels
    stage_done 7 "${cluster_type} clustering done"
fi


# Convert labels to RTTMs
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    stage_banner "Stage 8: labels -> RTTM"
    python3 wespeaker/diar/make_rttm.py \
            --labels exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_labels \
            --channel 1 > exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_rttm
    stage_done 8 "RTTM generated"
fi


# Evaluate
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    stage_banner "Stage 9: DER evaluation (md-eval)"
    MD_EVAL=external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl
    if [ ! -f "${MD_EVAL}" ]; then
        echo "$0: SCTK md-eval not found: ${MD_EVAL}"
        echo "    Run stage 1 (downloads SCTK zip + Python extract) or extract SCTK manually."
        exit 1
    fi
    ref_dir=data/voxconverse-master
    if [ ! -d "${ref_dir}/${partition}" ] \
        || [ "$(find "${ref_dir}/${partition}" -maxdepth 1 -name '*.rttm' 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "$0: reference RTTMs not found under ${ref_dir}/${partition}/"
        echo "    Run stage 2 (annotations zip) successfully, then re-run from --stage 9."
        exit 1
    fi
    echo -e "Get the DER results (w2v-BERT embeddings)\n..."
    perl "${MD_EVAL}" \
         -c 0.25 \
         -r <(cat "${ref_dir}/${partition}"/*.rttm) \
         -s exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_rttm 2>&1 | tee exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_res

    if [ ${get_each_file_res} -eq 1 ];then
        single_file_res_dir=exp/${cluster_type}_cluster/${partition}_${sad_type}_w2vbert_single_file_res
        mkdir -p $single_file_res_dir
        echo -e "\nPer-file DER -> ${single_file_res_dir}\n..."

        awk '{print $2}' exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_rttm | sort -u  | while read file_name; do
            perl "${MD_EVAL}" \
                 -c 0.25 \
                 -r <(cat "${ref_dir}/${partition}/${file_name}.rttm") \
                 -s <(grep "${file_name}" exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_rttm) > ${single_file_res_dir}/${partition}_${file_name}_res
        done
        echo "Per-file results written to ${single_file_res_dir}/"
    fi

    der_line=$(grep 'OVERALL SPEAKER DIARIZATION ERROR' exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_w2vbert_res 2>/dev/null || true)
    der_pct=$(echo "$der_line" | grep -oP '[\d.]+(?= percent)' || echo "?")
    stage_done 9 "DER = ${der_pct}%  (${partition}, ${sad_type} SAD, ${cluster_type} cluster)"
fi

# ---- Final summary (always printed) ----
print_summary

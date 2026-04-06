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

# Full pipeline: SCTK+ResNet (1) … data (2) … Demucs (3, optional) … VAD (4) … DER (9).
# With use_demucs=false, stage 3 is skipped (no wav_demucs.scp required).
stage=1
stop_stage=9
sad_type="system"       # oracle/system
partition="dev"         # dev/test
cluster_type="spectral" # spectral/umap

# do cmn on the sub-segment or on the vad segment
subseg_cmn=true
# whether print the evaluation result for each file
get_each_file_res=1

# Demucs vocal separation before VAD (optional). Requires: pip install demucs
use_demucs=false
demucs_device="${DEMUCS_DEVICE:-cuda}"
demucs_model="${DEMUCS_MODEL:-htdemucs}"

help_message="Usage: $0 [options]
Stages: 1=SCTK+ResNet 2=data 3=Demucs (optional) 4=VAD 5=fbank 6=embed 7=cluster 8=RTTM 9=DER
  --use_demucs true|false   Run Demucs vocals before VAD (default: false). Needs: pip install demucs
  --demucs_device cuda|cpu  Device for Demucs (default: cuda, env DEMUCS_DEVICE)
  --demucs_model NAME       e.g. htdemucs (default: htdemucs, env DEMUCS_MODEL)"

# Extract .zip with Python stdlib (no system `unzip` required — see local/extract_zip.py)
extract_zip() {
    python3 local/extract_zip.py "$1" "$2" || exit 1
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

# Mirror ALL stdout/stderr to a log file so nothing is lost when the terminal scrolls
# or the IDE truncates scrollback (this script does not call `clear`).
log_file="run_resnet_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Full session log (all stages): $log_file"
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
    echo "  PIPELINE SUMMARY (ResNet34 baseline)"
    echo "================================================================================"
    for snum in $(printf '%s\n' "${!stage_status[@]}" | sort -n); do
        echo "  Stage ${snum}: ${stage_status[$snum]}"
    done
    echo "--------------------------------------------------------------------------------"
    echo "  Elapsed: ${SECONDS}s"
    echo "  Log file: $log_file"
    echo "================================================================================"
}

# Prerequisite
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p external_tools

    # [1] Download evaluation toolkit
    wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip -O external_tools/SCTK-v2.4.12.zip
    extract_zip external_tools/SCTK-v2.4.12.zip external_tools

    # [2] Download ResNet34 speaker model pretrained by WeSpeaker Team
    mkdir -p pretrained_models

    wget -c https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx -O pretrained_models/voxceleb_resnet34_LM.onnx
    stage_done 1 "SCTK + ResNet34 ONNX ready"
fi


# Download VoxConverse dev/test audios and the corresponding annotations
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p data

    # Download annotations for dev and test sets (version 0.0.3)
    wget -c https://github.com/joonson/voxconverse/archive/refs/heads/master.zip -O data/voxconverse_master.zip
    extract_zip data/voxconverse_master.zip data

    # Download annotations from VoxSRC-23 validation toolkit (looks like version 0.0.2)
    # cd data && git clone https://github.com/JaesungHuh/VoxSRC2023.git --recursive && cd -

    # Download dev audios
    mkdir -p data/dev

    #wget --no-check-certificate -c https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip -O data/voxconverse_dev_wav.zip
    # The above url may not be reachable, you can try the link below.
    # This url is from https://github.com/joonson/voxconverse/blob/master/README.md
    wget --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip -O data/voxconverse_dev_wav.zip
    extract_zip data/voxconverse_dev_wav.zip data/dev

    # Create wav.scp for dev audios
    ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp

    # Test audios
    mkdir -p data/test

    #wget --no-check-certificate -c https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_test_wav.zip -O data/voxconverse_test_wav.zip
    # The above url may not be reachable, you can try the link below.
    # This url is from https://github.com/joonson/voxconverse/blob/master/README.md
    wget  --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip -O data/voxconverse_test_wav.zip
    extract_zip data/voxconverse_test_wav.zip data/test

    # Create wav.scp for test audios
    ls `pwd`/data/test/voxconverse_test_wav/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/test/wav.scp
    stage_done 2 "VoxConverse audio + RTTM refs under data/"
fi


# Demucs: vocals-only WAVs before VAD (optional; wespeaker/diar/demucs_vocals.py)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$use_demucs" = true ]; then
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
    # Set VAD min duration
    min_duration=0.255

    if [[ "x${sad_type}" == "xoracle" ]]; then
        # Oracle SAD: handling overlapping or too short regions in ground truth RTTM
        while read -r utt wav_path; do
            python3 wespeaker/diar/make_oracle_sad.py \
                    --rttm data/voxconverse-master/${partition}/${utt}.rttm \
                    --min-duration $min_duration
        done < "${wav_scp}" > data/${partition}/oracle_sad
    fi

    if [[ "x${sad_type}" == "xsystem" ]]; then
       # System SAD: applying 'silero' VAD
       python3 wespeaker/diar/make_system_sad.py \
               --scp "${wav_scp}" \
               --min-duration $min_duration > data/${partition}/system_sad
       sad_lines=$(wc -l < "data/${partition}/system_sad")
       echo "System SAD: ${sad_lines} segments -> data/${partition}/system_sad"
    fi
    if [[ "x${sad_type}" == "xoracle" ]]; then
        sad_lines=$(wc -l < "data/${partition}/oracle_sad")
    fi
    stage_done 4 "VAD / SAD (${sad_type}, ${sad_lines:-?} segments)"
fi


# Extract fbank features
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    resolve_wav_scp

    [ -d "exp/${partition}_${sad_type}_sad_fbank" ] && rm -r exp/${partition}_${sad_type}_sad_fbank

    echo "Make Fbank features -> exp/${partition}_${sad_type}_sad_fbank"
    echo "..."
    bash local/make_fbank.sh \
            --scp "${wav_scp}" \
            --segments data/${partition}/${sad_type}_sad \
            --store_dir exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --verbose true \
            --nj 24 || exit 1
    stage_done 5 "Fbank -> exp/${partition}_${sad_type}_sad_fbank"
fi

# Extract embeddings
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

    [ -d "exp/${partition}_${sad_type}_sad_embedding" ] && rm -r exp/${partition}_${sad_type}_sad_embedding

    echo "Extract embeddings -> exp/${partition}_${sad_type}_sad_embedding"
    echo "..."
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
    stage_done 6 "ResNet ONNX embeddings (${emb_lines} lines in emb.scp)"
fi


# Applying spectral or ump+hdbscan clustering algorithm
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then

    [ -f "exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_labels" ] && rm exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_labels

    echo "Doing ${cluster_type} clustering and store the result in exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_labels"
    echo "..."
    python3 wespeaker/diar/${cluster_type}_clusterer.py \
            --scp exp/${partition}_${sad_type}_sad_embedding/emb.scp \
            --output exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_labels
    stage_done 7 "${cluster_type} clustering done"
fi


# Convert labels to RTTMs
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    python3 wespeaker/diar/make_rttm.py \
            --labels exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_labels \
            --channel 1 > exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm
    stage_done 8 "RTTM -> exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm"
fi


# Evaluate the result
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    ref_dir=data/voxconverse-master/
    #ref_dir=data/VoxSRC2023/voxconverse/
    echo -e "Get the DER results\n..."
    perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
         -c 0.25 \
         -r <(cat ${ref_dir}/${partition}/*.rttm) \
         -s exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm 2>&1 | tee exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_res

    if [ ${get_each_file_res} -eq 1 ];then
        single_file_res_dir=exp/${cluster_type}_cluster/${partition}_${sad_type}_single_file_res
        mkdir -p $single_file_res_dir
        echo -e "\nGet the DER results for each file -> ${single_file_res_dir}\n..."

        awk '{print $2}' exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm | sort -u  | while read file_name; do
            perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
                 -c 0.25 \
                 -r <(cat ${ref_dir}/${partition}/${file_name}.rttm) \
                 -s <(grep "${file_name}" exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_rttm) > ${single_file_res_dir}/${partition}_${file_name}_res
        done
        echo "Per-file DER written under ${single_file_res_dir}/"
    fi

    res_file="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad_res"
    der_line=$(grep 'OVERALL SPEAKER DIARIZATION ERROR' "$res_file" 2>/dev/null || true)
    der_pct=$(echo "$der_line" | grep -oP '[\d.]+(?= percent)' || echo "?")
    stage_done 9 "DER = ${der_pct}% (${partition}, ${sad_type} SAD, ${cluster_type})"
fi

print_summary

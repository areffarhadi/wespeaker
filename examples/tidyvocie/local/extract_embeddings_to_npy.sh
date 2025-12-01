#!/bin/bash

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

exp_dir=''
model_path=''
data=data
dataset=""
output_dir=""
gpus="[0]"
wav_dir=""  # Optional: directory containing wav files (alternative to wav.scp)

. tools/parse_options.sh
set -e

config_file="${exp_dir}/config.yaml"

# Extract GPU ID
gpu_id=$(echo $gpus | tr -d '[]' | cut -d',' -f1)

# Python path
python_bin=/local/scratch/arfarh/wespeaker/we-env/bin/python3

# Determine which mode to use
if [ -n "${wav_dir}" ]; then
  # Mode 1: Extract from wav directory
  echo "Mode: Extracting from directory: ${wav_dir}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${python_bin} -u wespeaker/bin/extract_to_npy.py \
    --config ${config_file} \
    --model_path ${model_path} \
    --wav_dir ${wav_dir} \
    --output_dir ${output_dir}
else
  # Mode 2: Extract from wav.scp file
  wav_scp="${data}/${dataset}/wav.scp"
  echo "Mode: Extracting from wav.scp: ${wav_scp}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${python_bin} -u wespeaker/bin/extract_to_npy.py \
    --config ${config_file} \
    --model_path ${model_path} \
    --wav_scp ${wav_scp} \
    --output_dir ${output_dir}
fi


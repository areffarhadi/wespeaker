#!/bin/bash

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

exp_dir=''
model_path=''
nj=4
gpus="[0,1]"
data_type="shard"  # shard/raw/feat
data=data
dataset=""  # Single dataset to extract (e.g., "tidyvoice", "vox1", "vox2_dev")
config=''

. tools/parse_options.sh
set -e

if [ -z "$dataset" ]; then
  echo "Error: --dataset parameter is required"
  echo "Usage: $0 --dataset <dataset_name> --exp_dir <exp_dir> --model_path <model_path>"
  exit 1
fi

echo "Starting embedding extraction for dataset: $dataset"
echo "exp_dir: ${exp_dir}"
echo "model_path: ${model_path}"
echo "data_type: ${data_type}"
echo "data: ${data}"
echo "gpus: ${gpus}"
echo "config: ${config}"

# Convert GPU list [4,5,6] to array (4 5 6)
gpu_array=($(echo $gpus | tr -d '[]' | tr ',' ' '))
num_gpus=${#gpu_array[@]}
echo "Found $num_gpus GPUs: ${gpu_array[@]}"

# Setup paths for the specified dataset
data_list_path="${data}/${dataset}/${data_type}.list"
data_scp_path="${data}/${dataset}/wav.scp"

# Set batch size and workers based on dataset type
if [ "$dataset" = "vox2_dev" ]; then
  batch_size=16
  num_workers=4
else
  # For evaluation datasets like candor, vox1, etc.
  batch_size=1  # batch_size of test set must be 1 !!!
  num_workers=1
fi

echo "Checking data paths for dataset: $dataset"
echo "Checking ${data_list_path}"
if [ -f "${data_list_path}" ]; then
  echo "Found ${data_list_path}"
else
  echo "ERROR: ${data_list_path} not found"
  exit 1
fi

echo "Checking ${data_scp_path}"
if [ -f "${data_scp_path}" ]; then
  echo "Found ${data_scp_path}"
else
  echo "ERROR: ${data_scp_path} not found"
  exit 1
fi

wavs_num=$(wc -l ${data_scp_path} | awk '{print $1}')
echo "Processing ${dataset} with ${wavs_num} wavs"

# Use first GPU
current_gpu=${gpu_array[0]}
echo "Using GPU $current_gpu for ${dataset}"

#bash tools/extract_embedding.sh --exp_dir ${exp_dir} \
bash tools/extract_embedding_fixed.sh --exp_dir ${exp_dir} \
  --model_path $model_path \
  --data_type ${data_type} \
  --data_list ${data_list_path} \
  --wavs_num ${wavs_num} \
  --store_dir ${dataset} \
  --batch_size ${batch_size} \
  --num_workers ${num_workers} \
  --nj ${nj} \
  --gpus $current_gpu \
  ${config:+--config $config}

if [ $? -eq 0 ]; then
  echo "Successfully extract embedding for $dataset"
else
  echo "Failed to extract embedding for $dataset"
  exit 1
fi

echo "Checking output directory:"
echo "Looking in: ${exp_dir}/embeddings/${dataset}"
ls -l ${exp_dir}/embeddings/${dataset} || echo "embeddings directory for $dataset not found"

echo "Embedding extraction completed for dataset: $dataset" 

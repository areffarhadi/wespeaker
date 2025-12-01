#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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

exp_dir=''
model_path=''
nj=4
gpus="[0,1]"
data_type="shard"  # shard/raw/feat
data=data
config=''  # Add config parameter

. tools/parse_options.sh
set -e

echo "=== Starting embedding extraction with: ==="
echo "exp_dir: ${exp_dir}"
echo "model_path: ${model_path}"
echo "data_type: ${data_type}"
echo "data: ${data}"
echo "gpus: ${gpus}"
echo "config: ${config}"

# Check if model exists
if [ ! -f "${model_path}" ]; then
    echo "ERROR: Model file not found: ${model_path}"
    exit 1
fi

# Check if config exists
if [ ! -f "${exp_dir}/config.yaml" ]; then
    echo "ERROR: Config file not found: ${exp_dir}/config.yaml"
    exit 1
fi

# Convert GPU list [4,5,6] to array (4 5 6)
gpu_array=($(echo $gpus | tr -d '[]' | tr ',' ' '))
num_gpus=${#gpu_array[@]}
echo "Found $num_gpus GPUs: ${gpu_array[@]}"

data_name_array=("vox2_dev" "vox1")
data_list_path_array=("${data}/vox2_dev/shard.list" "${data}/vox1/shard.list")  # Changed to look for shard.list directly
data_scp_path_array=("${data}/vox2_dev/wav.scp" "${data}/vox1/wav.scp") # to count the number of wavs
nj_array=($nj $nj)
batch_size_array=(16 1) # batch_size of test set must be 1 !!!
num_workers_array=(4 1)
count=${#data_name_array[@]}

echo "=== Checking data paths: ==="
for i in $(seq 0 $(($count - 1))); do
  echo "Checking ${data_list_path_array[$i]}"
  if [ -f "${data_list_path_array[$i]}" ]; then
    echo "Found ${data_list_path_array[$i]}"
    echo "First few lines of ${data_list_path_array[$i]}:"
    head -n 3 "${data_list_path_array[$i]}"
  else
    echo "ERROR: ${data_list_path_array[$i]} not found"
    exit 1
  fi
  
  echo "Checking ${data_scp_path_array[$i]}"
  if [ -f "${data_scp_path_array[$i]}" ]; then
    echo "Found ${data_scp_path_array[$i]}"
    echo "First few lines of ${data_scp_path_array[$i]}:"
    head -n 3 "${data_scp_path_array[$i]}"
  else
    echo "ERROR: ${data_scp_path_array[$i]} not found"
    exit 1
  fi
done

# Create output directories
for dset in "${data_name_array[@]}"; do
    mkdir -p ${exp_dir}/embeddings/${dset}/log
    echo "Created directory: ${exp_dir}/embeddings/${dset}"
done

for i in $(seq 0 $(($count - 1))); do
  wavs_num=$(wc -l ${data_scp_path_array[$i]} | awk '{print $1}')
  echo "=== Processing ${data_name_array[$i]} with ${wavs_num} wavs ==="
  
  # Use GPU round-robin
  gpu_idx=$((i % num_gpus))
  current_gpu=${gpu_array[$gpu_idx]}
  echo "Using GPU $current_gpu for ${data_name_array[$i]}"
  
  echo "Running extract_embedding.sh with:"
  echo "data_list: ${data_list_path_array[$i]}"
  echo "store_dir: ${data_name_array[$i]}"
  echo "batch_size: ${batch_size_array[$i]}"
  echo "num_workers: ${num_workers_array[$i]}"
  echo "nj: ${nj_array[$i]}"
  
  bash tools/extract_embedding1.sh --exp_dir ${exp_dir} \
    --model_path $model_path \
    --data_type ${data_type} \
    --data_list ${data_list_path_array[$i]} \
    --wavs_num ${wavs_num} \
    --store_dir ${data_name_array[$i]} \
    --batch_size ${batch_size_array[$i]} \
    --num_workers ${num_workers_array[$i]} \
    --nj ${nj_array[$i]} \
    --gpus $current_gpu \
    ${config:+--config $config}
    
  # Check if extraction was successful
  if [ $? -ne 0 ]; then
    echo "ERROR: Extraction failed for ${data_name_array[$i]}"
    echo "Check logs in ${exp_dir}/embeddings/${data_name_array[$i]}/log/"
    exit 1
  fi
done

echo "=== Checking output directories: ==="
echo "Looking in: ${exp_dir}/embeddings"
ls -l ${exp_dir}/embeddings || echo "embeddings directory not found"

for dset in "${data_name_array[@]}"; do
    echo "=== Contents of ${exp_dir}/embeddings/${dset}: ==="
    ls -l ${exp_dir}/embeddings/${dset} || echo "Directory not found"
    
    echo "=== Log files for ${dset}: ==="
    ls -l ${exp_dir}/embeddings/${dset}/log/ || echo "No log files found"
    
    if [ -d "${exp_dir}/embeddings/${dset}/log" ]; then
        echo "=== Last few lines of log files: ==="
        for log in ${exp_dir}/embeddings/${dset}/log/extract_*.log; do
            if [ -f "$log" ]; then
                echo "=== $log ==="
                tail -n 5 "$log"
            fi
        done
    fi
done

echo "Embedding dir is (${exp_dir}/embeddings)."


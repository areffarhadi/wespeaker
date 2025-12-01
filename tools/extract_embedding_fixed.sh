#!/bin/bash

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge


exp_dir='exp/XVEC'
model_path='avg_model.pt'
data_type='shard'  # shard/raw/feat
data_list='shard.list'  # shard.list/raw.list/feat.list
wavs_num=
store_dir=
batch_size=1
num_workers=1
nj=1  # Force single job for reliability
reverb_data=data/rirs/lmdb
noise_data=data/musan/lmdb
aug_prob=0.0
gpus="[0,1]"

. tools/parse_options.sh
set -e

embed_dir=${exp_dir}/embeddings/${store_dir}
log_dir=${embed_dir}/log
[ ! -d ${log_dir} ] && mkdir -p ${log_dir}

echo "Starting embedding extraction with fixed script"
echo "exp_dir: ${exp_dir}"
echo "model_path: ${model_path}"
echo "data_type: ${data_type}"
echo "data_list: ${data_list}"
echo "store_dir: ${store_dir}"
echo "batch_size: ${batch_size}"
echo "num_workers: ${num_workers}"
echo "gpus: ${gpus}"

# Convert GPU list [4,5,6] to array (4 5 6)
gpu_array=($(echo $gpus | tr -d '[]' | tr ',' ' '))
num_gpus=${#gpu_array[@]}
echo "Available GPUs: ${gpu_array[@]}"

# Use first GPU
current_gpu=${gpu_array[0]}
echo "Using GPU: ${current_gpu}"

# Check if data list exists
if [ ! -f "${data_list}" ]; then
    echo "ERROR: Data list file ${data_list} not found!"
    exit 1
fi

data_num=$(wc -l ${data_list} | awk '{print $1}')
echo "Data list contains ${data_num} entries"

embed_ark=${embed_dir}/xvector_000.ark
embed_scp=${embed_dir}/xvector_000.scp

echo "Output files: ${embed_ark}, ${embed_scp}"

# Run extraction with single job for reliability
CUDA_VISIBLE_DEVICES=${current_gpu} python -u wespeaker/bin/extract.py \
  --config ${exp_dir}/config.yaml \
  --model_path ${model_path} \
  --data_type ${data_type} \
  --data_list ${data_list} \
  --embed_ark ${embed_ark} \
  --batch-size ${batch_size} \
  --num-workers ${num_workers} \
  --reverb_data ${reverb_data} \
  --noise_data ${noise_data} \
  --aug-prob ${aug_prob} \
  >${log_dir}/extract.log 2>&1

# Check extraction result
if [ $? -eq 0 ]; then
    echo "Extraction completed successfully"
    
    # Create combined scp file (even though we only have one)
    cp ${embed_scp} ${embed_dir}/xvector.scp
    
    embed_num=$(wc -l ${embed_dir}/xvector.scp | awk '{print $1}')
    echo "Extracted ${embed_num} embeddings"
    
    if [ $embed_num -eq $wavs_num ]; then
        echo "Successfully extract embedding for ${store_dir}" | tee ${embed_dir}/extract.result
    else
        echo "Warning: Expected ${wavs_num} embeddings but got ${embed_num}" | tee ${embed_dir}/extract.result
        echo "This may be due to corrupted audio files or shards"
    fi
else
    echo "Failed to extract embedding for ${store_dir}" | tee ${embed_dir}/extract.result
    echo "Check log file: ${log_dir}/extract.log"
    exit 1
fi





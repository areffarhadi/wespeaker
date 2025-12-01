#!/bin/bash

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

. ./path.sh || exit 1

stage=9
stop_stage=9


HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2025

data=data
data_type="shard"  # shard/raw

# CommonVoice DataCollective API Key
# Get your API key from: https://datacollective.mozillafoundation.org/api-reference
TIDYVOICE_API_KEY="Enter your Mozilla CommonVoice API key here"


######## Resnet34 Multi Lingual
exp_dir=exp/samresnet34_voxblink_ft_tidy
config=conf/tidyvoice_resnet34.yaml

gpus="[0,2]"
num_avg=5
checkpoint=

# ====================== EVAL DATASET SELECTION ======================

eval_dataset="tidyvoice_dev"  

# Dataset-specific trial configurations
case $eval_dataset in
  "tidyvoice_eval1")
    trials="trials.kaldi"
    ;;
  "tidyvoice_eval2")
    trials="trials.kaldi"
    ;;
  "tidyvoice_dev")
    trials="trials.kaldi"
    ;;
  *)
    echo "Warning: Unknown dataset $eval_dataset, using default trial file"
    trials="trials.kaldi"
    ;;
esac

echo "Using evaluation dataset: $eval_dataset"
echo "Using trial file: $trials"

score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/voxblink_resnet34_ft.yaml

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 1 --stop_stage 4 --data ${data} --api_key ${TIDYVOICE_API_KEY}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in tidyvoice_train $eval_dataset; do
    # Skip if eval_dataset is the same as tidyvoice_train to avoid duplication
    if [ "$dset" = "tidyvoice_train" ] && [ "$eval_dataset" = "tidyvoice_train" ]; then
      continue
    fi
    
    echo "Processing dataset: $dset"
    # Check if wav.scp exists for this dataset
    if [ ! -f "${data}/$dset/wav.scp" ]; then
      echo "Warning: ${data}/$dset/wav.scp not found, skipping $dset"
      continue
    fi
    
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  
  # Convert all musan data to LMDB (if it exists)
  if [ -f "${data}/musan/wav.scp" ]; then
    echo "Converting MUSAN to LMDB..."
    python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  else
    echo "Warning: ${data}/musan/wav.scp not found, skipping MUSAN LMDB conversion"
  fi
  
  # Convert all rirs data to LMDB (if it exists)
  if [ -f "${data}/rirs/wav.scp" ]; then
    echo "Converting RIRS to LMDB..."
    python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
  else
    echo "Warning: ${data}/rirs/wav.scp not found, skipping RIRS LMDB conversion"
  fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/tidyvoice_train/${data_type}.list \
      --train_label ${data}/tidyvoice_train/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  #avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model
  if [[ $config == *repvgg*.yaml ]]; then
    echo "convert repvgg model ..."
    python wespeaker/models/convert_repvgg.py \
      --config $exp_dir/config.yaml \
      --load $avg_model \
      --save $exp_dir/models/convert_model.pt
    model_path=$exp_dir/models/convert_model.pt
  fi

  echo "Extract embeddings for evaluation dataset: $eval_dataset ..."
  # Extract embeddings only for evaluation dataset (skip tidyvoice_train to save time)
  dset=$eval_dataset
  echo "Extracting embeddings for: $dset"
  if [ -f "${data}/$dset/wav.scp" ]; then
    local/extract_single_dataset.sh \
      --exp_dir $exp_dir --model_path $model_path \
      --nj 1 --gpus $gpus --data_type $data_type \
      --data ${data} --dataset $dset
  else
    echo "Warning: ${data}/$dset/wav.scp not found, skipping $dset"
  fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score evaluation dataset: $eval_dataset ..."
  local/score_custom.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --eval_dataset $eval_dataset \
    --trials "$trials" \
    --cal_mean True
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm for evaluation dataset: $eval_dataset ..."
  echo "Note: Using tidyvoice_dev as fixed cohort set for score normalization"
  local/score_norm_custom.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set tidyvoice_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --eval_dataset $eval_dataset \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score calibration for evaluation dataset: $eval_dataset ..."
  echo "Note: Using tidyvoice_dev as fixed cohort set for score calibration"
  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method $score_norm_method \
    --calibration_trial "tidyvoice_dev_cali.kaldi" \
    --cohort_set tidyvoice_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Starting large-margin fine-tuning from pretrained model..."
  echo "Using tidyvoice_train for fine-tuning..."

  # Use tidyvoice_train data (created in stage 2)
  train_list=${data}/tidyvoice_train/shard.list
  train_label=${data}/tidyvoice_train/utt2spk

  # Set random port for distributed training
  export MASTER_PORT=$((12000 + RANDOM % 1000))

  torchrun --standalone --nproc_per_node=2 \
    wespeaker/bin/train.py \
      --config conf/tidyvoice_resnet34_ft.yaml \
      --exp_dir exp/samresnet34_voxblink_ft_tidy \
      --gpus "[0,2]" \
      --data_type ${data_type} \
      --train_data ${train_list} \
      --train_label ${train_label} \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --num_avg 1
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "Extract embeddings to numpy format (.npy) for evaluation dataset: $eval_dataset ..."
  
  # Set model_path if not already set (in case stage 4 was skipped)
  if [ -z "${model_path}" ]; then
    model_path=$exp_dir/models/avg_model.pt
    if [[ $config == *repvgg*.yaml ]]; then
      model_path=$exp_dir/models/convert_model.pt
    fi
    echo "Using model: ${model_path}"
  fi
  
  local/extract_embeddings_to_npy.sh \
    --exp_dir ${exp_dir} \
    --model_path ${model_path} \
    --data ${data} \
    --dataset ${eval_dataset} \
    --output_dir ${exp_dir}/embeddings_npy/${eval_dataset} \
    --gpus ${gpus}
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "Score evaluation dataset from numpy embeddings: $eval_dataset ..."
  local/score_from_npy.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir ${exp_dir} \
    --eval_dataset ${eval_dataset} \
    --trials "${trials}" \
    --npy_embedding_dir ${exp_dir}/embeddings_npy/${eval_dataset} \
    --cal_mean True
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  echo "Extract embeddings from a directory with .wav files ..."

  exp_dir=/local/scratch/arfarh/wespeaker/wespeaker/examples/tidyvoice
  model_path=$exp_dir/models/avg_model.pt
  wav_input_dir="/path/to/your/wav/files"
  npy_output_dir_custom="${exp_dir}/embeddings_npy/custom_dataset"

  local/extract_embeddings_to_npy.sh \
    --exp_dir ${exp_dir} \
    --model_path ${model_path} \
    --wav_dir ${wav_input_dir} \
    --output_dir ${npy_output_dir_custom} \
    --gpus ${gpus}
  
  echo "Embeddings saved to: ${npy_output_dir_custom}"
  echo "Folder structure preserved from: ${wav_input_dir}"
fi


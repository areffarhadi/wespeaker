#!/bin/bash

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

exp_dir=
trials="vox1_O_cleaned.kaldi"
data=data
eval_dataset="vox1"
npy_embedding_dir=""
cal_mean=True

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

# Python path
python_bin=/local/scratch/arfarh/wespeaker/we-env/bin/python3

echo "Using evaluation dataset: $eval_dataset"
echo "Using trials: $trials"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "apply cosine scoring from numpy embeddings ..."
  mkdir -p ${exp_dir}/scores_npy
  trials_dir=${data}/${eval_dataset}/trials
  for x in $trials; do
    echo $x
    ${python_bin} wespeaker/bin/score_from_npy.py \
      --trial_file ${trials_dir}/${x} \
      --embedding_dir ${npy_embedding_dir} \
      --output_score_file ${exp_dir}/scores_npy/${x}.score \
      --cal_mean ${cal_mean}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores_npy
  for x in $trials; do
    ${python_bin} wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${x}.score \
        2>&1 | tee -a ${scores_dir}/${eval_dataset}_npy_result

    echo "compute DET curve ..."
    ${python_bin} wespeaker/bin/compute_det.py \
        ${scores_dir}/${x}.score
  done
fi


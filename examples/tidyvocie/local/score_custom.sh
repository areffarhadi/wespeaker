#!/bin/bash

# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data
eval_dataset="vox1"  # Default evaluation dataset
cal_mean=False  # Enable mean normalization: True/False

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

echo "Using evaluation dataset: $eval_dataset"
echo "Using trials: $trials"
echo "Mean normalization: $cal_mean"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "apply cosine scoring ..."
  mkdir -p ${exp_dir}/scores
  trials_dir=${data}/${eval_dataset}/trials
  for x in $trials; do
    echo $x
    python wespeaker/bin/score.py \
      --exp_dir ${exp_dir} \
      --eval_scp_path ${exp_dir}/embeddings/${eval_dataset}/xvector.scp \
      --cal_mean ${cal_mean} \
      --cal_mean_dir ${exp_dir}/embeddings/vox2_dev \
      ${trials_dir}/${x}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores
  for x in $trials; do
    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${x}.score \
        2>&1 | tee -a ${scores_dir}/${eval_dataset}_cos_result

    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir}/${x}.score
  done
fi


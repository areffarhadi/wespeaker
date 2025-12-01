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


# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge



stage=1
stop_stage=4
data=data
api_key=""

. tools/parse_options.sh || exit 1

data=`realpath ${data}`
download_dir=${data}/download_data
rawdata_dir=${data}/raw_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Downloading TidyVoice dataset and augmentation datasets..."
  echo "This may take a long time depending on your internet connection."

  ./local/download_tidyvoice.sh --download_dir ${download_dir} --rawdata_dir ${rawdata_dir} --api_key ${api_key}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress trial files and other archives ..."
  echo "This could take some time ..."

  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  # Extract TidyVoice dataset (recursively extract all nested tar.gz files)
  if [ -d ${rawdata_dir}/TidyVoiceX_ASV ]; then
    # Check if train/dev directories already exist (dataset already extracted)
    # Look for directories containing "Train" or "Dev" (case-insensitive)
    has_train_or_dev=$(find ${rawdata_dir}/TidyVoiceX_ASV -type d \( -iname "*train*" -o -iname "*dev*" \) 2>/dev/null | grep -E "(Train|Dev)" | head -n 1)
    
    if [ -n "${has_train_or_dev}" ]; then
      echo "TidyVoice dataset already extracted at ${rawdata_dir}/TidyVoiceX_ASV"
      echo "  Found data directories: $(find ${rawdata_dir}/TidyVoiceX_ASV -type d \( -iname "*train*" -o -iname "*dev*" \) 2>/dev/null | grep -E "(Train|Dev)" | head -n 2 | xargs -I {} basename {} | tr '\n' ' ')"
    else
      # Check if there are any tar.gz files to extract
      tar_files_count=$(find ${rawdata_dir}/TidyVoiceX_ASV -name "*.tar.gz" -type f 2>/dev/null | wc -l)
      
      if [ "${tar_files_count}" -eq 0 ]; then
        echo "No tar.gz files found in ${rawdata_dir}/TidyVoiceX_ASV"
        echo "Dataset may already be extracted or structure is different than expected."
      else
        echo "Extracting TidyVoice dataset (found ${tar_files_count} tar.gz file(s), may contain nested archives)..."
        
        # Recursively extract all tar.gz files until train/dev directories are found
        max_iterations=10  # Prevent infinite loops
        iteration=0
        
        while [ ${iteration} -lt ${max_iterations} ]; do
          # Check if train/dev directories exist now (extraction complete)
          # Look for directories containing "Train" or "Dev" (case-insensitive)
          has_train_or_dev=$(find ${rawdata_dir}/TidyVoiceX_ASV -type d \( -iname "*train*" -o -iname "*dev*" \) 2>/dev/null | grep -E "(Train|Dev)" | head -n 1)
          if [ -n "${has_train_or_dev}" ]; then
            echo "TidyVoice dataset extracted successfully!"
            break
          fi
          
          # Find all tar.gz files and sort by depth (shallowest first)
          # This ensures outer archives are extracted before inner ones
          tar_files=$(find ${rawdata_dir}/TidyVoiceX_ASV -name "*.tar.gz" -type f 2>/dev/null | \
            awk -F'/' '{print NF, $0}' | sort -n | cut -d' ' -f2-)
          
          if [ -z "${tar_files}" ]; then
            # No more tar.gz files found
            echo "No more tar.gz files to extract, but train/dev directories not found."
            echo "Please check the dataset structure manually."
            break
          fi
          
          # Extract each tar.gz file (outer archives first)
          for tar_file in ${tar_files}; do
            if [ -f "${tar_file}" ]; then
              echo "  Extracting: ${tar_file}"
              tar_dir=$(dirname "${tar_file}")
              tar_name=$(basename "${tar_file}")
              cd "${tar_dir}"
              tar -xzf "${tar_name}" 2>/dev/null
              # Remove the tar.gz file after extraction to avoid re-extracting
              rm -f "${tar_name}"
              cd - > /dev/null
            fi
          done
          
          iteration=$((iteration + 1))
        done
        
        # Final verification
        has_train_or_dev=$(find ${rawdata_dir}/TidyVoiceX_ASV -type d \( -iname "*train*" -o -iname "*dev*" \) 2>/dev/null | grep -E "(Train|Dev)" | head -n 1)
        if [ -z "${has_train_or_dev}" ]; then
          echo "Warning: TidyVoice dataset extraction completed but train/dev directories not found."
          echo "Please check the dataset structure manually."
        fi
      fi
    fi
  fi

  # Unzip TidyVoice trial files
  if [ ! -d ${data}/tidyvoice/trials ] || [ -z "$(ls -A ${data}/tidyvoice/trials 2>/dev/null)" ]; then
    echo "Extracting TidyVoice trial files..."
    mkdir -p ${data}/tidyvoice/trials
    unzip -o ${download_dir}/tidyvoice_trials.zip -d ${data}/tidyvoice/trials
    echo "Trial files extracted to ${data}/tidyvoice/trials"
  else
    echo "Trial files already extracted at ${data}/tidyvoice/trials"
  fi
  
  # Convert trial files to Kaldi format and place in tidyvoice_dev/trials
  # Look for trial files (handle both TidyVocieX and TidyVoiceX naming)
  trial_file=$(find ${data}/tidyvoice/trials -name "*trial*.txt" -type f 2>/dev/null | head -n 1)
  if [ -n "${trial_file}" ]; then
    # Create trials directory in tidyvoice_dev (evaluation data directory)
    mkdir -p ${data}/tidyvoice_dev/trials
    
    kaldi_trial_file=${data}/tidyvoice_dev/trials/trials.kaldi
    
    if [ ! -f "${kaldi_trial_file}" ]; then
      echo "Converting trial file to Kaldi format..."
      # Convert format: 0/1 utterance1 utterance2 -> utterance1 utterance2 nontarget/target
      awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${trial_file} > ${kaldi_trial_file}
      echo "Trial file converted and saved to: ${kaldi_trial_file}"
    else
      echo "Trial file already converted to Kaldi format: ${kaldi_trial_file}"
    fi
  else
    echo "Warning: No trial file found in ${data}/tidyvoice/trials"
  fi

  # Extract MUSAN for data augmentation
  if [ -f ${download_dir}/musan.tar.gz ] && [ ! -d ${rawdata_dir}/musan ]; then
    echo "Extracting musan..."
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
    echo "MUSAN extracted successfully!"
  elif [ -d ${rawdata_dir}/musan ]; then
    echo "MUSAN already extracted at ${rawdata_dir}/musan"
  fi

  # Extract RIRS_NOISES for data augmentation
  if [ -f ${download_dir}/rirs_noises.zip ] && [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    echo "Extracting RIRS_NOISES..."
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
    echo "RIRS_NOISES extracted successfully!"
  elif [ -d ${rawdata_dir}/RIRS_NOISES ]; then
    echo "RIRS_NOISES already extracted at ${rawdata_dir}/RIRS_NOISES"
  fi

  # Optional: Extract vox1 if it exists (for testing)
  if [ -f ${download_dir}/vox1_test_wav.zip ] && [ ! -d ${rawdata_dir}/voxceleb1 ]; then
    echo "Extracting voxceleb1 test set..."
    mkdir -p ${rawdata_dir}/voxceleb1/test ${rawdata_dir}/voxceleb1/dev
    unzip ${download_dir}/vox1_test_wav.zip -d ${rawdata_dir}/voxceleb1/test
  fi

  echo "Decompress success !!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "This stage is reserved for audio format conversion if needed."
  echo "TidyVoice dataset is already in WAV format, skipping..."
  echo "Stage 3 completed!"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for TidyVoice dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/tidyvoice_train ${data}/tidyvoice_dev

  # Find the actual TidyVoice data directory structure
  # Adjust this path based on how the dataset is organized after download
  tidyvoice_base=${rawdata_dir}/TidyVoiceX_ASV
  
  # Check if the dataset exists
  if [ ! -d ${tidyvoice_base} ]; then
    echo "ERROR: TidyVoice dataset not found at ${tidyvoice_base}"
    echo "Please run stage 1 first to download the dataset."
    exit 1
  fi
  
  # Prepare train set (search recursively for directories containing "Train")
  train_dir=$(find ${tidyvoice_base} -type d -iname "*train*" 2>/dev/null | grep -i "train" | head -n 1)
  if [ -n "${train_dir}" ]; then
    echo "Processing training set from ${train_dir}..."
    find ${train_dir} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/tidyvoice_train/wav.scp
    awk '{print $1}' ${data}/tidyvoice_train/wav.scp | awk -F "/" '{print $0,$1}' >${data}/tidyvoice_train/utt2spk
    ./tools/utt2spk_to_spk2utt.pl ${data}/tidyvoice_train/utt2spk >${data}/tidyvoice_train/spk2utt
    echo "Training set prepared: $(wc -l < ${data}/tidyvoice_train/wav.scp) utterances"
  else
    echo "Warning: Training set directory not found in ${tidyvoice_base}"
  fi
  
  # Prepare dev/validation set (search recursively for directories containing "Dev")
  dev_dir=$(find ${tidyvoice_base} -type d -iname "*dev*" 2>/dev/null | grep -i "dev" | head -n 1)
  if [ -n "${dev_dir}" ]; then
    echo "Processing development set from ${dev_dir}..."
    find ${dev_dir} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/tidyvoice_dev/wav.scp
    awk '{print $1}' ${data}/tidyvoice_dev/wav.scp | awk -F "/" '{print $0,$1}' >${data}/tidyvoice_dev/utt2spk
    ./tools/utt2spk_to_spk2utt.pl ${data}/tidyvoice_dev/utt2spk >${data}/tidyvoice_dev/spk2utt
    echo "Development set prepared: $(wc -l < ${data}/tidyvoice_dev/wav.scp) utterances"
  else
    echo "Warning: Development set directory not found in ${tidyvoice_base}"
  fi
  
  # Optional: prepare musan and rirs for data augmentation if they exist
  if [ -d ${rawdata_dir}/musan ]; then
    mkdir -p ${data}/musan
    find ${rawdata_dir}/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
    echo "MUSAN prepared: $(wc -l < ${data}/musan/wav.scp) files"
  fi
  
  if [ -d ${rawdata_dir}/RIRS_NOISES ]; then
    mkdir -p ${data}/rirs
    find ${rawdata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp
    echo "RIRS prepared: $(wc -l < ${data}/rirs/wav.scp) files"
  fi

  echo "Success !!!"
fi

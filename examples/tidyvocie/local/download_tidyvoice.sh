#!/bin/bash

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge


download_dir=data/download_data
rawdata_dir=data/raw_data
api_key=""

. tools/parse_options.sh || exit 1

[ ! -d ${download_dir} ] && mkdir -p ${download_dir}
[ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

download_dir=`realpath ${download_dir}`
[ -z "${rawdata_dir}" ] && rawdata_dir=$(dirname ${download_dir})/raw_data
[ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}
rawdata_dir=`realpath ${rawdata_dir}`

# Get the directory where this script is located
local_dir=$(dirname "$(readlink -f "$0")")
download_py_script=${local_dir}/download_tidyvoice.py

# Download TidyVoice dataset using DataCollective API
# Check if directory exists and contains data (not just tar.gz file)
if [ ! -d ${rawdata_dir}/TidyVoiceX_ASV ]; then
  echo "Downloading TidyVoice dataset..."
  if [ ! -f ${download_py_script} ]; then
    echo "ERROR: download_tidyvoice.py not found at ${download_py_script}"
    exit 1
  fi
  python3 ${download_py_script} ${rawdata_dir}/TidyVoiceX_ASV ${api_key}
  if [ $? -ne 0 ]; then
    echo "Failed to download TidyVoice dataset. Please check your internet connection and API key."
    exit 1
  fi
else
  # Check if it's just a tar.gz file or if data is already extracted
  has_data=$(find ${rawdata_dir}/TidyVoiceX_ASV -maxdepth 1 -type d \( -iname "train" -o -iname "dev" \) 2>/dev/null | head -n 1)
  if [ -z "${has_data}" ]; then
    # Directory exists but no train/dev folders - might just have tar.gz
    echo "TidyVoice dataset directory exists but data may not be extracted yet."
    echo "Will be extracted in stage 2 of prepare_data.sh"
  else
    echo "TidyVoice dataset already exists at ${rawdata_dir}/TidyVoiceX_ASV"
  fi
fi

# Download TidyVoice trial files
if [ ! -f ${download_dir}/tidyvoice_trials.zip ]; then
  echo "Downloading TidyVoice trial files..."
  wget --no-check-certificate \
    "https://drive.usercontent.google.com/download?id=1OLEKewhcGi_W_gmqEDpjx-fZwQGz2kWU&export=download&authuser=0&confirm=t&uuid=d047ecdd-7c7d-4b9c-9fbd-255a3b8c608e&at=ALWLOp6wIkymu3J13cOuKzfp_VAZ%3A1764519910241" \
    -O ${download_dir}/tidyvoice_trials.zip
  if [ $? -ne 0 ]; then
    echo "Failed to download trial files. Please check your internet connection."
    exit 1
  fi
  # Verify file size is reasonable (should be around 142MB)
  if [ -f ${download_dir}/tidyvoice_trials.zip ]; then
    file_size=$(wc -c < ${download_dir}/tidyvoice_trials.zip)
    if [ "${file_size}" -lt 100000000 ]; then
      echo "Warning: Downloaded trial file seems too small (${file_size} bytes). File may be corrupted."
    fi
  fi
else
  echo "TidyVoice trial files already downloaded at ${download_dir}/tidyvoice_trials.zip"
fi

# Download MUSAN dataset for data augmentation
if [ ! -f ${download_dir}/musan.tar.gz ]; then
  echo "Downloading musan.tar.gz ..."
  wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${download_dir}
  if [ $? -ne 0 ]; then
    echo "Failed to download musan.tar.gz. Please check your internet connection."
    exit 1
  fi
fi

# Verify MUSAN MD5 checksum (re-download if corrupted)
if [ -f ${download_dir}/musan.tar.gz ]; then
  md5=$(md5sum ${download_dir}/musan.tar.gz | awk '{print $1}')
  if [ "$md5" != "0c472d4fc0c5141eca47ad1ffeb2a7df" ]; then
    echo "Wrong md5sum of musan.tar.gz. Expected: 0c472d4fc0c5141eca47ad1ffeb2a7df, Got: $md5"
    echo "File may be corrupted. Re-downloading..."
    rm -f ${download_dir}/musan.tar.gz
    wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${download_dir}
    if [ $? -ne 0 ]; then
      echo "Failed to re-download musan.tar.gz. Please check your internet connection."
      exit 1
    fi
    md5=$(md5sum ${download_dir}/musan.tar.gz | awk '{print $1}')
    if [ "$md5" != "0c472d4fc0c5141eca47ad1ffeb2a7df" ]; then
      echo "Still wrong md5sum after re-download. Please check manually."
      exit 1
    fi
  fi
  echo "MUSAN already downloaded and verified at ${download_dir}/musan.tar.gz"
fi

# Download RIRS_NOISES dataset for data augmentation
if [ ! -f ${download_dir}/rirs_noises.zip ]; then
  echo "Downloading rirs_noises.zip ..."
  wget --no-check-certificate https://us.openslr.org/resources/28/rirs_noises.zip -P ${download_dir}
  if [ $? -ne 0 ]; then
    echo "Failed to download rirs_noises.zip. Please check your internet connection."
    exit 1
  fi
fi

# Verify RIRS_NOISES MD5 checksum (re-download if corrupted)
if [ -f ${download_dir}/rirs_noises.zip ]; then
  md5=$(md5sum ${download_dir}/rirs_noises.zip | awk '{print $1}')
  if [ "$md5" != "e6f48e257286e05de56413b4779d8ffb" ]; then
    echo "Wrong md5sum of rirs_noises.zip. Expected: e6f48e257286e05de56413b4779d8ffb, Got: $md5"
    echo "File may be corrupted. Re-downloading..."
    rm -f ${download_dir}/rirs_noises.zip
    wget --no-check-certificate https://us.openslr.org/resources/28/rirs_noises.zip -P ${download_dir}
    if [ $? -ne 0 ]; then
      echo "Failed to re-download rirs_noises.zip. Please check your internet connection."
      exit 1
    fi
    md5=$(md5sum ${download_dir}/rirs_noises.zip | awk '{print $1}')
    if [ "$md5" != "e6f48e257286e05de56413b4779d8ffb" ]; then
      echo "Still wrong md5sum after re-download. Please check manually."
      exit 1
    fi
  fi
  echo "RIRS_NOISES already downloaded and verified at ${download_dir}/rirs_noises.zip"
fi

echo "Download success !!!"


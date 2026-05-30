#!/bin/bash
# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
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

scp=''
segments=''
store_dir=''
subseg_cmn=true
nj=1
verbose=false

. tools/parse_options.sh

split_dir=$store_dir/split_scp
log_dir=$store_dir/log
mkdir -p $split_dir
mkdir -p $log_dir

# Split wav.scp into chunks. GNU split creates only as many files as needed (often < nj);
# do NOT loop 0..nj-1 or workers hit missing split_scp_XXX and Python exits with FileNotFoundError.
file_len=`wc -l $scp | awk '{print $1}'`
subfile_len=$[$file_len / $nj + 1]
prefix='split'
split -l $subfile_len -d -a 3 $scp ${split_dir}/${prefix}_scp_

shopt -s nullglob
split_files=( "${split_dir}/${prefix}_scp_"* )
if [ ${#split_files[@]} -eq 0 ]; then
    echo "$0: split produced no chunk files under ${split_dir} (is $scp empty?)" >&2
    exit 1
fi

pids=()
logfs=()
idx=0
for scp_subfile in "${split_files[@]}"; do
    suffix=$(printf '%03d' $idx)
    write_ark=$store_dir/fbank_${suffix}.ark
    logf="${log_dir}/${prefix}.${suffix}.log"
    if [ ! -s "$scp_subfile" ]; then
        echo "$0: skip empty chunk: $scp_subfile" >&2
        idx=$((idx + 1))
        continue
    fi
    if [ "$verbose" = true ] && [ "$nj" -eq 1 ]; then
        echo "$0: running make_fbank.py (verbose, nj=1) -> $logf" >&2
        python3 wespeaker/diar/make_fbank.py \
                --scp ${scp_subfile} \
                --segments ${segments} \
                --ark-path ${write_ark} \
                --subseg-cmn ${subseg_cmn} \
                2>&1 | tee "$logf" || exit 1
    else
        python3 wespeaker/diar/make_fbank.py \
                --scp ${scp_subfile} \
                --segments ${segments} \
                --ark-path ${write_ark} \
                --subseg-cmn ${subseg_cmn} \
                > "$logf" 2>&1 &
        pids+=($!)
        logfs+=("$logf")
    fi
    idx=$((idx + 1))
done

if [ ${#pids[@]} -gt 0 ]; then
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "$0: make_fbank.py failed (pid ${pids[$i]}). Log: ${logfs[$i]}" >&2
            echo "----- tail ${logfs[$i]} -----" >&2
            tail -n 80 "${logfs[$i]}" >&2
            exit 1
        fi
    done
fi

if [ "$verbose" = true ] && [ "$nj" -gt 1 ]; then
    echo "----- $0: per-job logs (nj=$nj; use --nj 1 --verbose true for live tqdm on terminal) -----" >&2
    for f in "${log_dir}/${prefix}".*.log; do
        [ -f "$f" ] || continue
        echo "===== $f =====" >&2
        cat "$f" >&2
    done
fi

shopt -s nullglob
fbank_scps=( "${store_dir}"/fbank_*.scp )
if [ ${#fbank_scps[@]} -eq 0 ]; then
    echo "$0: no fbank_*.scp under ${store_dir}. Check ${log_dir}" >&2
    exit 1
fi
cat "${fbank_scps[@]}" > "$store_dir/fbank.scp"
echo "Finish make Fbank. ($(wc -l < "$store_dir/fbank.scp") lines in fbank.scp)"

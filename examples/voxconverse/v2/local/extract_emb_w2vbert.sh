#!/bin/bash
# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2026 — w2v-BERT-2.0 SV variant (PyTorch checkpoint via infer_w2v_bert_sv_embedding.py)
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
w2vbert_repo=''
checkpoint=''
device=cuda
store_dir=''
subseg_cmn=true
nj=1
verbose=false

batch_size=8
frame_shift=10
window_secs=1.5
period_secs=0.75

. tools/parse_options.sh

[ -z "${w2vbert_repo}" ] && echo "$0: set --w2vbert-repo to w2v-BERT-2.0_SV repo root" && exit 1
[ -z "${checkpoint}" ] && echo "$0: set --checkpoint to model_base_*.pth" && exit 1

split_dir=$store_dir/split_scp
log_dir=$store_dir/log
mkdir -p $split_dir
mkdir -p $log_dir

# Split wav.scp; GNU split creates only as many chunk files as needed (often < nj).
file_len=`wc -l $scp | awk '{print $1}'`
subfile_len=$[$file_len / $nj + 1]
prefix='split'
split -l $subfile_len -d -a 3 $scp ${split_dir}/${prefix}_scp_

shopt -s nullglob
split_files=( "${split_dir}/${prefix}_scp_"* )
if [ ${#split_files[@]} -eq 0 ]; then
    echo "$0: split produced no chunk files under ${split_dir}" >&2
    exit 1
fi

pids=()
logfs=()
idx=0
for scp_subfile in "${split_files[@]}"; do
    suffix=$(printf '%03d' $idx)
    write_ark=$store_dir/emb_${suffix}.ark
    logf="${log_dir}/${prefix}.${suffix}.log"
    if [ ! -s "$scp_subfile" ]; then
        idx=$((idx + 1))
        continue
    fi
    if [ "$verbose" = true ] && [ "$nj" -eq 1 ]; then
        echo "$0: running extract_emb_w2vbert.py (verbose, nj=1) -> $logf" >&2
        python3 wespeaker/diar/extract_emb_w2vbert.py \
                --scp ${scp_subfile} \
                --segments ${segments} \
                --ark-path ${write_ark} \
                --w2vbert-repo ${w2vbert_repo} \
                --checkpoint ${checkpoint} \
                --device ${device} \
                --batch-size ${batch_size} \
                --frame-shift ${frame_shift} \
                --window-secs ${window_secs} \
                --period-secs ${period_secs} \
                --subseg-cmn ${subseg_cmn} \
                2>&1 | tee "$logf" || exit 1
    else
        python3 wespeaker/diar/extract_emb_w2vbert.py \
                --scp ${scp_subfile} \
                --segments ${segments} \
                --ark-path ${write_ark} \
                --w2vbert-repo ${w2vbert_repo} \
                --checkpoint ${checkpoint} \
                --device ${device} \
                --batch-size ${batch_size} \
                --frame-shift ${frame_shift} \
                --window-secs ${window_secs} \
                --period-secs ${period_secs} \
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
            echo "$0: extract_emb_w2vbert.py failed (pid ${pids[$i]}). Log: ${logfs[$i]}" >&2
            echo "----- tail ${logfs[$i]} -----" >&2
            tail -n 80 "${logfs[$i]}" >&2
            exit 1
        fi
    done
fi

if [ "$verbose" = true ] && [ "$nj" -gt 1 ]; then
    echo "----- $0: per-job logs (nj=$nj) -----" >&2
    for f in "${log_dir}/${prefix}".*.log; do
        [ -f "$f" ] || continue
        echo "===== $f =====" >&2
        cat "$f" >&2
    done
fi

shopt -s nullglob
emb_scps=( "${store_dir}"/emb_*.scp )
if [ ${#emb_scps[@]} -eq 0 ]; then
    echo "$0: no emb_*.scp under ${store_dir} (Python jobs may have crashed). Check ${log_dir}" >&2
    exit 1
fi
cat "${emb_scps[@]}" > "$store_dir/emb.scp"
emb_lines=$(wc -l < "$store_dir/emb.scp")
if [ "$emb_lines" -eq 0 ]; then
    echo "$0: emb.scp is empty after concat. Check ${log_dir}" >&2
    exit 1
fi
echo "Finish extract embedding (w2v-BERT-2.0 SV). (${emb_lines} lines in emb.scp)"

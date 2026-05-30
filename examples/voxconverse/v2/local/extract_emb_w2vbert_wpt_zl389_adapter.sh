#!/bin/bash
# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2026 — WPT + W2V-BERT-2.0 + zl389 Adapter/ASP/Bottleneck (USM v2 GPU mel train)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

. ./path.sh || exit 1

scp=''
segments=''
ckpt_dir=''
usm_ftcode=''
checkpoint_name='best_model.pt'
extract_python=''
device=cuda
store_dir=''
subseg_cmn=true
nj=1
verbose=false

batch_size=16
frame_shift=10
window_secs=1.5
period_secs=0.75

. tools/parse_options.sh

[ -z "${ckpt_dir}" ] && echo "$0: set --ckpt-dir (args.json + best_model.pt)" && exit 1
if [ -z "${usm_ftcode}" ]; then
    usm_ftcode="${USM_FTCODE:-$HOME/Encode-explore/USM_FTcode}"
fi
if [ ! -d "${usm_ftcode}" ] || [ ! -f "${usm_ftcode}/main_train_simple_sv_wpt_w2vbert_mhfa_zl389_v2_gpu_mel.py" ]; then
    echo "$0: USM training code missing under ${usm_ftcode} (expected main_train_simple_sv_wpt_w2vbert_mhfa_zl389_v2_gpu_mel.py; set --usm-ftcode or USM_FTCODE)" >&2
    exit 1
fi

if [ -n "${extract_python}" ] && [ ! -x "${extract_python}" ]; then
    echo "$0: --extract-python not executable (${extract_python}); falling back." >&2
    extract_python=""
fi
if [ -z "${extract_python}" ]; then
    if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
        extract_python="${PYTHON}"
    elif [ -n "${USM_WPT_ZL389_ADAPTER_PYTHON:-}" ] && [ -x "${USM_WPT_ZL389_ADAPTER_PYTHON}" ]; then
        extract_python="${USM_WPT_ZL389_ADAPTER_PYTHON}"
    elif [ -n "${USM_WPT_MHFA_PYTHON:-}" ] && [ -x "${USM_WPT_MHFA_PYTHON}" ]; then
        extract_python="${USM_WPT_MHFA_PYTHON}"
    else
        extract_python="python3"
    fi
fi

args_json="${ckpt_dir}/args.json"
best_pt="${ckpt_dir}/${checkpoint_name}"
if [ ! -f "${args_json}" ]; then
    echo "$0: missing ${args_json}" >&2
    exit 1
fi
if [ ! -f "${best_pt}" ]; then
    echo "$0: missing ${best_pt}" >&2
    exit 1
fi

split_dir=$store_dir/split_scp
log_dir=$store_dir/log
mkdir -p $split_dir
mkdir -p $log_dir

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
    _cmd=(
        "${extract_python}" wespeaker/diar/extract_emb_w2vbert_wpt_zl389_adapter.py
        --scp "${scp_subfile}"
        --segments "${segments}"
        --ark-path "${write_ark}"
        --ckpt-dir "${ckpt_dir}"
        --checkpoint-name "${checkpoint_name}"
        --usm-ftcode "${usm_ftcode}"
        --device "${device}"
        --batch-size "${batch_size}"
        --frame-shift "${frame_shift}"
        --window-secs "${window_secs}"
        --period-secs "${period_secs}"
        --subseg-cmn "${subseg_cmn}"
    )
    if [ "$verbose" = true ] && [ "$nj" -eq 1 ]; then
        echo "$0: ${_cmd[*]} -> $logf" >&2
        "${_cmd[@]}" 2>&1 | tee "$logf" || exit 1
    else
        "${_cmd[@]}" > "$logf" 2>&1 &
        pids+=($!)
        logfs+=("$logf")
    fi
    idx=$((idx + 1))
done

if [ ${#pids[@]} -gt 0 ]; then
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "$0: extract_emb_w2vbert_wpt_zl389_adapter.py failed (pid ${pids[$i]}). Log: ${logfs[$i]}" >&2
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
    echo "$0: no emb_*.scp under ${store_dir}. Check ${log_dir}" >&2
    exit 1
fi
cat "${emb_scps[@]}" > "$store_dir/emb.scp"
emb_lines=$(wc -l < "$store_dir/emb.scp")
if [ "$emb_lines" -eq 0 ]; then
    echo "$0: emb.scp is empty after concat." >&2
    exit 1
fi
echo "Finish extract embedding (WPT+zl389 adapter, GPU mel recipe). (${emb_lines} lines in emb.scp)"

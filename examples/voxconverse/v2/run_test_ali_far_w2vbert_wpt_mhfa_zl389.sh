#!/bin/bash
# Speaker diarization on Test_Ali_far dataset
# WPT + w2v-BERT-2.0 + MHFA embeddings (USM_FTcode zl389 recipe)
#
# Stages match run_w2vbert_wpt_mhfa_zl389_notsofar_mtg_sc.sh:
#   1=SCTK  2=data prep (wav.scp + TextGrid→RTTM)  3=Demucs
#   4=VAD   5=fbank   6=WPT+MHFA embeddings
#   7=cluster  8=RTTM+overlap  9=DER
#
# Dataset layout (Test_Ali_far):
#   audio_dir/    R{room}_M{mic}_MS{ch}.wav   (20 files)
#   textgrid_dir/ R{room}_M{mic}.TextGrid     (20 files)
#
# File ID = R{room}_M{mic}  (last _MS{ch} component stripped).
# Reference RTTMs are written to ${ALI_FAR_REF_ROOT}/${partition}/ by stage 2.

. ./path.sh || exit 1

# Python: prefer repo .venv; override with PYTHON=/path/to/python
_ws_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"
if [ -z "${PYTHON}" ] && [ -x "${_ws_root}/.venv/bin/python" ]; then
    PYTHON="${_ws_root}/.venv/bin/python"
    export PATH="${_ws_root}/.venv/bin:${PATH}"
else
    PYTHON="${PYTHON:-python3}"
fi

# ── Dataset paths ─────────────────────────────────────────────────────
# Root of Test_Ali_far (must contain audio_dir/ and textgrid_dir/)
TEST_ALI_FAR_DIR="${TEST_ALI_FAR_DIR:-$HOME/DATASETS/Test_Ali/Test_Ali_far}"
# Where reference RTTMs land: ${ALI_FAR_REF_ROOT}/${partition}/<file_id>.rttm
ALI_FAR_REF_ROOT="${ALI_FAR_REF_ROOT:-data/ali_far_master}"

stage=2
stop_stage=9
partition="test_ali_far"
subseg_cmn=true
get_each_file_res=1
skip_prep_if_present=true

# ── VAD ────────────────────────────────────────────────────────────────
sad_type="funasr_fsmn"       # oracle / system / pyannote / funasr_fsmn
pyannote_device="${PYANNOTE_DEVICE:-cpu}"
pyannote_onset=0.5
pyannote_offset=0.5
pyannote_nj=16
funasr_hub="${FUNASR_HUB:-hf}"
funasr_revision="${FUNASR_REVISION:-v2.0.4}"
funasr_device="${FUNASR_DEVICE:-cpu}"
funasr_nj="${FUNASR_NJ:-4}"

# ── Demucs (optional) ─────────────────────────────────────────────────
use_demucs=false
demucs_device="${DEMUCS_DEVICE:-cuda}"
demucs_model="${DEMUCS_MODEL:-htdemucs}"

# ── WPT + MHFA + w2v-BERT ─────────────────────────────────────────────
_v2_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
USM_FTCODE="${USM_FTCODE:-$_v2_dir/wpt_mhfa_zl389}"
WPT_MHFA_CKPT_DIR="${WPT_MHFA_CKPT_DIR:-$HOME/Encode-explore/USM_FTcode/ckpt_asv/simple_sv_wpt_w2vbert_mhfa_zl389_h8_c128_fixed24000}"
WPT_MHFA_CHECKPOINT_NAME="${WPT_MHFA_CHECKPOINT_NAME:-best_model.pt}"
USM_WPT_MHFA_PYTHON="${USM_WPT_MHFA_PYTHON:-}"
emb_batch_size="${EMB_BATCH_SIZE:-32}"
emb_nj="${EMB_NJ:-1}"
emb_device="${W2VBERT_EMB_DEVICE:-cuda}"
frame_shift=10
window_secs=1.5
period_secs=0.75

# ── Clustering ────────────────────────────────────────────────────────
cluster_type="umap"   # spectral / umap / ahc / doverlap
merge_cutoff=0.2
ahc_threshold=0.21
ahc_linkage="average"
doverlap_weight_type="rank"
doverlap_custom_weights=""
doverlap_gaussian_std=0.5
doverlap_dover_weight=0.05

# ── Overlap ───────────────────────────────────────────────────────────
use_overlap=false
overlap_device="${OVERLAP_DEVICE:-cpu}"
overlap_min_dur=0.1
overlap_nj=16

# ── Misc ─────────────────────────────────────────────────────────────
verbose=true
bash_trace=false

# Suffix for all embedding artifact names
W2V="_w2vbert_wptmhfa_zl389_ali"

help_message="Usage: $0 [options]
Test_Ali_far — WPT+MHFA+w2v-BERT (USM_FTcode zl389 recipe).

Stages: 1=SCTK 2=data prep 3=Demucs 4=VAD 5=fbank 6=embeddings 7=cluster 8=RTTM 9=DER

  --sad_type oracle|system|pyannote|funasr_fsmn  (default: oracle)
  --cluster_type spectral|umap|ahc|doverlap      (default: umap)
  --use_demucs true|false
  --use_overlap true|false
  --verbose true|false
  --bash_trace true|false
  --skip_prep_if_present true|false  skip stages 1–2 if artifacts exist (default: true)

Env: TEST_ALI_FAR_DIR   root of Test_Ali_far dataset (audio_dir/ + textgrid_dir/)
     ALI_FAR_REF_ROOT   where reference RTTMs are written (default: data/ali_far_master)
     USM_FTCODE, WPT_MHFA_CKPT_DIR, WPT_MHFA_CHECKPOINT_NAME, USM_WPT_MHFA_PYTHON,
     W2VBERT_EMB_DEVICE, EMB_BATCH_SIZE, EMB_NJ,
     HF_TOKEN (PyAnnote), DEMUCS_DEVICE, PYANNOTE_DEVICE, OVERLAP_DEVICE,
     FUNASR_HUB, FUNASR_REVISION, FUNASR_DEVICE, FUNASR_NJ"

. tools/parse_options.sh

if [ "$bash_trace" = true ]; then set -x; fi

log_file="run_test_ali_far_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Full log: $log_file"
echo ""
echo "TEST_ALI_FAR_DIR=${TEST_ALI_FAR_DIR}"
echo "ALI_FAR_REF_ROOT=${ALI_FAR_REF_ROOT} (RTTMs: \${ALI_FAR_REF_ROOT}/${partition}/)"
echo "partition=${partition}  |  W2V suffix=${W2V}"
echo ""

SECONDS=0
declare -A stage_status

stage_banner() {
    echo ""
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
    echo ""
}

stage_done() { stage_status[$1]="$2"; }

print_summary() {
    echo ""
    echo "================================================================================"
    echo "  PIPELINE SUMMARY (Test_Ali_far, WPT+MHFA+w2v-BERT zl389)"
    echo "================================================================================"
    for snum in $(printf '%s\n' "${!stage_status[@]}" | sort -n); do
        echo "  Stage ${snum}: ${stage_status[$snum]}"
    done
    echo "--------------------------------------------------------------------------------"
    echo "  Elapsed: ${SECONDS}s | Log: $log_file"
    echo "================================================================================"
}

resolve_wav_scp() {
    if [ "$use_demucs" = true ]; then
        wav_scp="data/${partition}/wav_demucs.scp"
        if [ ! -s "$wav_scp" ]; then
            echo "$0: expected ${wav_scp} (run stage 3 with --use_demucs true first)." >&2
            exit 1
        fi
    else
        wav_scp="data/${partition}/wav.scp"
    fi
}

# ─── Stage 1: SCTK ───────────────────────────────────────────────────
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    stage_banner "Stage 1: SCTK (md-eval)"
    if [ "${skip_prep_if_present}" = true ] \
            && [ -f external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl ]; then
        echo "$0: Stage 1: SCTK already present."
        stage_done 1 "SCTK (existing)"
    else
        mkdir -p external_tools
        wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip \
             -O external_tools/SCTK-v2.4.12.zip
        "${PYTHON}" local/extract_zip.py external_tools/SCTK-v2.4.12.zip external_tools || exit 1
        stage_done 1 "SCTK ready"
    fi
fi

# ─── Stage 2: Data preparation (wav.scp + TextGrid → RTTM) ───────────
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    stage_banner "Stage 2: Test_Ali_far data prep (wav.scp + TextGrid → RTTM)"

    ref_partition_dir="${ALI_FAR_REF_ROOT}/${partition}"

    if [ "${skip_prep_if_present}" = true ] \
            && [ -s "data/${partition}/wav.scp" ] \
            && compgen -G "${ref_partition_dir}/*.rttm" > /dev/null; then
        n_utts=$(wc -l < "data/${partition}/wav.scp")
        echo "$0: Stage 2: wav.scp + refs already present (${n_utts} utts, skipping)."
        stage_done 2 "data prep (existing, ${n_utts} utts)"
    else
        if [ ! -d "${TEST_ALI_FAR_DIR}/audio_dir" ] || [ ! -d "${TEST_ALI_FAR_DIR}/textgrid_dir" ]; then
            echo "$0: TEST_ALI_FAR_DIR must contain audio_dir/ and textgrid_dir/: ${TEST_ALI_FAR_DIR}" >&2
            exit 1
        fi

        mkdir -p "data/${partition}" "${ref_partition_dir}"

        # Build wav.scp: file_id = first two _-separated parts of audio stem
        echo "Building wav.scp ..."
        {
            for wav in "${TEST_ALI_FAR_DIR}/audio_dir"/*.wav; do
                base=$(basename "$wav" .wav)
                # Strip trailing _MS<code> to get file_id matching TextGrid stem
                file_id=$(echo "$base" | sed 's/_[^_]*$//')
                echo "${file_id} ${wav}"
            done
        } | sort > "data/${partition}/wav.scp"

        n_utts=$(wc -l < "data/${partition}/wav.scp")
        if [ "${n_utts}" -eq 0 ]; then
            echo "$0: no WAV files found under ${TEST_ALI_FAR_DIR}/audio_dir/" >&2
            exit 1
        fi
        echo "wav.scp: ${n_utts} utterances"

        # Convert TextGrid → per-file RTTM
        echo "Converting TextGrid → RTTM ..."
        "${PYTHON}" local/textgrid_to_rttm.py \
                --textgrid-dir "${TEST_ALI_FAR_DIR}/textgrid_dir" \
                --out-rttm-dir "${ref_partition_dir}" \
                --channel 1 || exit 1

        n_rttm=$(ls "${ref_partition_dir}"/*.rttm 2>/dev/null | wc -l)
        echo "Reference RTTMs: ${n_rttm} files under ${ref_partition_dir}/"

        stage_done 2 "data prep done (${n_utts} utts, ${n_rttm} ref RTTMs)"
    fi
fi

# ─── Stage 3: Demucs ─────────────────────────────────────────────────
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$use_demucs" = true ]; then
        stage_banner "Stage 3: Demucs (${demucs_model})"
        demucs_vocals_dir="data/${partition}/demucs_vocals"
        rm -rf "${demucs_vocals_dir}" 2>/dev/null
        mkdir -p "${demucs_vocals_dir}"
        "${PYTHON}" wespeaker/diar/demucs_vocals.py \
                --scp "data/${partition}/wav.scp" \
                --out-dir "${demucs_vocals_dir}" \
                --wav-scp-out "data/${partition}/wav_demucs.scp" \
                --model "${demucs_model}" \
                --device "${demucs_device}" || exit 1
        stage_done 3 "Demucs -> data/${partition}/wav_demucs.scp"
    else
        echo "Stage 3: Demucs skipped (use_demucs=false)."
    fi
fi

# ─── Stage 4: VAD ────────────────────────────────────────────────────
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    resolve_wav_scp
    stage_banner "Stage 4: VAD (${sad_type})"
    min_duration=0.255

    if [[ "x${sad_type}" == "xoracle" ]]; then
        while read -r utt wav_path; do
            ref_rttm="${ALI_FAR_REF_ROOT}/${partition}/${utt}.rttm"
            if [ ! -f "${ref_rttm}" ]; then
                echo "$0: missing ref RTTM for ${utt}: ${ref_rttm}" >&2
                exit 1
            fi
            "${PYTHON}" wespeaker/diar/make_oracle_sad.py \
                    --rttm "${ref_rttm}" \
                    --min-duration ${min_duration}
        done < "${wav_scp}" > "data/${partition}/oracle_sad"
        sad_lines=$(wc -l < "data/${partition}/oracle_sad")
        echo "Oracle SAD: ${sad_lines} segments"
    fi

    if [[ "x${sad_type}" == "xsystem" ]]; then
        if [ "$verbose" = true ]; then
            "${PYTHON}" wespeaker/diar/make_system_sad.py \
                    --scp "${wav_scp}" \
                    --min-duration ${min_duration} | tee "data/${partition}/system_sad"
        else
            "${PYTHON}" wespeaker/diar/make_system_sad.py \
                    --scp "${wav_scp}" \
                    --min-duration ${min_duration} > "data/${partition}/system_sad" 2>/dev/null
        fi
        sad_lines=$(wc -l < "data/${partition}/system_sad")
        echo "System SAD: ${sad_lines} segments"
        [ "${sad_lines}" -eq 0 ] && { echo "$0: system_sad is empty." >&2; exit 1; }
    fi

    if [[ "x${sad_type}" == "xpyannote" ]]; then
        echo "PyAnnote VAD (device=${pyannote_device}, nj=${pyannote_nj}) ..."
        "${PYTHON}" wespeaker/diar/make_pyannote_sad.py \
                --scp "${wav_scp}" \
                --min-duration ${min_duration} \
                --onset ${pyannote_onset} \
                --offset ${pyannote_offset} \
                --device ${pyannote_device} \
                --nj ${pyannote_nj} > "data/${partition}/pyannote_sad" || exit 1
        sad_lines=$(wc -l < "data/${partition}/pyannote_sad")
        echo "PyAnnote SAD: ${sad_lines} segments"
        [ "${sad_lines}" -eq 0 ] && { echo "$0: pyannote_sad empty — check HF_TOKEN." >&2; exit 1; }
    fi

    if [[ "x${sad_type}" == "xfunasr_fsmn" ]]; then
        echo "FunASR FSMN-VAD (hub=${funasr_hub}, device=${funasr_device}, nj=${funasr_nj}) ..."
        "${PYTHON}" wespeaker/diar/make_funasr_fsmn_sad.py \
                --scp "${wav_scp}" \
                --min-duration ${min_duration} \
                --hub "${funasr_hub}" \
                --model-revision "${funasr_revision}" \
                --device "${funasr_device}" \
                --nj "${funasr_nj}" > "data/${partition}/funasr_fsmn_sad" || exit 1
        sad_lines=$(wc -l < "data/${partition}/funasr_fsmn_sad")
        echo "FunASR FSMN SAD: ${sad_lines} segments"
        [ "${sad_lines}" -eq 0 ] && { echo "$0: funasr_fsmn_sad empty." >&2; exit 1; }
    fi

    stage_done 4 "VAD (${sad_type}, ${sad_lines:-?} segments)"
fi

# ─── Stage 5: Fbank ───────────────────────────────────────────────────
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    resolve_wav_scp
    stage_banner "Stage 5: Fbank"
    [ -d "exp/${partition}_${sad_type}_sad_fbank" ] && rm -r "exp/${partition}_${sad_type}_sad_fbank"
    bash local/make_fbank.sh \
            --scp "${wav_scp}" \
            --segments "data/${partition}/${sad_type}_sad" \
            --store_dir "exp/${partition}_${sad_type}_sad_fbank" \
            --subseg_cmn ${subseg_cmn} \
            --verbose ${verbose} \
            --nj 24 || exit 1
    stage_done 5 "Fbank done"
fi

# ─── Stage 6: WPT + MHFA + w2v-BERT embeddings ───────────────────────
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    resolve_wav_scp
    stage_banner "Stage 6: WPT+MHFA+w2v-BERT embeddings (USM zl389)"
    [ ! -d "${USM_FTCODE}" ] && { echo "$0: USM_FTCODE not found: ${USM_FTCODE}" >&2; exit 1; }
    [ ! -f "${WPT_MHFA_CKPT_DIR}/args.json" ] && { echo "$0: missing args.json in WPT_MHFA_CKPT_DIR=${WPT_MHFA_CKPT_DIR}" >&2; exit 1; }
    [ ! -f "${WPT_MHFA_CKPT_DIR}/${WPT_MHFA_CHECKPOINT_NAME}" ] && { echo "$0: missing checkpoint ${WPT_MHFA_CKPT_DIR}/${WPT_MHFA_CHECKPOINT_NAME}" >&2; exit 1; }

    emb_root="exp/${partition}_${sad_type}_sad_embedding${W2V}"
    [ -d "${emb_root}" ] && rm -r "${emb_root}"

    bash local/extract_emb_w2vbert_wpt_mhfa_zl389.sh \
            --scp "${wav_scp}" \
            --segments "data/${partition}/${sad_type}_sad" \
            --ckpt-dir "${WPT_MHFA_CKPT_DIR}" \
            --checkpoint-name "${WPT_MHFA_CHECKPOINT_NAME}" \
            --usm-ftcode "${USM_FTCODE}" \
            --device ${emb_device} \
            --store_dir "${emb_root}" \
            --batch_size ${emb_batch_size} \
            --frame_shift ${frame_shift} \
            --window_secs ${window_secs} \
            --period_secs ${period_secs} \
            --subseg_cmn ${subseg_cmn} \
            --verbose ${verbose} \
            --nj ${emb_nj} || exit 1

    emb_scp="${emb_root}/emb.scp"
    [ ! -s "${emb_scp}" ] && { echo "$0: ${emb_scp} missing or empty." >&2; exit 1; }
    emb_lines=$(wc -l < "${emb_scp}")
    stage_done 6 "WPT+MHFA embeddings (${emb_lines} segments, ${emb_device})"
fi

# ─── Stage 7: Clustering ─────────────────────────────────────────────
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    emb_scp="exp/${partition}_${sad_type}_sad_embedding${W2V}/emb.scp"
    labels_suffix="${partition}_${sad_type}_sad${W2V}_labels"
    rttm_suffix="${partition}_${sad_type}_sad${W2V}_rttm"

    stage_banner "Stage 7: ${cluster_type} clustering"

    if [ "${cluster_type}" == "doverlap" ]; then
        echo "DOVER-Lap: umap + ahc + spectral -> fused RTTM"
        mkdir -p exp/umap_cluster exp/ahc_cluster exp/spectral_cluster exp/doverlap_cluster

        echo "  [1/3] UMAP+HDBSCAN (merge_cutoff=${merge_cutoff})"
        "${PYTHON}" wespeaker/diar/umap_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/umap_cluster/${labels_suffix} \
                --merge_cutoff ${merge_cutoff}
        "${PYTHON}" wespeaker/diar/make_rttm.py \
                --labels exp/umap_cluster/${labels_suffix} \
                --channel 1 > exp/umap_cluster/${rttm_suffix}

        echo "  [2/3] AHC (threshold=${ahc_threshold}, ${ahc_linkage})"
        "${PYTHON}" wespeaker/diar/ahc_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/ahc_cluster/${labels_suffix} \
                --threshold ${ahc_threshold} --linkage ${ahc_linkage}
        "${PYTHON}" wespeaker/diar/make_rttm.py \
                --labels exp/ahc_cluster/${labels_suffix} \
                --channel 1 > exp/ahc_cluster/${rttm_suffix}

        echo "  [3/3] Spectral"
        "${PYTHON}" wespeaker/diar/spectral_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/spectral_cluster/${labels_suffix}
        "${PYTHON}" wespeaker/diar/make_rttm.py \
                --labels exp/spectral_cluster/${labels_suffix} \
                --channel 1 > exp/spectral_cluster/${rttm_suffix}

        doverlap_extra_args=""
        if [ -n "${doverlap_custom_weights}" ] && [ "${doverlap_weight_type}" = "custom" ]; then
            doverlap_extra_args="--custom-weight ${doverlap_custom_weights}"
        fi
        echo "  Fusing (${doverlap_weight_type}, gstd=${doverlap_gaussian_std}, dw=${doverlap_dover_weight}) ..."
        dover-lap \
            exp/doverlap_cluster/${rttm_suffix} \
            exp/umap_cluster/${rttm_suffix} \
            exp/ahc_cluster/${rttm_suffix} \
            exp/spectral_cluster/${rttm_suffix} \
            --weight-type ${doverlap_weight_type} \
            --gaussian-filter-std ${doverlap_gaussian_std} \
            --dover-weight ${doverlap_dover_weight} \
            ${doverlap_extra_args} 2>&1
        stage_done 7 "DOVER-Lap fusion"
    else
        mkdir -p exp/${cluster_type}_cluster
        [ -f "exp/${cluster_type}_cluster/${labels_suffix}" ] && rm "exp/${cluster_type}_cluster/${labels_suffix}"

        cluster_extra_args=""
        if [ "${cluster_type}" == "umap" ]; then
            cluster_extra_args="--merge_cutoff ${merge_cutoff}"
        elif [ "${cluster_type}" == "ahc" ]; then
            cluster_extra_args="--threshold ${ahc_threshold} --linkage ${ahc_linkage}"
        fi
        "${PYTHON}" wespeaker/diar/${cluster_type}_clusterer.py \
                --scp "${emb_scp}" \
                --output exp/${cluster_type}_cluster/${labels_suffix} \
                ${cluster_extra_args}
        stage_done 7 "${cluster_type} clustering"
    fi
fi

# ─── Stage 8: RTTM ───────────────────────────────────────────────────
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && [ "$cluster_type" != "doverlap" ]; then
    stage_banner "Stage 8: labels -> RTTM"
    "${PYTHON}" wespeaker/diar/make_rttm.py \
            --labels exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_labels \
            --channel 1 > exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm
    stage_done 8 "RTTM generated"
fi

# ─── Stage 8b: Overlap detection ────────────────────────────────────
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && [ "$use_overlap" = true ]; then
    resolve_wav_scp
    emb_scp="exp/${partition}_${sad_type}_sad_embedding${W2V}/emb.scp"
    rttm_in="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm"
    rttm_ovl="${rttm_in}_overlap"
    stage_banner "Stage 8b: overlap detection"
    "${PYTHON}" wespeaker/diar/overlap_detection.py \
            --rttm "${rttm_in}" \
            --scp-wav "${wav_scp}" \
            --scp-emb "${emb_scp}" \
            --output "${rttm_ovl}" \
            --min-overlap-dur ${overlap_min_dur} \
            --device ${overlap_device} \
            --nj ${overlap_nj} \
            --channel 1
    cp "${rttm_ovl}" "${rttm_in}"
    stage_done 8 "RTTM + overlap"
fi

# ─── Stage 9: DER ────────────────────────────────────────────────────
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    stage_banner "Stage 9: DER (md-eval)"
    MD_EVAL=external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl
    if [ ! -f "${MD_EVAL}" ]; then
        echo "$0: SCTK not found: ${MD_EVAL} — run stage 1." >&2; exit 1
    fi

    ref_partition_dir="${ALI_FAR_REF_ROOT}/${partition}"
    sys_rttm="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_rttm"

    if [ ! -d "${ref_partition_dir}" ] || ! compgen -G "${ref_partition_dir}/*.rttm" > /dev/null; then
        echo "$0: no reference RTTMs under ${ref_partition_dir}/ — run stage 2." >&2; exit 1
    fi

    echo "Evaluating DER (Test_Ali_far, WPT+MHFA+w2v-BERT) ..."
    perl "${MD_EVAL}" \
         -c 0.25 \
         -r <(cat "${ref_partition_dir}"/*.rttm) \
         -s "${sys_rttm}" 2>&1 \
         | grep -v '^WARNING:' \
         | tee exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_res

    if [ ${get_each_file_res} -eq 1 ]; then
        single_file_res_dir=exp/${cluster_type}_cluster/${partition}_${sad_type}${W2V}_single_file_res
        mkdir -p "${single_file_res_dir}"
        echo "Per-file DER -> ${single_file_res_dir}"
        awk '{print $2}' "${sys_rttm}" | sort -u | while read -r file_name; do
            ref_rttm="${ref_partition_dir}/${file_name}.rttm"
            if [ -f "${ref_rttm}" ]; then
                perl "${MD_EVAL}" \
                     -c 0.25 \
                     -r <(cat "${ref_rttm}") \
                     -s <(grep " ${file_name} " "${sys_rttm}") 2>/dev/null \
                     > "${single_file_res_dir}/${partition}_${file_name}_res"
            fi
        done
    fi

    res_file="exp/${cluster_type}_cluster/${partition}_${sad_type}_sad${W2V}_res"
    der_line=$(grep 'OVERALL SPEAKER DIARIZATION ERROR' "$res_file" 2>/dev/null || true)
    der_pct=$(echo "$der_line" | grep -oP '[\d.]+(?= percent)' || echo "?")
    stage_done 9 "DER = ${der_pct}% (${partition}, ${sad_type}, ${cluster_type})"
fi

print_summary

#!/bin/bash
# Poly-sim evaluation on custom English/Urdu folders
# using WPT + w2v-BERT-2.0 + MHFA embeddings (USM_FTcode zl389 recipe).
#
# This script reuses the best-performing pipeline from
# run_w2vbert_wpt_mhfa_zl389.sh but:
#   - Runs only stages 4–8 (VAD → fbank → embeddings → clustering → RTTM)
#   - Does NOT compute DER (no reference RTTMs for this data)
#   - Splits the final RTTM into per-recording, per-speaker RTTMs
#     inside each language folder:
#         <lang_dir>/<out_subdir>/<orig>.rttm          (1 speaker)
#         <lang_dir>/<out_subdir>/<orig>_1.rttm, ...   (>1 speakers)
#
# Datasets (defaults from the user paths):
#   ENGLISH_AUDIO_DIR = poly-sim/v1_val_English_disagree/audio_wrong/voices/English
#   URDU_AUDIO_DIR    = poly-sim/v1_val_Urdu_disagree/audio_wrong/voices/Urdu

. ./path.sh || exit 1

_v2_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_ws_root="$(cd "${_v2_dir}/../../.." && pwd)"

# Prefer repo .venv (FunASR, torch); override with PYTHON=/path/to/python.
if [ -z "${PYTHON}" ] && [ -x "${_ws_root}/.venv/bin/python" ]; then
    PYTHON="${_ws_root}/.venv/bin/python"
    export PATH="${_ws_root}/.venv/bin:${PATH}"
else
    PYTHON="${PYTHON:-python3}"
fi

# --- Input folders (override via env if needed) ----------------------------
ENGLISH_AUDIO_DIR="${ENGLISH_AUDIO_DIR:-${_v2_dir}/poly-sim/v1_val_English_disagree/audio_wrong/voices/English}"
URDU_AUDIO_DIR="${URDU_AUDIO_DIR:-${_v2_dir}/poly-sim/v1_val_Urdu_disagree/audio_wrong/voices/Urdu}"

OUT_SUBDIR="${OUT_SUBDIR:-diar_rttm}"  # created under each language dir

if [ ! -d "${ENGLISH_AUDIO_DIR}" ]; then
    echo "$0: ENGLISH_AUDIO_DIR not found: ${ENGLISH_AUDIO_DIR}" >&2
    exit 1
fi
if [ ! -d "${URDU_AUDIO_DIR}" ]; then
    echo "$0: URDU_AUDIO_DIR not found: ${URDU_AUDIO_DIR}" >&2
    exit 1
fi

# --- Shared diarization settings (mirrors run_w2vbert_wpt_mhfa_zl389.sh) ---
subseg_cmn=true
sad_type="funasr_fsmn"       # oracle / system (Silero) / pyannote / funasr_fsmn
pyannote_device="${PYANNOTE_DEVICE:-cpu}"
pyannote_onset=0.5
pyannote_offset=0.5
pyannote_nj=16
funasr_hub="${FUNASR_HUB:-hf}"
funasr_revision="${FUNASR_REVISION:-v2.0.4}"
funasr_device="${FUNASR_DEVICE:-cpu}"
funasr_nj="${FUNASR_NJ:-4}"

use_demucs=false
demucs_device="${DEMUCS_DEVICE:-cuda}"
demucs_model="${DEMUCS_MODEL:-htdemucs}"

USM_FTCODE="${USM_FTCODE:-${_v2_dir}/wpt_mhfa_zl389}"
WPT_MHFA_CKPT_DIR="${WPT_MHFA_CKPT_DIR:-$HOME/Encode-explore/USM_FTcode/ckpt_asv/simple_sv_wpt_w2vbert_mhfa_zl389_h8_c128_fixed24000}"
WPT_MHFA_CHECKPOINT_NAME="${WPT_MHFA_CHECKPOINT_NAME:-best_model.pt}"
USM_WPT_MHFA_PYTHON="${USM_WPT_MHFA_PYTHON:-}"
emb_batch_size="${EMB_BATCH_SIZE:-32}"
emb_nj="${EMB_NJ:-1}"
emb_device="${W2VBERT_EMB_DEVICE:-cuda}"
frame_shift=10
window_secs=1.5
period_secs=0.75

cluster_type="umap"   # spectral / umap / ahc / doverlap
merge_cutoff=0.2
ahc_threshold=0.21
ahc_linkage="average"
doverlap_weight_type="rank"
doverlap_custom_weights=""
doverlap_gaussian_std=0.5
doverlap_dover_weight=0.05

use_overlap=false
overlap_device="${OVERLAP_DEVICE:-cpu}"
overlap_min_dur=0.1
overlap_nj=16

verbose=true
bash_trace=false

W2V="_w2vbert_wptmhfa_zl389_poly"

. tools/parse_options.sh

if [ "$bash_trace" = true ]; then
    set -x
fi

log_file="run_poly_sim_eval_w2vbert_wpt_mhfa_zl389_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Full log: $log_file"
echo ""

SECONDS=0

stage_banner() {
    echo ""
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
    echo ""
}

resolve_wav_scp() {
    local partition="$1"
    local use_demucs_local="$2"
    local wav_scp_var
    if [ "$use_demucs_local" = true ]; then
        wav_scp_var="data/${partition}/wav_demucs.scp"
        if [ ! -s "$wav_scp_var" ]; then
            echo "$0: expected ${wav_scp_var} (Demucs stage not run?)." >&2
            exit 1
        fi
    else
        wav_scp_var="data/${partition}/wav.scp"
    fi
    echo "$wav_scp_var"
}

run_partition() {
    local lang_tag="$1"     # "English" or "Urdu"
    local audio_dir="$2"    # path to wavs
    local partition="poly_${lang_tag}"

    echo ""
    echo "##########################"
    echo "  Partition: ${partition}"
    echo "  Audio dir: ${audio_dir}"
    echo "##########################"
    echo ""

    mkdir -p "data/${partition}"

    # --- Build wav.scp from the audio folder --------------------------------
    stage_banner "(${partition}) Build wav.scp from ${audio_dir}"
    {
        for wav in "${audio_dir}"/*.wav; do
            [ -e "$wav" ] || continue
            base="$(basename "$wav" .wav)"
            # Use base name as utt ID; this matches the user's "filename" notion.
            echo "${base} ${wav}"
        done
    } | sort > "data/${partition}/wav.scp"

    n_utts=$(wc -l < "data/${partition}/wav.scp")
    if [ "${n_utts}" -eq 0 ]; then
        echo "$0: no WAV files found under ${audio_dir}" >&2
        return
    fi
    echo "wav.scp (${partition}): ${n_utts} utterances"

    # --- Stage 3: Demucs (optional, usually off) ----------------------------
    if [ "$use_demucs" = true ]; then
        stage_banner "(${partition}) Stage 3: Demucs (${demucs_model})"
        demucs_vocals_dir="data/${partition}/demucs_vocals"
        rm -rf "${demucs_vocals_dir}" 2>/dev/null
        mkdir -p "${demucs_vocals_dir}"
        "${PYTHON}" wespeaker/diar/demucs_vocals.py \
                --scp "data/${partition}/wav.scp" \
                --out-dir "${demucs_vocals_dir}" \
                --wav-scp-out "data/${partition}/wav_demucs.scp" \
                --model "${demucs_model}" \
                --device "${demucs_device}" || exit 1
    else
        echo "(${partition}) Stage 3: Demucs skipped (use_demucs=false)."
    fi

    # --- Stage 4: VAD -------------------------------------------------------
    stage_banner "(${partition}) Stage 4: VAD (${sad_type})"
    wav_scp=$(resolve_wav_scp "${partition}" "${use_demucs}")
    min_duration=0.255

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

    # --- Stage 5: Fbank -----------------------------------------------------
    stage_banner "(${partition}) Stage 5: Fbank"
    wav_scp=$(resolve_wav_scp "${partition}" "${use_demucs}")
    [ -d "exp/${partition}_${sad_type}_sad_fbank" ] && rm -r "exp/${partition}_${sad_type}_sad_fbank"
    bash local/make_fbank.sh \
            --scp "${wav_scp}" \
            --segments "data/${partition}/${sad_type}_sad" \
            --store_dir "exp/${partition}_${sad_type}_sad_fbank" \
            --subseg_cmn ${subseg_cmn} \
            --verbose ${verbose} \
            --nj 24 || exit 1

    # --- Stage 6: WPT+MHFA+w2v-BERT embeddings ------------------------------
    stage_banner "(${partition}) Stage 6: WPT+MHFA+w2v-BERT embeddings"
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
    echo "Embeddings: ${emb_lines} segments"

    # --- Stage 7: clustering -------------------------------------------------
    stage_banner "(${partition}) Stage 7: ${cluster_type} clustering"
    labels_suffix="${partition}_${sad_type}_sad${W2V}_labels"
    rttm_suffix="${partition}_${sad_type}_sad${W2V}_rttm"

    if [ "${cluster_type}" = "doverlap" ]; then
        echo "DOVER-Lap not wired in this evaluation script (use cluster_type!=doverlap)."
        exit 1
    fi

    mkdir -p "exp/${cluster_type}_cluster"
    [ -f "exp/${cluster_type}_cluster/${labels_suffix}" ] && rm "exp/${cluster_type}_cluster/${labels_suffix}"

    cluster_extra_args=""
    if [ "${cluster_type}" = "umap" ]; then
        cluster_extra_args="--merge_cutoff ${merge_cutoff}"
    elif [ "${cluster_type}" = "ahc" ]; then
        cluster_extra_args="--threshold ${ahc_threshold} --linkage ${ahc_linkage}"
    fi

    "${PYTHON}" wespeaker/diar/${cluster_type}_clusterer.py \
            --scp "${emb_scp}" \
            --output "exp/${cluster_type}_cluster/${labels_suffix}" \
            ${cluster_extra_args}

    # --- Stage 8: RTTM + (optional) overlap ---------------------------------
    stage_banner "(${partition}) Stage 8: labels -> RTTM"
    "${PYTHON}" wespeaker/diar/make_rttm.py \
            --labels "exp/${cluster_type}_cluster/${labels_suffix}" \
            --channel 1 > "exp/${cluster_type}_cluster/${rttm_suffix}"

    if [ "$use_overlap" = true ]; then
        stage_banner "(${partition}) Stage 8b: overlap detection"
        wav_scp=$(resolve_wav_scp "${partition}" "${use_demucs}")
        rttm_in="exp/${cluster_type}_cluster/${rttm_suffix}"
        rttm_ovl="${rttm_in}_overlap"
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
    fi

    # --- Post-processing: split RTTM + WAV per recording & speaker ----------
    stage_banner "(${partition}) Split RTTM into per-file, per-speaker RTTMs"
    lang_out_dir="${audio_dir}/${OUT_SUBDIR}"
    mkdir -p "${lang_out_dir}"
    "${PYTHON}" local/split_rttm_by_speaker.py \
            --rttm "exp/${cluster_type}_cluster/${rttm_suffix}" \
            --out-dir "${lang_out_dir}"

    stage_banner "(${partition}) Create per-speaker WAVs with same naming rule"
    wav_scp_for_cut=$(resolve_wav_scp "${partition}" "${use_demucs}")
    "${PYTHON}" local/split_wav_by_speaker_from_rttm.py \
            --rttm "exp/${cluster_type}_cluster/${rttm_suffix}" \
            --wav-scp "${wav_scp_for_cut}" \
            --out-dir "${lang_out_dir}"

    echo "(${partition}) Per-file RTTMs and WAVs -> ${lang_out_dir}"
}

run_partition "English" "${ENGLISH_AUDIO_DIR}"
run_partition "Urdu" "${URDU_AUDIO_DIR}"

echo ""
echo "================================================================================"
echo "Poly-sim evaluation finished in ${SECONDS}s"
echo "RTTM outputs are under:"
echo "  English: ${ENGLISH_AUDIO_DIR}/${OUT_SUBDIR}"
echo "  Urdu   : ${URDU_AUDIO_DIR}/${OUT_SUBDIR}"
echo "================================================================================"


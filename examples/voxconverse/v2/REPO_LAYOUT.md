# What to push for VoxConverse v2 (GitHub)

Target: [areffarhadi/wespeaker — `examples/voxconverse/v2`](https://github.com/areffarhadi/wespeaker/tree/master/examples/voxconverse/v2)

This folder is a **WeSpeaker recipe**, not a standalone repo. Pushing **only** `v2/` is not enough to run the new encoder pipelines.

## Push these paths (same branch on `areffarhadi/wespeaker`)

| Path | Why |
|------|-----|
| `examples/voxconverse/v2/` | Recipe scripts, `local/`, `wpt_mhfa_zl389/`, README, `requirements.txt` |
| `wespeaker/diar/*.py` | Diarization Python (w2v-BERT, WPT+MHFA, adapter, VAD, clustering) |
| `wespeaker/utils/` (if changed) | Shared helpers used by `wespeaker/diar/` |
| `tools/parse_options.sh` etc. | Linked as `v2/tools` → `../../../tools` |

Symlinks inside `v2/`:

- `wespeaker` → `../../../wespeaker`
- `tools` → `../../../tools`

They resolve only when the **full wespeaker repo** is cloned and you run from `examples/voxconverse/v2/`.

## Do not push

| Path | Reason |
|------|--------|
| `v2/exp/` | Pipeline outputs (~fbank, embeddings, RTTMs) |
| `v2/data/` | VoxConverse WAVs/zips (re-download in stage 2) |
| `v2/pretrained_models/` | ResNet ONNX (wget in stage 1) |
| `v2/external_tools/` | SCTK (wget in stage 1) |
| `v2/run_*.log` | Session logs |
| `v2/poly-sim/` | Local polyglot eval RTTMs/audio refs |

## Encoder entry points (this fork)

| Encoder | Run script | Extraction module (parent package) |
|---------|------------|--------------------------------------|
| ResNet34 ONNX | `run_org.sh`, `run_updated.sh` | `wespeaker/diar/extract_emb.py` |
| w2v-BERT SV | `run_w2vbert.sh` | `wespeaker/diar/extract_emb_w2vbert.py` |
| WPT + MHFA + w2v-BERT | `run_w2vbert_wpt_mhfa_zl389.sh` | `wespeaker/diar/extract_emb_w2vbert_wpt_mhfa_zl389.py` |
| zl389 adapter + GPU mel | `run_w2vbert_wpt_zl389_adapter_gpu_mel.sh` | `wespeaker/diar/extract_emb_w2vbert_wpt_zl389_adapter.py` |

Model **weights** are not in git (see README sections for `HF_MODELS`, `WPT_MHFA_CKPT_DIR`).

## Run after clone

```bash
cd wespeaker
pip install -e .
pip install -r examples/voxconverse/v2/requirements.txt
# + pytorch-wavelets, funasr if not already installed

cd examples/voxconverse/v2
./run_w2vbert_wpt_mhfa_zl389.sh --help
```

Download weights and data per script stages / env vars in each `run_*.sh` header.

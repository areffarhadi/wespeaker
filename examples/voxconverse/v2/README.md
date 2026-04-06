## Overview

* We suggest to run this recipe on a gpu-available machine, with onnxruntime-gpu supported.
* Dataset: Voxconverse2020 (dev: 216 utts, test: 232 utts)
* Speaker model: ResNet34 model pretrained by WeSpeaker
  * Refer to [voxceleb sv recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)
  * [pretrained model path](https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx)
* Speaker activity detection model:
  * oracle SAD (from ground truth annotation)
  * system SAD (VAD model pretrained by [silero-vad](https://github.com/snakers4/silero-vad), v3.1 => v5.1)
* Clustering method:
  * spectral clustering
  * umap dimensionality reduction + hdbscan clustering
* Metric: DER = MISS + FALSE ALARM + SPEAKER CONFUSION (%)

## Results

* Dev set

    | system | MISS | FA | SC | DER |
    |:---|:---:|:---:|:---:|:---:|
    | Ours (oracle SAD + spectral clustering) | 2.3 | 0.0 | 2.1 | 4.4 |
    | Ours (oracle SAD + umap clustering) | 2.3 | 0.0 | 1.3 | 3.6 |
    | Ours (silero-vad v3.1 + spectral clustering) | 3.7 | 0.8 | 2.2 | 6.7 |
    | Ours (silero-vad v5.1 + spectral clustering) | 3.4 | 0.6 | 2.3 | 6.3 |
    | Ours (silero-vad v5.1 + umap clustering) | 3.4 | 0.6 | 1.4 | 5.4 |
    | DIHARD 2019 baseline [^1] | 11.1 | 1.4 | 11.3 | 23.8 |
    | DIHARD 2019 baseline w/ SE [^1] | 9.3 | 1.3 | 9.7 | 20.2 |
    | (SyncNet ASD only) [^1] | 2.2 | 4.1 | 4.0 | 10.4 |
    | (AVSE ASD only) [^1] | 2.0 | 5.9 | 4.6 | 12.4 |
    | (proposed) [^1] | 2.4 | 2.3 | 3.0 | 7.7 |

* Test set

    | system | MISS | FA | SC | DER |
    |:---|:---:|:---:|:---:|:---:|
    | Ours (oracle SAD + spectral clustering) | 1.6 | 0.0 | 3.3 | 4.9 |
    | Ours (oracle SAD + umap clustering) | 1.6 | 0.0 | 1.9 | 3.5 |
    | Ours (silero-vad v3.1 + spectral clustering) | 4.0 | 2.4 | 3.4 | 9.8 |
    | Ours (silero-vad v5.1 + spectral clustering) | 3.8 | 1.7 | 3.3 | 8.8 |
    | Ours (silero-vad v5.1 + umap clustering) | 3.8 | 1.7 | 1.8 | 7.3 |


[^1]: Spot the conversation: speaker diarisation in the wild, https://arxiv.org/pdf/2007.01216.pdf

## My evaluations

The figures below use the same pipeline configured in [`run_updated.sh`](run_updated.sh) (PyAnnote VAD, optional Demucs / overlap handling, clustering, and DER). See that script for defaults. Switch partition with `--partition dev` or `--partition test`.

```bash
./run_updated.sh --partition dev
./run_updated.sh --partition test
```

(Run from this `v2` recipe directory.)

By default `run_updated.sh` sets `stage=9` / `stop_stage=9` (DER evaluation only). Use `--stage 1 --stop_stage 9` (or adjust as needed) to run the full recipe from scratch.

### Dev (`--partition dev`)

Excerpt from `md-eval` (**speaker-type confusion matrices**) for the **dev** partition:

#### Speaker weighted (counts)

| REF \ SYS | unknown | MISS |
|:---|:---:|:---:|
| unknown | 876 / 90.1% | 96 / 9.9% |
| FALSE ALARM | 29 / 3.0% | — |

#### Time weighted (seconds)

| REF \ SYS | unknown | MISS |
|:---|:---:|:---:|
| unknown | 63505.67 / 98.4% | 1019.67 / 1.6% |
| FALSE ALARM | 426.21 / 0.7% | — |

Raw log fragment:

```
 Speaker type confusion matrix -- speaker weighted
  REF\SYS (count)      unknown               MISS
unknown                 876 /  90.1%         96 /   9.9%
  FALSE ALARM            29 /   3.0%
---------------------------------------------
 Speaker type confusion matrix -- time weighted
  REF\SYS (seconds)    unknown               MISS
unknown            63505.67 /  98.4%    1019.67 /   1.6%
  FALSE ALARM        426.21 /   0.7%
```

### Test (`--partition test`)

**OVERALL SPEAKER DIARIZATION ERROR = 6.64%** (of scored speaker time, `md-eval`).

#### Speaker weighted (counts)

| REF \ SYS | unknown | MISS |
|:---|:---:|:---:|
| unknown | 1279 / 85.1% | 224 / 14.9% |
| FALSE ALARM | 220 / 14.6% | — |

#### Time weighted (seconds)

| REF \ SYS | unknown | MISS |
|:---|:---:|:---:|
| unknown | 127817.75 / 97.6% | 3136.57 / 2.4% |
| FALSE ALARM | 2357.95 / 1.8% | — |

Raw log fragment:

```
 OVERALL SPEAKER DIARIZATION ERROR = 6.64 percent of scored speaker time  `(ALL)
---------------------------------------------
 Speaker type confusion matrix -- speaker weighted
  REF\SYS (count)      unknown               MISS
unknown                1279 /  85.1%        224 /  14.9%
  FALSE ALARM           220 /  14.6%
---------------------------------------------
 Speaker type confusion matrix -- time weighted
  REF\SYS (seconds)    unknown               MISS
unknown           127817.75 /  97.6%    3136.57 /   2.4%
  FALSE ALARM       2357.95 /   1.8%
---------------------------------------------
```

### w2v-BERT (`run_w2vbert.sh`)

Results below use [`run_w2vbert.sh`](run_w2vbert.sh) with this base configuration (see variables at the top of the script):

* `sad_type="pyannote"`
* `use_demucs=false`
* `cluster_type="doverlap"` (DOVER-Lap over UMAP + AHC + spectral)

DOVER-Lap / UMAP / AHC hyperparameters match `run_w2vbert.sh`. The only deliberate difference between the two tables is **`use_overlap`**: overlap refinement after clustering (stage 8) is **on** or **off**.

Percentages are **of scored speaker time**, from SCTK `md-eval`. **DER = MISS + FA + SE**, with **SE** = speaker error (`SPEAKER ERROR TIME` in the tool; same component as **SC** in many papers).

#### `use_overlap=true` (overlap refinement after clustering)

| Partition | MISS (%) | FA (%) | SE (%) | DER (%) |
|:---|:---:|:---:|:---:|:---:|
| dev | 1.6 | 0.7 | 3.2 | 5.49 |
| test | 2.4 | 1.8 | 2.8 | 7.00 |

##### Dev (`--partition dev`)

`md-eval` excerpt:

```
SCORED SPEAKER TIME =  64525.34 secs (102.4 percent of scored speech)
MISSED SPEAKER TIME =   1036.90 secs (  1.6 percent of scored speaker time)
FALARM SPEAKER TIME =    422.97 secs (  0.7 percent of scored speaker time)
 SPEAKER ERROR TIME =   2081.23 secs (  3.2 percent of scored speaker time)
---------------------------------------------
 OVERALL SPEAKER DIARIZATION ERROR = 5.49 percent of scored speaker time  `(ALL)
---------------------------------------------
 Speaker type confusion matrix -- speaker weighted
  REF\SYS (count)      unknown               MISS
unknown                 872 /  89.7%        100 /  10.3%
  FALSE ALARM            46 /   4.7%
---------------------------------------------
 Speaker type confusion matrix -- time weighted
  REF\SYS (seconds)    unknown               MISS
unknown            63488.44 /  98.4%    1036.90 /   1.6%
  FALSE ALARM        422.97 /   0.7%
---------------------------------------------
```

##### Test (`--partition test`)

`md-eval` excerpt:

```
SCORED SPEAKER TIME = 130954.32 secs (101.6 percent of scored speech)
MISSED SPEAKER TIME =   3104.96 secs (  2.4 percent of scored speaker time)
FALARM SPEAKER TIME =   2373.34 secs (  1.8 percent of scored speaker time)
 SPEAKER ERROR TIME =   3685.58 secs (  2.8 percent of scored speaker time)
---------------------------------------------
 OVERALL SPEAKER DIARIZATION ERROR = 7.00 percent of scored speaker time  `(ALL)
---------------------------------------------
 Speaker type confusion matrix -- speaker weighted
  REF\SYS (count)      unknown               MISS
unknown                1278 /  85.0%        225 /  15.0%
  FALSE ALARM           126 /   8.4%
---------------------------------------------
 Speaker type confusion matrix -- time weighted
  REF\SYS (seconds)    unknown               MISS
unknown           127849.36 /  97.6%    3104.96 /   2.4%
  FALSE ALARM       2373.34 /   1.8%
---------------------------------------------
```

#### `use_overlap=false` (no overlap stage)

Re-run with e.g. `./run_w2vbert.sh --use_overlap false --partition dev --stage 9 --stop_stage 9` (after prior stages produced embeddings and cluster RTTMs).

| Partition | MISS (%) | FA (%) | SE (%) | DER (%) |
|:---|:---:|:---:|:---:|:---:|
| dev | 3.1 | 0.3 | 2.5 | 5.88 |
| test | 3.4 | 1.1 | 2.4 | 6.80 |

##### Dev (`--partition dev`)

`md-eval` excerpt:

```
SCORED SPEAKER TIME =  64525.34 secs (102.4 percent of scored speech)
MISSED SPEAKER TIME =   1984.81 secs (  3.1 percent of scored speaker time)
FALARM SPEAKER TIME =    196.71 secs (  0.3 percent of scored speaker time)
 SPEAKER ERROR TIME =   1611.05 secs (  2.5 percent of scored speaker time)
---------------------------------------------
 OVERALL SPEAKER DIARIZATION ERROR = 5.88 percent of scored speaker time  `(ALL)
---------------------------------------------
 Speaker type confusion matrix -- speaker weighted
  REF\SYS (count)      unknown               MISS
unknown                 870 /  89.5%        102 /  10.5%
  FALSE ALARM            48 /   4.9%
---------------------------------------------
 Speaker type confusion matrix -- time weighted
  REF\SYS (seconds)    unknown               MISS
unknown            62540.53 /  96.9%    1984.81 /   3.1%
  FALSE ALARM        196.71 /   0.3%
---------------------------------------------
```

##### Test (`--partition test`)

`md-eval` excerpt:

```
SCORED SPEAKER TIME = 130954.32 secs (101.6 percent of scored speech)
MISSED SPEAKER TIME =   4396.34 secs (  3.4 percent of scored speaker time)
FALARM SPEAKER TIME =   1398.96 secs (  1.1 percent of scored speaker time)
 SPEAKER ERROR TIME =   3111.79 secs (  2.4 percent of scored speaker time)
---------------------------------------------
 OVERALL SPEAKER DIARIZATION ERROR = 6.80 percent of scored speaker time  `(ALL)
---------------------------------------------
 Speaker type confusion matrix -- speaker weighted
  REF\SYS (count)      unknown               MISS
unknown                1275 /  84.8%        228 /  15.2%
  FALSE ALARM           131 /   8.7%
---------------------------------------------
 Speaker type confusion matrix -- time weighted
  REF\SYS (seconds)    unknown               MISS
unknown           126557.98 /  96.6%    4396.34 /   3.4%
  FALSE ALARM       1398.96 /   1.1%
---------------------------------------------
```

### Running `run_w2vbert.sh` (w2v-BERT diarization)

The script [`run_w2vbert.sh`](run_w2vbert.sh) runs the **VoxConverse v2** recipe with **w2v-BERT-2.0 speaker-verification embeddings** (PyTorch checkpoint) instead of the default ResNet34 ONNX pipeline (`run.sh`). From this directory:

```bash
./run_w2vbert.sh --help
./run_w2vbert.sh --partition dev --stage 1 --stop_stage 9
```

Useful overrides (see `help_message` in the script for the full list):

* **Stages:** `1`=SCTK, `2`=download data + `wav.scp`, `3`=optional Demucs, `4`=VAD, `5`=fbank, `6`=w2v-BERT embeddings, `7`=clustering, `8`=RTTM (and optional overlap), `9`=DER. Defaults (`stage` / `stop_stage` in the script) are set to start after prerequisites; adjust `--stage` / `--stop_stage` if you need to download tools/data or run Demucs.
* **Partition:** `--partition dev` or `--partition test`.
* **VAD:** `--sad_type oracle` | `system` (Silero) | `pyannote` (PyAnnote; may require HF token / model setup).
* **Demucs:** `--use_demucs true` (requires `pip install demucs`; run stage 3 before VAD).
* **Clustering:** `--cluster_type spectral` | `umap` | `ahc` | `doverlap` (DOVER-Lap fusion of UMAP + AHC + spectral RTTMs).
* **Overlap:** `--use_overlap true|false` (PyAnnote-based overlap pass on the system RTTM).
* **w2v-BERT paths:** set `W2VBERT_REPO`, `W2VBERT_CHECKPOINT`, or `HF_MODELS` so the checkpoint and fairseq-style repo resolve (defaults are in the script header).

Logs are written to a timestamped `run_w2vbert_*.log` in the same directory.

**What this recipe implements (high level):**

* **w2v-BERT SV embeddings** — Segment-level embeddings from a **w2v-BERT-2.0 SV** checkpoint for diarization, with outputs tagged `*_w2vbert` so they do not overwrite ResNet runs.
* **Optional Demucs** — **Vocal separation** (`htdemucs` by default) before VAD to reduce music/noise; writes `wav_demucs.scp` when enabled.
* **Flexible SAD** — **Oracle** (RTTM-derived), **Silero** VAD, or **PyAnnote** segmentation before fbank and embedding extraction.
* **Fbank + sliding windows** — Standard fbank features for the pipeline, then **w2v-BERT** extraction with configurable window/stride (see `extract_emb_w2vbert.sh` / script defaults).
* **Clustering options** — **Spectral**, **UMAP+HDBSCAN**, **AHC**, or **DOVER-Lap** to fuse three base clusterers into one RTTM.
* **Optional overlap handling** — Post-processing **overlap detection** on the system RTTM using embeddings and audio (PyAnnote-based tool in-repo).
* **Evaluation** — **SCTK `md-eval`** DER and per-file scores, same protocol as the rest of the recipe.

This path is independent of the ResNet `run_updated.sh` / `run.sh` embedding step but shares the same VoxConverse data layout and reference RTTMs under `data/voxconverse-master/`.

# TidyVoice2026 Challenge Baseline

## About the Challenge

This repository contains the baseline system for the **TidyVoice Challenge: Cross-Lingual Speaker Verification** at Interspeech 2026. The challenge addresses the critical problem of speaker verification under language mismatch, where system performance degrades significantly when speakers use different languages.

The challenge leverages the **TidyVoiceX dataset**, a large-scale, multilingual corpus derived from Mozilla Common Voice, specifically curated to isolate the effect of language switching across approximately 40 languages. The dataset features:

- Over 4,474 speakers across 40 languages
- Approximately 321,711 utterances totaling 457 hours
- Clearly defined training and development splits
- Pseudonymized speaker identities for privacy protection

**Challenge Website**: [https://tidyvoice2026.github.io](https://tidyvoice2026.github.io)

## Baseline System

This baseline system uses a **SimAM-ResNet34** architecture that is:
1. **Pretrained** on VoxBlink2 and VoxCeleb2 datasets
2. **Fine-tuned** on the TidyVoiceX training set using large-margin training

### Baseline Results

The baseline achieves the following performance on the TidyVoice development set:

| Architecture | Pretraining Data | Fine-tuning Data | EER (%) | MinDCF |
|:-------------|:----------------|:----------------|:-------:|:------:|
| SimAM-ResNet34 | VoxBlink2 + VoxCeleb2 | TidyVoiceX Train | 3.07 | 0.82 |

These results demonstrate the baseline's capability to handle cross-lingual speaker verification tasks.

---

## Prerequisites

### 1. API Key Setup (Required)

**IMPORTANT**: Before running the code, you must obtain and configure your Mozilla Common Voice API key to download the TidyVoiceX dataset.

1. Get your API key from: [https://datacollective.mozillafoundation.org/api-reference](https://datacollective.mozillafoundation.org/api-reference)

2. Edit `run_vox_custom_paak.sh` and set your API key:

```bash
# CommonVoice DataCollective API Key
# Get your API key from: https://datacollective.mozillafoundation.org/api-reference
TIDYVOICE_API_KEY="Enter your Mozilla CommonVoice API key here"
```

Replace `"Enter your Mozilla CommonVoice API key here"` with your actual API key.

### 2. Installation

#### Install WeSpeaker

Follow the main WeSpeaker installation instructions. Please refer to the [WeSpeaker documentation](https://github.com/wenet-e2e/wespeaker) for detailed installation steps.

Generally, the installation includes:
- Python 3.9+
- PyTorch (with CUDA support)
- CUDA-capable GPU
- Required Python packages

#### Install DataCollective

For downloading the TidyVoiceX dataset, you need to install the `datacollective` package:

```bash
pip install datacollective
```

This package is required for automatic dataset download via the DataCollective API.

---

## Running the Baseline

The main script is `run_vox_custom_paak.sh`, which performs the complete pipeline from data preparation to evaluation.

### Pipeline Stages

The script includes the following stages:

1. **Stage 1**: Download and prepare datasets (TidyVoiceX, MUSAN, RIRS_NOISES)
2. **Stage 2**: Convert data to shard/raw format for training
3. **Stage 3**: Train the model (if training from scratch)
4. **Stage 4**: Model averaging and embedding extraction
5. **Stage 5**: Score evaluation dataset
6. **Stage 6**: Score normalization (AS-Norm/S-Norm)
7. **Stage 7**: Score calibration
8. **Stage 8**: Export the best model
9. **Stage 9**: Large-margin fine-tuning on TidyVoiceX training set
10. **Stage 10**: Extract embeddings to numpy format
11. **Stage 11**: Score from numpy embeddings
12. **Stage 12**: Extract embeddings from a directory with WAV files

### Quick Start

#### 1. Configure the Script

Edit `run_vox_custom_paak.sh` and set:
- `TIDYVOICE_API_KEY`: Your API key (see above)
- `stage` and `stop_stage`: Which stages to run
- `gpus`: GPU IDs to use (e.g., `"[0]"` for single GPU)
- `eval_dataset`: Evaluation dataset (`"tidyvoice_dev"` for development set)

#### 2. Run the Complete Pipeline

To run all stages (data preparation through evaluation):

```bash
./run_vox_custom_paak.sh --stage 1 --stop_stage 7
```

#### 3. Run Fine-tuning Only

If you have already prepared the data and want to fine-tune the pretrained model:

```bash
./run_vox_custom_paak.sh --stage 9 --stop_stage 9
```

#### 4. Run Evaluation Only

To evaluate an existing model:

```bash
./run_vox_custom_paak.sh --stage 4 --stop_stage 7
```

### Configuration Options

Key parameters in `run_vox_custom_paak.sh`:

- `eval_dataset`: Set to `"tidyvoice_dev"` for development phase evaluation
- `score_norm_method`: `"asnorm"` or `"snorm"` for score normalization
- `top_n`: Number of cohort speakers for score normalization (default: 300)
- `gpus`: GPU IDs in list format, e.g., `"[0]"` or `"[0,1]"` for multi-GPU
- `data_type`: `"shard"` or `"raw"` for data format

---

## Evaluation Datasets

The script supports multiple evaluation datasets:

### Development Phase

- **`tidyvoice_dev`**: Development set with ground truth labels (used during development phase)

### Evaluation Phase

The following evaluation sets will be released during the evaluation phase:

- **`tidyvoice_eval1`**: Trial List 1 - Enrollment from seen languages, test from unseen languages
- **`tidyvoice_eval2`**: Trial List 2 - Both enrollment and test from unseen languages (38 unseen languages)

To use these evaluation sets, set `eval_dataset` in the script:

```bash
eval_dataset="tidyvoice_eval1"  # or "tidyvoice_eval2"
```

### Preparing Evaluation Data

The easiest way to prepare the evaluation datasets (`tidyvoice_eval1` and `tidyvoice_eval2`) is to use the provided `prepare_eval_data.py` script:

1. **Prepare the evaluation dataset** using `prepare_eval_data.py`:

```bash
# For tidyvoice_eval1
python prepare_eval_data.py
```

Make sure to configure the script with the correct paths:
- `WAV_ROOT`: Directory containing the evaluation WAV files (e.g., `TidyVoiceX_eval`)
- `TRIAL_FILE`: Path to the trial file (e.g., `eval_trials/tidyvoice_eval1.txt`)
- `OUTPUT_DIR`: Will be set to `data/tidyvoice_eval1` (or `data/tidyvoice_eval2`)

The script will:
- Extract required WAV files from the trial file
- Create `wav.scp` and `utt2spk` files
- Convert trials to Kaldi format
- Optionally create shard lists for efficient data loading

2. **Run the pretrained baseline** on the evaluation sets:

After preparing the data, edit `run.sh` and:
- Set `eval_dataset` to `"tidyvoice_eval1"` or `"tidyvoice_eval2"`
- Run stages 4-5 to extract embeddings and compute scores:

```bash
./run.sh --stage 4 --stop_stage 5
```

This will:
- **Stage 4**: Extract embeddings for the evaluation dataset using the pretrained baseline model
- **Stage 5**: Compute scores and metrics (EER, MinDCF) on the evaluation dataset

**Note**: This approach uses the pretrained baseline model (without fine-tuning) to evaluate performance on the evaluation sets. For best results, you should fine-tune the model first (stage 9) and then evaluate.

---

## Expected Directory Structure

After running the pipeline, you should have the following structure:

```
data/
├── tidyvoice_train/          # Training data
│   ├── wav.scp
│   ├── utt2spk
│   ├── shard.list
│   └── shards/
├── tidyvoice_dev/             # Development data
│   ├── wav.scp
│   ├── utt2spk
│   ├── trials/
│   │   └── trials.kaldi
│   └── shards/
├── musan/                     # MUSAN augmentation data
└── rirs/                      # RIRS_NOISES augmentation data

exp/
└── samresnet34_voxblink_ft_tidy/  # Experiment directory
    ├── models/
    ├── embeddings/
    ├── scores/
    └── config.yaml
```

---

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. **Reduce batch size**: Edit `conf/voxblink_resnet34_ft.yaml` and reduce `batch_size` in `dataloader_args`
2. **Use fewer GPUs**: Set `gpus="[0]"` and `--nproc_per_node=1`
3. **Reduce number of workers**: Lower `num_workers` in the config file
4. **Use a GPU with more memory**: Check available GPUs with `nvidia-smi`

### Dataset Download Issues

- Verify your API key is correct
- Check your internet connection
- Ensure you have sufficient disk space (the dataset requires several GB)

---

## Citation

If you use this baseline in your research, please cite:

```bibtex
@inproceedings{tidyvoice2026,
  title={TidyVoice Challenge: Cross-Lingual Speaker Verification},
  author={...},
  booktitle={Interspeech},
  year={2026}
}
```

---

## Contact

For questions about the challenge or this baseline:

- **Aref Farhadipour**: aref.farhadipour@uzh.ch
- **Challenge Website**: [https://tidyvoice2026.github.io](https://tidyvoice2026.github.io)

---

## License

This baseline code is adapted from the WeSpeaker toolkit. Please refer to the original WeSpeaker license for details.

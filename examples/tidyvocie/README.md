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

| Architecture | Pretraining Data | Fine-tuning Data | EER (%) | MinDCF | Model |
|:-------------|:----------------|:----------------|:-------:|:------:|:-----:|
| SimAM-ResNet34 | VoxBlink2 + VoxCeleb2 | TidyVoiceX Train | 3.07 | 0.82 | [🤗 Download](https://huggingface.co/areffarhadi/Resnet34-tidyvoiceX-ASV) |

These results demonstrate the baseline's capability to handle cross-lingual speaker verification tasks.

---

## Prerequisites

### 1. API Key Setup (Required)

**IMPORTANT**: Before running the code, you must obtain and configure your Mozilla Common Voice API key to download the TidyVoiceX dataset.

1. Get your API key from: [https://datacollective.mozillafoundation.org/api-reference](https://datacollective.mozillafoundation.org/api-reference)

2. Edit `run.sh` and set your API key:

```bash
# CommonVoice DataCollective API Key
# Get your API key from: https://datacollective.mozillafoundation.org/api-reference
TIDYVOICE_API_KEY="Enter your Mozilla CommonVoice API key here"
```

Replace `"Enter your Mozilla CommonVoice API key here"` with your actual API key.

### 2. Installation

Follow these steps to set up the environment:

#### Step 1: Clone the Repository

```bash
git clone https://github.com/areffarhadi/wespeaker.git
cd wespeaker
```

#### Step 2: Create Virtual Environment

```bash
python3.9 -m venv we-env
```

#### Step 3: Activate Virtual Environment

```bash
# On Linux/Mac:
source we-env/bin/activate

# On Windows:
we-env\Scripts\activate
```

#### Step 4: Install PyTorch

Install PyTorch 1.12.1 with CUDA 11.3 support:

```bash
pip install torch==1.12.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

> **Note**: If you need a different CUDA version or CPU-only version, visit [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/) for the appropriate pip command.

#### Step 5: Install Requirements

```bash
pip install -r requirements.txt
```

#### Step 6: Install Additional Packages

```bash
pip install whisper datacollective
```

#### Step 7: Install Pre-commit (Optional but Recommended)

For clean and tidy code:

```bash
pre-commit install
```

#### (Optional) Install Hugging Face Hub

If you want the **baseline pretrained model** to be downloaded automatically:

```bash
pip install huggingface_hub
```

Otherwise, you can download the model manually from Hugging Face (see below).

---

### 3. Download the Baseline Pretrained Model

The official baseline model checkpoint for this recipe is hosted on Hugging Face:

- **Model page**: `https://huggingface.co/areffarhadi/Resnet34-tidyvoiceX-ASV`

You have **two options** to get the model:

#### Option A: Automatic download (dataset + model)

Use the provided downloader script, which will:
- Download the **TidyVoiceX dataset** from Mozilla DataCollective
- Download the **baseline pretrained model** from Hugging Face and place it in the expected directory

```bash
cd examples/tidyvocie

# <DATA_ROOT> is where you want the TidyVoiceX data to be stored
python local/download_tidyvoice.py <DATA_ROOT> <TIDYVOICE_API_KEY>
```

This will:
- Save the dataset into `<DATA_ROOT>`
- Save the model files (`avg_model.pt` and `config.yaml`) into:

```text
exp/samresnet34_voxblink_ft_tidy/models
```

> **Note**: For the automatic Hugging Face download to work, you must have `huggingface_hub` installed.

After downloading the model, you need to:

1. **Prepare the data** (download dataset and create manifests):
   ```bash
   ./run.sh --stage 1 --stop_stage 2
   ```

2. **Run inference** (extract embeddings and compute scores):
   ```bash
   ./run.sh --stage 4 --stop_stage 5
   ```

This will extract embeddings and score the evaluation dataset using the **baseline model**.

#### Option B: Manual download from Hugging Face

If you prefer to download manually:

1. Go to the model page:  
   `https://huggingface.co/areffarhadi/Resnet34-tidyvoiceX-ASV`
2. Download:
   - `avg_model.pt`
   - `config.yaml`
3. Place both files into:

```text
exp/samresnet34_voxblink_ft_tidy/models
```

Then you need to:

1. **Prepare the data** (download dataset and create manifests):
   ```bash
   ./run.sh --stage 1 --stop_stage 2
   ```

2. **Run inference** (extract embeddings and compute scores):
   ```bash
   ./run.sh --stage 4 --stop_stage 5
   ```

This will run inference with the pretrained baseline model.

---

## Running the Baseline

The main script is `run.sh`, which performs the complete pipeline from data preparation to evaluation.

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

> **⚠️ Important Note on Stages 6 and 7**: Stages 6 (score normalization) and 7 (score calibration) are **not valid for proper evaluation** when using the development set (`tidyvoice_dev`) because they utilize the same dev data for both:
> - Training the normalization/calibration parameters (cohort set)
> - Evaluating on the same data
> 
> This creates **data leakage** and will produce overly optimistic results that do not reflect true model performance. These stages are included in the codebase to demonstrate the implementation correctness, but **should not be used for final evaluation metrics** on the development set.
> 
> For proper evaluation, use only **stages 4-5** (embedding extraction and scoring) without normalization/calibration, or ensure that normalization/calibration use a different dataset than the one being evaluated.

### Quick Start

#### 1. Configure the Script

Edit `run.sh` and set:
- `TIDYVOICE_API_KEY`: Your API key (see above)
- `stage` and `stop_stage`: Which stages to run
- `gpus`: GPU IDs to use (e.g., `"[0]"` for single GPU)
- `eval_dataset`: Evaluation dataset (`"tidyvoice_dev"` for development set)

#### 2. Run the Complete Pipeline

To run all stages (data preparation through evaluation):

```bash
# Note: This includes stages 6-7 which have data leakage issues for tidyvoice_dev
# For proper evaluation, use: ./run.sh --stage 1 --stop_stage 5
./run.sh --stage 1 --stop_stage 7
```

> **Note**: The above command includes stages 6-7, which are included for code correctness demonstration but should not be used for final evaluation metrics on `tidyvoice_dev` due to data leakage (see warning in Pipeline Stages section above).

#### 3. Run Fine-tuning Only

If you have already prepared the data and want to fine-tune the pretrained model:

```bash
./run.sh --stage 9 --stop_stage 9
```

#### 4. Run Inference/Evaluation

To run inference with the pretrained baseline model:

**Step 1**: Download data and prepare manifests (if not already done):

```bash
./run.sh --stage 1 --stop_stage 2
```

This will:
- Download the TidyVoiceX dataset (and augmentation data)
- Prepare data manifests (`wav.scp`, `utt2spk`, etc.)

**Step 2**: Run inference to extract embeddings and compute scores:

```bash
./run.sh --stage 4 --stop_stage 5
```

This will:
- Extract embeddings for the evaluation dataset using the pretrained baseline model
- Compute scores and metrics (EER, MinDCF) on the evaluation dataset

> **Note**: Stages 6-7 (normalization/calibration) are included for code correctness but should NOT be used for evaluation on `tidyvoice_dev` due to data leakage (see warning above in Pipeline Stages section). For proper evaluation, use only stages 4-5.

### Configuration Options

Key parameters in `run.sh`:

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

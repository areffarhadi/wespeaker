#!/usr/bin/env python3

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge


import os
import sys
from datacollective import DataCollective

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

DATASET_ID = "cmihtsewu023so207xot1iqqw"
HF_MODEL_ID = "areffarhadi/Resnet34-tidyvoiceX-ASV"

def main():
    if len(sys.argv) < 3:
        print("Usage: python download_tidyvoice.py <output_directory> <api_key>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    api_key = sys.argv[2]
    
    if not api_key or api_key.strip() == "":
        print("ERROR: API key is required")
        print("Please provide your DataCollective API key")
        sys.exit(1)
    
    print("TidyVoice 2026 Challenge Auto-Downloader")
    print("==========================================")
    os.makedirs(output_dir, exist_ok=True)
    
    os.environ["MDC_API_KEY"] = api_key
    os.environ["MDC_DOWNLOAD_PATH"] = output_dir
    
    print(f"Saving to: {output_dir}")
    
    # 1) Download TidyVoiceX dataset
    try:
        client = DataCollective()
        client.get_dataset(DATASET_ID)
        print("\nTidyVoiceX dataset download completed successfully!")
        print(f"Dataset saved in: {output_dir}\n")
    except Exception as e:
        print("\nERROR while downloading TidyVoiceX dataset:")
        print(str(e))
        print("\nMake sure datacollective is installed: pip install datacollective\n")
        sys.exit(1)

    # 2) (Optional) Download pretrained baseline model from Hugging Face
    print("Downloading pretrained baseline model from Hugging Face (optional step)...")
    exp_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "exp",
        "samresnet34_voxblink_ft_tidy",
        "models",
    )
    os.makedirs(exp_model_dir, exist_ok=True)

    if not HF_AVAILABLE:
        print(
            "\nWARNING: Could not import 'huggingface_hub'. "
            "To auto-download the baseline model, install it with:\n"
            "  pip install huggingface_hub\n"
            "Then re-run this script, or manually download the model from:\n"
            "  https://huggingface.co/areffarhadi/Resnet34-tidyvoiceX-ASV\n"
            "and place 'avg_model.pt' and 'config.yaml' into:\n"
            f"  {exp_model_dir}\n"
        )
        return

    for filename in ["avg_model.pt", "config.yaml"]:
        try:
            print(f"  - Downloading {filename} from {HF_MODEL_ID} ...")
            local_path = hf_hub_download(
                repo_id=HF_MODEL_ID,
                filename=filename,
                local_dir=exp_model_dir,
                local_dir_use_symlinks=False,
            )
            print(f"    Saved to: {local_path}")
        except Exception as e:
            print(f"\nWARNING: Failed to download {filename} from Hugging Face:")
            print(str(e))
            print(
                "You can still manually download the files from:\n"
                "  https://huggingface.co/areffarhadi/Resnet34-tidyvoiceX-ASV\n"
                "and place 'avg_model.pt' and 'config.yaml' into:\n"
                f"  {exp_model_dir}\n"
            )
            break

    print(
        "\nIf the model files were downloaded successfully, you can run inference with:\n"
        "  ./run.sh --stage 4 --stop_stage 5\n"
    )

if __name__ == "__main__":
    main()


#!/usr/bin/env python3

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge


import os
import sys
from datacollective import DataCollective

DATASET_ID = "cmihtsewu023so207xot1iqqw"

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
    
    try:
        client = DataCollective()
        client.get_dataset(DATASET_ID)
        print("\nDownload completed successfully!")
        print(f"Dataset saved in: {output_dir}\n")
    except Exception as e:
        print("\nERROR while downloading:")
        print(str(e))
        print("\nMake sure datacollective is installed: pip install datacollective\n")
        sys.exit(1)

if __name__ == "__main__":
    main()


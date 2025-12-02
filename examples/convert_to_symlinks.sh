#!/bin/bash
# Convert tools and wespeaker text files to symlinks
# This script should be run on a filesystem that supports symlinks (not exFAT)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Converting tools and wespeaker files to symlinks..."

# Directories with ../../../tools and ../../../wespeaker
for dir in cnceleb/v2 cnceleb/v3_finetune sre/v2 sre/v3 voxceleb/v2 voxceleb/v2_deprecated voxconverse/v1 voxconverse/v2 tidyvocie; do
  if [ -d "$dir" ]; then
    if [ -f "$dir/tools" ] && [ ! -L "$dir/tools" ]; then
      target=$(cat "$dir/tools" | tr -d '\n\r')
      echo "Converting $dir/tools -> $target"
      rm -f "$dir/tools"
      ln -s "$target" "$dir/tools"
    fi
    if [ -f "$dir/wespeaker" ] && [ ! -L "$dir/wespeaker" ]; then
      target=$(cat "$dir/wespeaker" | tr -d '\n\r')
      echo "Converting $dir/wespeaker -> $target"
      rm -f "$dir/wespeaker"
      ln -s "$target" "$dir/wespeaker"
    fi
  fi
done

# Directories with ../../../../tools and ../../../../wespeaker
for dir in voxceleb/v1/Whisper-PMFA voxceleb/v3/dino voxceleb/v3/moco voxceleb/v3/simclr; do
  if [ -d "$dir" ]; then
    if [ -f "$dir/tools" ] && [ ! -L "$dir/tools" ]; then
      target=$(cat "$dir/tools" | tr -d '\n\r')
      echo "Converting $dir/tools -> $target"
      rm -f "$dir/tools"
      ln -s "$target" "$dir/tools"
    fi
    if [ -f "$dir/wespeaker" ] && [ ! -L "$dir/wespeaker" ]; then
      target=$(cat "$dir/wespeaker" | tr -d '\n\r')
      echo "Converting $dir/wespeaker -> $target"
      rm -f "$dir/wespeaker"
      ln -s "$target" "$dir/wespeaker"
    fi
  fi
done

echo "Done! All files converted to symlinks."


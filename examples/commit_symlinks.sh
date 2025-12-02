#!/bin/bash
# Script to commit tools and wespeaker as symlinks in git
# This works even on exFAT filesystem that doesn't support symlinks
# Run this from the repository root (not from examples directory)

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

echo "Converting tools and wespeaker files to symlinks in git..."
echo "Note: This will modify git's index, not the actual files on disk"

cd examples

# Find all tools and wespeaker files
for file in $(find . -type f \( -name "tools" -o -name "wespeaker" \)); do
    # Read the target path from the file
    target=$(cat "$file" | tr -d '\n\r')
    
    # Get the relative path from repo root
    git_path="examples/$file"
    
    echo "Converting $git_path -> $target"
    
    # Remove from git index
    git rm --cached "$git_path" 2>/dev/null || true
    
    # Create the symlink content (just the target path)
    echo -n "$target" | git hash-object -w --stdin > /tmp/symlink_hash.txt
    symlink_hash=$(cat /tmp/symlink_hash.txt)
    
    # Add as symlink (mode 120000 is symlink)
    git update-index --add --cacheinfo 120000 "$symlink_hash" "$git_path"
done

rm -f /tmp/symlink_hash.txt

echo ""
echo "Done! The files are now staged as symlinks in git."
echo ""
echo "To verify, run: git ls-files -s examples/*/v*/tools examples/*/v*/wespeaker"
echo "You should see mode 120000 (symlink) for these files."
echo ""
echo "To commit: git commit -m 'Convert tools and wespeaker to symlinks'"


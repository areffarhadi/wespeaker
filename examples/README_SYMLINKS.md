# Converting tools and wespeaker to Symlinks

## Important: YOU Need to Do This Before Committing

**You** need to convert these to symlinks and commit them as symlinks in git. Once committed, users who clone your repository will automatically get symlinks (on filesystems that support them). They don't need to do anything.

## Current Status

Due to the exFAT filesystem limitation (which doesn't support symlinks), the `tools` and `wespeaker` files in example directories are currently stored as text files containing relative paths.

## Converting to Symlinks for Git

### Option 1: Use the Automated Script (Recommended)

If you're on exFAT or any filesystem, use this script to commit them as symlinks:

```bash
# From the repository root (not examples directory)
cd /path/to/wespeaker
./examples/commit_symlinks.sh

# Then commit
git commit -m "Convert tools and wespeaker to symlinks"
```

This script works even on exFAT because it directly modifies git's index, not the filesystem.

### Option 2: Convert on a filesystem that supports symlinks

1. Clone or copy the repository to a filesystem that supports symlinks (ext4, NTFS with proper settings, etc.)
2. Run the conversion script:
   ```bash
   cd examples
   ./convert_to_symlinks.sh
   ```
3. Commit the changes:
   ```bash
   git add examples/*/v*/tools examples/*/v*/wespeaker
   git commit -m "Convert tools and wespeaker to symlinks"
   ```

## Verification

After conversion, verify symlinks are working:

```bash
cd examples
find . -type l \( -name "tools" -o -name "wespeaker" \) | while read link; do
  echo "$link -> $(readlink "$link")"
done
```

All should show the correct relative paths (e.g., `../../../tools`, `../../../../wespeaker`).


# Converting to Symlinks on NTFS

## Steps to Convert on NTFS Storage

1. **Copy the repository to your NTFS drive:**
   ```bash
   # Example: if your NTFS drive is mounted at /mnt/ntfs
   cp -r /media/rf/T91/tidyvoice_challenge/wespeaker /mnt/ntfs/wespeaker
   cd /mnt/ntfs/wespeaker
   ```

2. **Verify symlink support:**
   ```bash
   # Test if symlinks work
   ln -s /tmp test_link && ls -l test_link && rm test_link
   # If this works, symlinks are supported
   ```

3. **Run the conversion script:**
   ```bash
   cd examples
   ./convert_to_symlinks.sh
   ```

4. **Verify the conversion:**
   ```bash
   # Check that files are now symlinks
   find . -type l \( -name "tools" -o -name "wespeaker" \) | head -5
   # Should show symlinks like: ./cnceleb/v2/tools -> ../../../tools
   ```

5. **Commit the changes:**
   ```bash
   cd ..  # Back to repository root
   git add examples/*/v*/tools examples/*/v*/wespeaker
   git status  # Verify they show as modified
   git commit -m "Convert tools and wespeaker to symlinks"
   ```

6. **Copy back to exFAT (optional):**
   If you want to continue working on exFAT, you can copy the repository back. Git will still track them as symlinks even though exFAT can't display them as such.

## Note on NTFS Mount Options

If symlinks don't work on your NTFS mount, you may need to remount with proper options:

```bash
# Check current mount options
mount | grep ntfs

# Remount with symlink support (if needed)
sudo mount -o remount,user_id=1000,group_id=1000,permissions /dev/sdXX /mnt/ntfs
```

However, NTFS-3G usually supports symlinks by default on modern Linux systems.


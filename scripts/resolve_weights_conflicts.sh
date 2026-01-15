#!/bin/bash
# Replace conflicting checkpoints - keep source (iddaw_*) over target (idd-aw_*)
# The iddaw_* versions are newer (Jan 15) and may have label fixes
# Created: 2026-01-15

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"

echo "=========================================="
echo "WEIGHTS Directory Conflict Resolution"
echo "=========================================="
echo "Strategy: Keep SOURCE (iddaw_*) over TARGET (idd-aw_*)"
echo ""

replaced=0

replace_with_source() {
    local src_dir="$1"
    local tgt_dir="$2"
    
    if [ ! -d "$src_dir" ]; then
        echo "  Source does not exist, skipping"
        return
    fi
    
    echo ""
    echo "Processing: $(basename $(dirname "$src_dir"))/$(basename "$src_dir")"
    
    # For each subdirectory in source
    for src_subdir in "$src_dir"/*/; do
        [ ! -d "$src_subdir" ] && continue
        subname=$(basename "$src_subdir")
        tgt_subdir="$tgt_dir/$subname"
        
        if [ -d "$tgt_subdir" ]; then
            # Backup target and replace with source
            backup_name="${subname}_old_backup"
            echo "  REPLACE: $subname"
            echo "    - Backing up target to ${backup_name}"
            mv "$tgt_subdir" "$tgt_dir/${backup_name}"
            echo "    - Moving source to target"
            mv "$src_subdir" "$tgt_subdir"
            replaced=$((replaced + 1))
        fi
    done
    
    # Remove source if empty
    if [ -z "$(ls -A "$src_dir" 2>/dev/null)" ]; then
        echo "  Removing empty source directory: $(basename "$src_dir")"
        rmdir "$src_dir"
    fi
}

echo "=== Processing baseline ==="
replace_with_source "$WEIGHTS_ROOT/baseline/iddaw_ad" "$WEIGHTS_ROOT/baseline/idd-aw_ad"
replace_with_source "$WEIGHTS_ROOT/baseline/iddaw_cd" "$WEIGHTS_ROOT/baseline/idd-aw_cd"

echo "=== Processing gen_Attribute_Hallucination ==="
replace_with_source "$WEIGHTS_ROOT/gen_Attribute_Hallucination/iddaw_cd" "$WEIGHTS_ROOT/gen_Attribute_Hallucination/idd-aw_cd"

echo "=== Processing gen_CNetSeg ==="
replace_with_source "$WEIGHTS_ROOT/gen_CNetSeg/iddaw_ad" "$WEIGHTS_ROOT/gen_CNetSeg/idd-aw_ad"
replace_with_source "$WEIGHTS_ROOT/gen_CNetSeg/iddaw_cd" "$WEIGHTS_ROOT/gen_CNetSeg/idd-aw_cd"

echo "=== Processing gen_stargan_v2 ==="
replace_with_source "$WEIGHTS_ROOT/gen_stargan_v2/iddaw_cd" "$WEIGHTS_ROOT/gen_stargan_v2/idd-aw_cd"

echo "=== Processing gen_step1x_v1p2 ==="
replace_with_source "$WEIGHTS_ROOT/gen_step1x_v1p2/iddaw_cd" "$WEIGHTS_ROOT/gen_step1x_v1p2/idd-aw_cd"

echo "=== Processing photometric_distort ==="
replace_with_source "$WEIGHTS_ROOT/photometric_distort/iddaw_cd" "$WEIGHTS_ROOT/photometric_distort/idd-aw_cd"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "  Replaced: $replaced models"
echo ""
echo "Old versions backed up with _old_backup suffix."
echo "Review and delete backups when ready."

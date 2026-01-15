#!/bin/bash
# Merge WEIGHTS directories - copy non-conflicting subdirectories
# Created: 2026-01-15

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"

echo "=========================================="
echo "WEIGHTS Directory Merge Script"
echo "=========================================="
echo ""

merged=0
skipped=0

merge_to_target() {
    local src_dir="$1"
    local tgt_dir="$2"
    
    echo ""
    echo "Processing: $(basename $(dirname "$src_dir"))/$(basename "$src_dir") → $(basename "$tgt_dir")"
    
    if [ ! -d "$src_dir" ]; then
        echo "  Source does not exist, skipping"
        return
    fi
    
    if [ ! -d "$tgt_dir" ]; then
        echo "  Target does not exist, doing simple rename"
        mv "$src_dir" "$tgt_dir"
        merged=$((merged + 1))
        return
    fi
    
    # Merge: copy each subdirectory that doesn't exist in target
    for subdir in "$src_dir"/*/; do
        [ ! -d "$subdir" ] && continue
        subname=$(basename "$subdir")
        
        if [ -d "$tgt_dir/$subname" ]; then
            echo "  SKIP (exists): $subname"
            skipped=$((skipped + 1))
        else
            echo "  COPY: $subname"
            mv "$subdir" "$tgt_dir/"
            merged=$((merged + 1))
        fi
    done
    
    # Remove source if empty
    if [ -z "$(ls -A "$src_dir" 2>/dev/null)" ]; then
        echo "  Removing empty source directory"
        rmdir "$src_dir"
    else
        echo "  Source still has contents, not removing"
    fi
}

echo "=== baseline ==="
merge_to_target "$WEIGHTS_ROOT/baseline/iddaw_ad" "$WEIGHTS_ROOT/baseline/idd-aw_ad"
merge_to_target "$WEIGHTS_ROOT/baseline/iddaw_cd" "$WEIGHTS_ROOT/baseline/idd-aw_cd"

echo "=== gen_Attribute_Hallucination ==="
merge_to_target "$WEIGHTS_ROOT/gen_Attribute_Hallucination/iddaw_cd" "$WEIGHTS_ROOT/gen_Attribute_Hallucination/idd-aw_cd"

echo "=== gen_CNetSeg ==="
merge_to_target "$WEIGHTS_ROOT/gen_CNetSeg/iddaw_ad" "$WEIGHTS_ROOT/gen_CNetSeg/idd-aw_ad"
merge_to_target "$WEIGHTS_ROOT/gen_CNetSeg/iddaw_cd" "$WEIGHTS_ROOT/gen_CNetSeg/idd-aw_cd"

echo "=== gen_CUT ==="
merge_to_target "$WEIGHTS_ROOT/gen_CUT/iddaw_cd" "$WEIGHTS_ROOT/gen_CUT/idd-aw_cd"

echo "=== gen_IP2P ==="
merge_to_target "$WEIGHTS_ROOT/gen_IP2P/iddaw_cd" "$WEIGHTS_ROOT/gen_IP2P/idd-aw_cd"

echo "=== gen_Qwen_Image_Edit ==="
merge_to_target "$WEIGHTS_ROOT/gen_Qwen_Image_Edit/iddaw_ad" "$WEIGHTS_ROOT/gen_Qwen_Image_Edit/idd-aw_ad"

echo "=== gen_TSIT ==="
merge_to_target "$WEIGHTS_ROOT/gen_TSIT/iddaw_ad" "$WEIGHTS_ROOT/gen_TSIT/idd-aw_ad"

echo "=== gen_cycleGAN ==="
merge_to_target "$WEIGHTS_ROOT/gen_cycleGAN/iddaw_cd" "$WEIGHTS_ROOT/gen_cycleGAN/idd-aw_cd"

echo "=== gen_cyclediffusion ==="
merge_to_target "$WEIGHTS_ROOT/gen_cyclediffusion/iddaw_cd" "$WEIGHTS_ROOT/gen_cyclediffusion/idd-aw_cd"

echo "=== gen_stargan_v2 ==="
merge_to_target "$WEIGHTS_ROOT/gen_stargan_v2/iddaw_cd" "$WEIGHTS_ROOT/gen_stargan_v2/idd-aw_cd"

echo "=== gen_step1x_v1p2 ==="
merge_to_target "$WEIGHTS_ROOT/gen_step1x_v1p2/iddaw_cd" "$WEIGHTS_ROOT/gen_step1x_v1p2/idd-aw_cd"

echo "=== photometric_distort ==="
merge_to_target "$WEIGHTS_ROOT/photometric_distort/iddaw_cd" "$WEIGHTS_ROOT/photometric_distort/idd-aw_cd"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "  Merged: $merged subdirectories"
echo "  Skipped (conflict): $skipped subdirectories"

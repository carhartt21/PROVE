#!/bin/bash
# =============================================================================
# Move Test Results from results/ to WEIGHTS directories
# Updated: 2026-01-20
# =============================================================================
#
# This script moves test results from the results/ directory to the 
# corresponding WEIGHTS/test_results_detailed/ directories.
# It reads the checkpoint_path from results.json to determine if
# results should go to WEIGHTS (Stage 1) or WEIGHTS_STAGE_2 (Stage 2).
#
# Usage:
#   ./scripts/move_test_results_to_weights.sh [--dry-run]
#
# =============================================================================

set -e
cd ${HOME}/repositories/PROVE

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

moved_s1=0
moved_s2=0
skipped=0

# Find all results.json files in results/std_* directories
echo "=== Processing Results ==="
for result_file in $(find results/std_* -name "results.json" 2>/dev/null); do
    # Get the directory containing results.json
    src_dir=$(dirname "$result_file")
    timestamp=$(basename "$src_dir")
    
    # Get checkpoint path from results.json
    ckpt_path=$(grep -oP '"checkpoint_path":\s*"\K[^"]+' "$result_file" 2>/dev/null || echo "")
    
    if [ -z "$ckpt_path" ]; then
        echo "  [ERROR] No checkpoint_path in $result_file"
        continue
    fi
    
    # Determine if Stage 1 or Stage 2 based on checkpoint path
    if echo "$ckpt_path" | grep -q "WEIGHTS_STAGE_2"; then
        stage="Stage2"
        weights_root="${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2"
    else
        stage="Stage1"
        weights_root="${AWARE_DATA_ROOT}/WEIGHTS"
    fi
    
    # Extract strategy/dataset/model from checkpoint path
    # e.g., ${AWARE_DATA_ROOT}/WEIGHTS/std_autoaugment/bdd10k/deeplabv3plus_r50/iter_80000.pth
    rel_path=$(echo "$ckpt_path" | sed 's|.*/WEIGHTS[^/]*/||;s|/iter_[0-9]*.pth||')
    
    # Build destination path
    dst_base="${weights_root}/${rel_path}/test_results_detailed"
    dst_dir="${dst_base}/${timestamp}"
    
    # Check if destination already exists
    if [ -d "$dst_dir" ]; then
        echo "  [SKIP] ${stage} ${rel_path}/${timestamp} - already exists"
        skipped=$((skipped + 1))
        continue
    fi
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "  [WOULD MOVE] ${stage} ${src_dir} -> ${dst_dir}"
    else
        # Create destination directory
        mkdir -p "$dst_base"
        # Copy the results (use cp to preserve source for safety)
        cp -r "$src_dir" "$dst_dir"
        echo "  [MOVED] ${stage} ${rel_path}/${timestamp}"
    fi
    
    if [ "$stage" == "Stage1" ]; then
        moved_s1=$((moved_s1 + 1))
    else
        moved_s2=$((moved_s2 + 1))
    fi
done

echo ""
echo "=== Summary ==="
if [ "$DRY_RUN" == "true" ]; then
    echo "Would move ${moved_s1} Stage 1 results, ${moved_s2} Stage 2 results"
    echo "Would skip ${skipped} (already exist)"
else
    echo "Moved ${moved_s1} Stage 1 results, ${moved_s2} Stage 2 results"
    echo "Skipped ${skipped} (already exist)"
fi

#!/bin/bash
# Cleanup script to:
# 1. Delete old buggy test_results_detailed/ directories
# 2. Rename test_results_detailed_fixed/ to test_results_detailed/
#
# This consolidates all test results to use the "fixed" version without the suffix

set -e

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
DRY_RUN=${1:-"false"}

if [ "$DRY_RUN" = "--dry-run" ] || [ "$DRY_RUN" = "-n" ]; then
    DRY_RUN="true"
    echo "=== DRY RUN MODE - No changes will be made ==="
fi

echo "=== Test Results Cleanup Script ==="
echo "Weights root: $WEIGHTS_ROOT"
echo ""

# Count directories
OLD_COUNT=$(find "$WEIGHTS_ROOT" -type d -name "test_results_detailed" 2>/dev/null | wc -l)
FIXED_COUNT=$(find "$WEIGHTS_ROOT" -type d -name "test_results_detailed_fixed" 2>/dev/null | wc -l)

echo "Found:"
echo "  - Old test_results_detailed: $OLD_COUNT"
echo "  - Fixed test_results_detailed_fixed: $FIXED_COUNT"
echo ""

# Step 1: Delete old test_results_detailed directories
echo "=== Step 1: Deleting old test_results_detailed directories ==="
deleted_count=0
while IFS= read -r old_dir; do
    if [ -d "$old_dir" ]; then
        # Check if there's a _fixed version alongside
        parent_dir=$(dirname "$old_dir")
        fixed_dir="${parent_dir}/test_results_detailed_fixed"
        
        if [ -d "$fixed_dir" ]; then
            echo "DELETE: $old_dir (has fixed version)"
            if [ "$DRY_RUN" = "false" ]; then
                rm -rf "$old_dir"
            fi
            ((deleted_count++))
        else
            echo "KEEP: $old_dir (no fixed version exists)"
        fi
    fi
done < <(find "$WEIGHTS_ROOT" -type d -name "test_results_detailed" 2>/dev/null)

echo ""
echo "Deleted $deleted_count old directories (where fixed version exists)"
echo ""

# Step 2: Rename test_results_detailed_fixed to test_results_detailed
echo "=== Step 2: Renaming test_results_detailed_fixed to test_results_detailed ==="
renamed_count=0
while IFS= read -r fixed_dir; do
    if [ -d "$fixed_dir" ]; then
        parent_dir=$(dirname "$fixed_dir")
        new_name="${parent_dir}/test_results_detailed"
        
        if [ ! -e "$new_name" ]; then
            echo "RENAME: $fixed_dir -> test_results_detailed"
            if [ "$DRY_RUN" = "false" ]; then
                mv "$fixed_dir" "$new_name"
            fi
            ((renamed_count++))
        else
            echo "SKIP: $fixed_dir (target exists - this shouldn't happen)"
        fi
    fi
done < <(find "$WEIGHTS_ROOT" -type d -name "test_results_detailed_fixed" 2>/dev/null)

echo ""
echo "Renamed $renamed_count directories"
echo ""

# Final verification
echo "=== Verification ==="
if [ "$DRY_RUN" = "false" ]; then
    NEW_OLD_COUNT=$(find "$WEIGHTS_ROOT" -type d -name "test_results_detailed" 2>/dev/null | wc -l)
    NEW_FIXED_COUNT=$(find "$WEIGHTS_ROOT" -type d -name "test_results_detailed_fixed" 2>/dev/null | wc -l)
    echo "After cleanup:"
    echo "  - test_results_detailed: $NEW_OLD_COUNT"
    echo "  - test_results_detailed_fixed: $NEW_FIXED_COUNT (should be 0)"
else
    echo "Run without --dry-run to apply changes"
fi

echo ""
echo "Done!"

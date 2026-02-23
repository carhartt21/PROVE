#!/bin/bash
# Script to delete faulty BDD10k fine-grained test results
# These results were generated with a bug that applied Cityscapes label conversion to BDD10k
# which corrupted class mappings (BDD10k labels are already in trainID format)

WEIGHTS_DIR="${AWARE_DATA_ROOT}/WEIGHTS"

echo "======================================"
echo "Delete Faulty BDD10k Test Results"
echo "======================================"
echo ""
echo "BUG: Old fine_grained_test.py incorrectly applied"
echo "CityscapesLabelIdToTrainId to BDD10k dataset."
echo "BDD10k labels are already in trainID format, so this"
echo "corrupted class mappings causing area_label=0 for classes 8-18."
echo ""

# Find all faulty directories
DIRS=$(find "$WEIGHTS_DIR" -path "*/bdd10k/*" -name "test_results_detailed" -type d 2>/dev/null)

if [ -z "$DIRS" ]; then
    echo "No faulty test results found."
    exit 0
fi

# Count
COUNT=$(echo "$DIRS" | wc -l)
echo "Found $COUNT directories with faulty BDD10k test results:"
echo ""

# Calculate total size
TOTAL_SIZE=0
while IFS= read -r dir; do
    SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
    # Extract strategy and model from path
    RELATIVE=${dir#$WEIGHTS_DIR/}
    echo "  $SIZE  $RELATIVE"
    # Get size in KB for total
    SIZE_KB=$(du -sk "$dir" 2>/dev/null | cut -f1)
    TOTAL_SIZE=$((TOTAL_SIZE + SIZE_KB))
done <<< "$DIRS"

# Convert total to human readable
if [ $TOTAL_SIZE -gt 1048576 ]; then
    TOTAL_HR=$(echo "scale=2; $TOTAL_SIZE/1048576" | bc)G
elif [ $TOTAL_SIZE -gt 1024 ]; then
    TOTAL_HR=$(echo "scale=2; $TOTAL_SIZE/1024" | bc)M
else
    TOTAL_HR="${TOTAL_SIZE}K"
fi

echo ""
echo "Total size: $TOTAL_HR"
echo ""

# Dry run or actual delete
if [ "$1" == "--delete" ]; then
    echo "Deleting directories..."
    echo ""
    
    DELETED=0
    FAILED=0
    
    while IFS= read -r dir; do
        RELATIVE=${dir#$WEIGHTS_DIR/}
        if rm -rf "$dir" 2>/dev/null; then
            echo "  Deleted: $RELATIVE"
            DELETED=$((DELETED + 1))
        else
            echo "  FAILED: $RELATIVE (permission denied)"
            FAILED=$((FAILED + 1))
        fi
    done <<< "$DIRS"
    
    echo ""
    echo "Deleted: $DELETED directories"
    if [ $FAILED -gt 0 ]; then
        echo "Failed: $FAILED directories (permission denied)"
    fi
    echo ""
    echo "Now run the re-test jobs to generate corrected results:"
    echo "  python scripts/retest_bdd10k_fine_grained.py --submit-all"
else
    echo "This is a DRY RUN. No files have been deleted."
    echo ""
    echo "To delete, run:"
    echo "  bash scripts/delete_faulty_bdd10k_results.sh --delete"
    echo ""
    echo "After deletion, re-run tests with:"
    echo "  python scripts/retest_bdd10k_fine_grained.py --submit-all"
fi

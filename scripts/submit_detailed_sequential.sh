#!/bin/bash
# PROVE - Submit Missing Detailed Tests (Sequential per Dataset)
#
# This script submits fine-grained (detailed) test jobs one at a time per dataset,
# using LSF job dependencies to chain jobs so they run sequentially.
# This minimizes impact on the shared queue.
#
# Usage:
#   ./submit_detailed_sequential.sh [--dry-run] [--jobs-per-batch N]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Options
DRY_RUN=false
JOBS_PER_BATCH=5  # Number of jobs to run in parallel per batch

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --jobs-per-batch)
            JOBS_PER_BATCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--jobs-per-batch N]"
            exit 1
            ;;
    esac
done

echo "PROVE - Submit Missing Detailed Tests (Sequential per Dataset)"
echo "=============================================================="
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN"
else
    echo "Mode: LIVE SUBMISSION"
fi
echo "Jobs per batch: $JOBS_PER_BATCH"
echo ""

# Datasets to process in order
DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")

# Analyze configurations
echo "Analyzing configurations..."
mamba run -n prove python weights_analyzer.py --format json >/dev/null 2>&1

# Show summary per dataset
echo ""
echo "Jobs per dataset:"
for DATASET in "${DATASETS[@]}"; do
    COUNT=$(mamba run -n prove python3 -c "
import json
with open('weights_summary.json') as f:
    data = json.load(f)
missing = [c for c in data if c.get('has_test_results') and not c.get('has_detailed_test_results') and c['dataset'].lower() == '${DATASET,,}']
print(len(missing))
" 2>/dev/null)
    echo "  $DATASET: $COUNT jobs"
done
echo ""

# Ask for confirmation
if [ "$DRY_RUN" != true ]; then
    read -p "Submit all jobs? (y/n): " CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Process each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Processing dataset: $DATASET"
    echo "=============================================="
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        ./scripts/submit_missing_detailed_tests.sh --dataset "$DATASET" --dry-run --batch-size 100
    else
        # Submit jobs in small batches with pauses between
        ./scripts/submit_missing_detailed_tests.sh --dataset "$DATASET" --batch-size "$JOBS_PER_BATCH"
    fi
    
    if [ "$DRY_RUN" != true ]; then
        echo ""
        echo "Dataset $DATASET jobs submitted. Waiting 60 seconds before next dataset..."
        sleep 60
    fi
done

echo ""
echo "=== All datasets submitted ==="
echo ""
if [ "$DRY_RUN" != true ]; then
    echo "Check job status with: bjobs -w"
fi

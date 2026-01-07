#!/bin/bash
# PROVE - Submit Missing Detailed Tests
#
# This script submits fine-grained (detailed) test jobs for all configurations
# that have basic test results but are missing detailed per-domain metrics.
#
# Usage:
#   ./submit_missing_detailed_tests.sh [--dry-run] [--limit N] [--dataset NAME]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Options
DRY_RUN=false
LIMIT=0  # 0 = no limit
BATCH_SIZE=10
BATCH_DELAY=30  # seconds between batches
FILTER_DATASET=""  # empty = all datasets

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--limit N] [--batch-size N] [--dataset NAME]"
            exit 1
            ;;
    esac
done

echo "PROVE - Submit Missing Detailed Tests"
echo "======================================"
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN"
else
    echo "Mode: LIVE SUBMISSION"
fi
echo "Batch size: $BATCH_SIZE jobs"
echo "Batch delay: $BATCH_DELAY seconds"
if [ -n "$FILTER_DATASET" ]; then
    echo "Dataset filter: $FILTER_DATASET"
fi
if [ "$LIMIT" -gt 0 ]; then
    echo "Limit: $LIMIT jobs"
fi
echo ""

# Generate list of configurations to test
echo "Analyzing configurations..."

mamba run -n PROVE python weights_analyzer.py --format json >/dev/null 2>&1

# Create temp file with configs to test
CONFIGS_FILE=$(mktemp)

mamba run -n PROVE python3 << PYTHON_SCRIPT > "$CONFIGS_FILE"
import json

with open('weights_summary.json') as f:
    data = json.load(f)

# Filter for configs that have basic tests but no detailed tests
missing = [c for c in data if c.get('has_test_results') and not c.get('has_detailed_test_results')]

# Apply dataset filter if specified
filter_dataset = "${FILTER_DATASET}".lower()
if filter_dataset:
    missing = [c for c in missing if c['dataset'].lower() == filter_dataset]

# Sort by strategy, dataset, model
missing.sort(key=lambda x: (x['strategy'], x['dataset'], x['model']))

# Apply limit
limit = ${LIMIT}
if limit > 0:
    missing = missing[:limit]

# Dataset name mapping for proper casing
dataset_map = {
    'acdc': 'ACDC',
    'bdd10k': 'BDD10k',
    'bdd100k': 'BDD100k',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k'
}

# Output configs
for c in missing:
    dataset_proper = dataset_map.get(c['dataset'].lower(), c['dataset'])
    print(f"{c['strategy']} {dataset_proper} {c['model']}")
PYTHON_SCRIPT

TOTAL=$(wc -l < "$CONFIGS_FILE")
echo "Found $TOTAL configurations to test"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No configurations need detailed testing."
    rm -f "$CONFIGS_FILE"
    exit 0
fi

# Print summary
echo "Configurations to test:"
head -20 "$CONFIGS_FILE" | while read strategy dataset model; do
    echo "  $strategy / $dataset / $model"
done
if [ "$TOTAL" -gt 20 ]; then
    echo "  ... and $((TOTAL - 20)) more"
fi
echo ""

# Submit jobs
JOB_COUNT=0
BATCH_COUNT=0

while read strategy dataset model; do
    JOB_COUNT=$((JOB_COUNT + 1))
    
    echo "[$JOB_COUNT/$TOTAL] $strategy / $dataset / $model"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would submit: $SCRIPT_DIR/test_unified.sh submit-detailed --dataset $dataset --model $model --strategy $strategy"
    else
        $SCRIPT_DIR/test_unified.sh submit-detailed --dataset "$dataset" --model "$model" --strategy "$strategy"
    fi
    
    # Batch delay
    BATCH_COUNT=$((BATCH_COUNT + 1))
    if [ "$BATCH_COUNT" -ge "$BATCH_SIZE" ] && [ "$JOB_COUNT" -lt "$TOTAL" ]; then
        BATCH_COUNT=0
        echo ""
        echo "=== Batch complete ($JOB_COUNT/$TOTAL jobs). Pausing $BATCH_DELAY seconds... ==="
        echo ""
        if [ "$DRY_RUN" != true ]; then
            sleep $BATCH_DELAY
        fi
    fi
    
    sleep 0.5  # Small delay between submissions
    
done < "$CONFIGS_FILE"

echo ""
echo "=== Submission Complete ==="
echo "Total jobs: $JOB_COUNT"

rm -f "$CONFIGS_FILE"

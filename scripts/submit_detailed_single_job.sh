#!/bin/bash
# PROVE - Submit Single Batch Job for All Detailed Tests
#
# This script submits a single batch job that runs all missing detailed tests
# sequentially within that job. This is more queue-friendly than submitting
# hundreds of individual jobs.
#
# Usage:
#   ./submit_detailed_single_job.sh [--dataset NAME] [--dry-run]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Options
DRY_RUN=false
FILTER_DATASET=""
QUEUE="BatchGPU"
GPU_MEM="24G"
NUM_CPUS="4"
TIME_LIMIT="72:00"  # 72 hours max

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset NAME] [--queue NAME] [--gpu-mem SIZE] [--time HH:MM] [--dry-run]"
            exit 1
            ;;
    esac
done

# Build job name
if [ -n "$FILTER_DATASET" ]; then
    JOB_NAME="prove_detailed_all_${FILTER_DATASET}"
    DATASET_ARG="--dataset $FILTER_DATASET"
else
    JOB_NAME="prove_detailed_all_datasets"
    DATASET_ARG=""
fi

# Build the command to run
TEST_CMD="$SCRIPT_DIR/run_detailed_tests_sequential.sh $DATASET_ARG"

# Build bsub command
BSUB_CMD="bsub \
    -gpu \"num=1:gmem=${GPU_MEM}\" \
    -q ${QUEUE} \
    -R \"span[hosts=1]\" \
    -n ${NUM_CPUS} \
    -W ${TIME_LIMIT} \
    -oo \"logs/${JOB_NAME}_%J.log\" \
    -eo \"logs/${JOB_NAME}_%J.err\" \
    -L /bin/bash \
    -J \"${JOB_NAME}\" \
    \"${TEST_CMD}\""

echo "PROVE - Submit Detailed Tests (Single Job)"
echo "==========================================="
echo ""
echo "Job name:    $JOB_NAME"
echo "Queue:       $QUEUE"
echo "GPU memory:  $GPU_MEM"
echo "Time limit:  $TIME_LIMIT"
echo "CPUs:        $NUM_CPUS"
if [ -n "$FILTER_DATASET" ]; then
    echo "Dataset:     $FILTER_DATASET"
else
    echo "Dataset:     ALL (BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k)"
fi
echo ""

# Count configurations
echo "Counting configurations to test..."
mamba run -n PROVE python weights_analyzer.py --format json >/dev/null 2>&1

if [ -n "$FILTER_DATASET" ]; then
    FILTER_LOWER=$(echo "$FILTER_DATASET" | tr '[:upper:]' '[:lower:]')
    COUNT=$(mamba run -n PROVE python3 -c "
import json
with open('weights_summary.json') as f:
    data = json.load(f)
missing = [c for c in data if c.get('has_test_results') and not c.get('has_detailed_test_results') and c['dataset'].lower() == '$FILTER_LOWER']
print(len(missing))
" 2>/dev/null)
else
    COUNT=$(mamba run -n PROVE python3 -c "
import json
with open('weights_summary.json') as f:
    data = json.load(f)
missing = [c for c in data if c.get('has_test_results') and not c.get('has_detailed_test_results')]
print(len(missing))
" 2>/dev/null)
fi

echo "Configurations to test: $COUNT"
echo ""

# Estimate time (roughly 2-5 minutes per test)
EST_HOURS=$(python3 -c "print(f'{$COUNT * 3 / 60:.1f}')")
echo "Estimated time: ~$EST_HOURS hours (3 min/test average)"
echo ""

mkdir -p logs

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute:"
    echo "$BSUB_CMD"
else
    echo "Submitting job..."
    echo ""
    eval $BSUB_CMD
    echo ""
    echo "Job submitted. Monitor with: bjobs -w"
    echo "View logs: tail -f logs/${JOB_NAME}_*.log"
fi

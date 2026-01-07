#!/bin/bash
# PROVE - Submit One Job Per Strategy for Detailed Tests
#
# This script submits one batch job for each strategy that needs detailed testing.
# Each job runs all tests for that strategy sequentially.
#
# Benefits:
# - One job per strategy (not flooding the queue)
# - Tests run sequentially within each job
# - Better visibility and failure isolation per strategy
#
# Usage:
#   ./submit_detailed_per_strategy.sh [--dry-run] [--strategy NAME]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Options
DRY_RUN=false
FILTER_STRATEGY=""
QUEUE="BatchGPU"
GPU_MEM="24G"
NUM_CPUS="4"
TIME_LIMIT="24:00"  # 24 hours per strategy should be plenty

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --strategy)
            FILTER_STRATEGY="$2"
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
            echo "Usage: $0 [--strategy NAME] [--queue NAME] [--gpu-mem SIZE] [--time HH:MM] [--dry-run]"
            exit 1
            ;;
    esac
done

echo "PROVE - Submit Detailed Tests (One Job Per Strategy)"
echo "====================================================="
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN"
else
    echo "Mode: LIVE SUBMISSION"
fi
echo "Queue: $QUEUE"
echo "GPU memory: $GPU_MEM"
echo "Time limit: $TIME_LIMIT per strategy"
echo ""

# Analyze configurations
echo "Analyzing configurations..."
mamba run -n prove python weights_analyzer.py --format json >/dev/null 2>&1

# Get strategies with missing detailed tests
STRATEGIES=$(mamba run -n prove python3 << PYTHON_SCRIPT
import json
from collections import defaultdict

with open('weights_summary.json') as f:
    data = json.load(f)

# Filter for configs that have basic tests but no detailed tests
missing = [c for c in data if c.get('has_test_results') and not c.get('has_detailed_test_results')]

# Apply strategy filter if specified
filter_strategy = "${FILTER_STRATEGY}"
if filter_strategy:
    missing = [c for c in missing if c['strategy'] == filter_strategy]

# Group by strategy
by_strategy = defaultdict(int)
for c in missing:
    by_strategy[c['strategy']] += 1

# Output strategies with counts
for strategy in sorted(by_strategy.keys()):
    print(f"{strategy},{by_strategy[strategy]}")
PYTHON_SCRIPT
2>/dev/null)

TOTAL_STRATEGIES=$(echo "$STRATEGIES" | grep -c . || echo "0")
TOTAL_TESTS=$(echo "$STRATEGIES" | awk -F',' '{sum+=$2} END {print sum}')

echo "Found $TOTAL_STRATEGIES strategies with missing detailed tests"
echo "Total tests across all strategies: $TOTAL_TESTS"
echo ""

if [ "$TOTAL_STRATEGIES" -eq 0 ]; then
    echo "No strategies need detailed testing."
    exit 0
fi

# Print summary
echo "Strategies to process:"
echo "$STRATEGIES" | while IFS=',' read -r STRATEGY COUNT; do
    EST_HOURS=$(python3 -c "print(f'{$COUNT * 3 / 60:.1f}')")
    echo "  $STRATEGY: $COUNT tests (~$EST_HOURS hours)"
done
echo ""

mkdir -p logs

# Submit jobs
JOB_COUNT=0
echo "$STRATEGIES" | while IFS=',' read -r STRATEGY COUNT; do
    JOB_COUNT=$((JOB_COUNT + 1))
    
    JOB_NAME="prove_detailed_${STRATEGY}"
    TEST_CMD="$SCRIPT_DIR/run_detailed_tests_strategy.sh --strategy $STRATEGY"
    
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
    
    echo "[$JOB_COUNT/$TOTAL_STRATEGIES] Submitting: $STRATEGY ($COUNT tests)"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $BSUB_CMD"
    else
        eval $BSUB_CMD
        sleep 1  # Small delay between submissions
    fi
done

echo ""
echo "=== Submission Complete ==="
echo "Jobs submitted: $TOTAL_STRATEGIES"
if [ "$DRY_RUN" != true ]; then
    echo ""
    echo "Monitor jobs: bjobs -w"
    echo "View logs: ls logs/prove_detailed_*.log"
fi

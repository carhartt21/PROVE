#!/bin/bash
# Resubmit all MapillaryVistas tests after BGR to RGB fix
# This script submits tests for all MapillaryVistas models in both Stage 1 and Stage 2

set -e

WEIGHTS_STAGE1="/scratch/aaa_exchange/AWARE/WEIGHTS"
WEIGHTS_STAGE2="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Counter for jobs
JOB_COUNT=0
MAX_JOBS=${1:-100}  # Default to 100 jobs, can override with argument

echo "=========================================="
echo "Resubmitting MapillaryVistas Tests"
echo "After BGR to RGB fix in fine_grained_test.py"
echo "=========================================="
echo ""

submit_test() {
    local checkpoint="$1"
    local stage="$2"
    
    if [ $JOB_COUNT -ge $MAX_JOBS ]; then
        echo "Reached max jobs ($MAX_JOBS), stopping..."
        return 1
    fi
    
    # Get the config path
    local config_dir=$(dirname "$checkpoint")
    local config="$config_dir/training_config.py"
    
    if [ ! -f "$config" ]; then
        echo "  SKIP: No config found at $config"
        return 0
    fi
    
    # Get model info
    local model_dir=$(basename "$config_dir")
    local strategy_dir=$(dirname "$config_dir")
    local dataset_dir=$(basename "$strategy_dir")
    strategy_dir=$(dirname "$strategy_dir")
    local strategy=$(basename "$strategy_dir")
    
    echo "  Submitting: $strategy/$dataset_dir/$model_dir (Stage $stage)"
    
    # Output directory
    local output_dir="$config_dir/test_results_detailed"
    
    # Submit test job with 6 hour time limit (MapillaryVistas takes 4-5 hours)
    bsub -J "retest_mv_${strategy}_${model_dir}" \
        -q BatchGPU \
        -gpu "num=1" \
        -R "rusage[mem=16000]" \
        -W 06:00 \
        -o "$REPO_ROOT/logs/retest_mapillary_${strategy}_${model_dir}_%J.log" \
        -e "$REPO_ROOT/logs/retest_mapillary_${strategy}_${model_dir}_%J.err" \
        "cd $REPO_ROOT && python fine_grained_test.py --config '$config' --checkpoint '$checkpoint' --output-dir '$output_dir' --dataset MapillaryVistas"
    
    JOB_COUNT=$((JOB_COUNT + 1))
    sleep 0.5  # Avoid overwhelming the scheduler
}

echo "=== Stage 1 MapillaryVistas Tests ==="
for checkpoint in $(find "$WEIGHTS_STAGE1" -path "*/mapillaryvistas/*/iter_80000.pth" -type f 2>/dev/null); do
    submit_test "$checkpoint" "1" || break
done

echo ""
echo "=== Stage 2 MapillaryVistas Tests ==="
for checkpoint in $(find "$WEIGHTS_STAGE2" -path "*/mapillaryvistas/*/iter_80000.pth" -type f 2>/dev/null); do
    submit_test "$checkpoint" "2" || break
done

echo ""
echo "=========================================="
echo "Submitted $JOB_COUNT test jobs"
echo "=========================================="

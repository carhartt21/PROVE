#!/bin/bash
# PROVE Strategy Domain Adaptation Evaluation
#
# Submit domain adaptation evaluation jobs for augmentation strategies.
# Each job evaluates a strategy's models on Cityscapes (clear_day) + ACDC (adverse weather).
#
# Tests for each strategy run sequentially within a single job to reduce queue pressure.
#
# Usage:
#   ./scripts/submit_strategy_domain_adaptation.sh --strategy gen_NST
#   ./scripts/submit_strategy_domain_adaptation.sh --all
#   ./scripts/submit_strategy_domain_adaptation.sh --list

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

set +e  # Disable exit on error for the main loop

# ============================================================================
# Configuration
# ============================================================================

# Weights directory
WEIGHTS_ROOT="${PROVE_WEIGHTS_ROOT:-/scratch/aaa_exchange/AWARE/WEIGHTS}"

# Source datasets to evaluate
SOURCE_DATASETS=("bdd10k" "idd-aw" "mapillaryvistas")

# Models to evaluate (base models)
BASE_MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

# Model variants - empty string for full, _clear_day for clear-day trained
MODEL_VARIANTS=("" "_clear_day")

# Default LSF settings
DEFAULT_QUEUE="BatchGPU"
DEFAULT_GPU_MEM="16G"
DEFAULT_GPU_MODE="shared"
DEFAULT_NUM_CPUS=4
DEFAULT_TIME="24:00"  # Max 24 hours per strategy

# Python environment
CONDA_ENV="prove"

# Strategies to exclude (baselines are evaluated separately)
EXCLUDE_STRATEGIES=("baseline" "WEIGHTS_COMBINATIONS" "domain_adaptation_ablation")

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Strategy Domain Adaptation Evaluation"
    echo "============================================"
    echo ""
    echo "Evaluates augmentation strategies on domain adaptation task:"
    echo "  Target: Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy)"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --strategy <name>   Specific strategy to evaluate (e.g., gen_NST)"
    echo "  --all               Submit jobs for all strategies"
    echo "  --list              List available strategies"
    echo "  --dry-run           Show commands without executing"
    echo "  --include-clear-day Include _clear_day model variants (default: false)"
    echo "  --queue <name>      LSF queue name (default: $DEFAULT_QUEUE)"
    echo "  --gpu-mem <size>    GPU memory (default: $DEFAULT_GPU_MEM)"
    echo "  --time <hh:mm>      Max runtime (default: $DEFAULT_TIME)"
    echo ""
    echo "Examples:"
    echo "  $0 --strategy gen_NST"
    echo "  $0 --all --dry-run"
    echo "  $0 --list"
    echo ""
}

# Get list of all strategies from WEIGHTS directory
get_all_strategies() {
    local strategies=()
    
    for dir in "${WEIGHTS_ROOT}"/*/; do
        local name=$(basename "$dir")
        
        # Skip excluded directories
        local skip=false
        for exclude in "${EXCLUDE_STRATEGIES[@]}"; do
            if [[ "$name" == "$exclude" ]]; then
                skip=true
                break
            fi
        done
        
        # Skip non-strategy directories (those without model checkpoints)
        if [[ "$skip" == "false" ]] && [[ -d "$dir/bdd10k" || -d "$dir/idd-aw" || -d "$dir/mapillaryvistas" ]]; then
            strategies+=("$name")
        fi
    done
    
    printf '%s\n' "${strategies[@]}" | sort
}

# Check if strategy has valid checkpoints
check_strategy_checkpoints() {
    local strategy="$1"
    local strategy_dir="${WEIGHTS_ROOT}/${strategy}"
    local found=0
    
    for dataset in "${SOURCE_DATASETS[@]}"; do
        for model in "${BASE_MODELS[@]}"; do
            local ckpt_path="${strategy_dir}/${dataset}/${model}/iter_80000.pth"
            if [[ -f "$ckpt_path" ]]; then
                ((found++))
            fi
        done
    done
    
    echo "$found"
}

# Count total model checkpoints for a strategy
count_strategy_models() {
    local strategy="$1"
    local include_clearday="$2"
    local strategy_dir="${WEIGHTS_ROOT}/${strategy}"
    local count=0
    
    for dataset in "${SOURCE_DATASETS[@]}"; do
        for model in "${BASE_MODELS[@]}"; do
            # Check full model
            if [[ -f "${strategy_dir}/${dataset}/${model}/iter_80000.pth" ]]; then
                ((count++))
            fi
            
            # Check clear_day variant
            if [[ "$include_clearday" == "true" ]]; then
                if [[ -f "${strategy_dir}/${dataset}/${model}_clear_day/iter_80000.pth" ]]; then
                    ((count++))
                fi
            fi
        done
    done
    
    echo "$count"
}

# List all available strategies with checkpoint counts
list_strategies() {
    echo "Available Strategies for Domain Adaptation Evaluation"
    echo "======================================================"
    echo ""
    echo "Strategy                        | Full Models | With Clear Day"
    echo "--------------------------------|-------------|---------------"
    
    local strategies=($(get_all_strategies))
    
    for strategy in "${strategies[@]}"; do
        local full_count=$(count_strategy_models "$strategy" "false")
        local total_count=$(count_strategy_models "$strategy" "true")
        printf "%-31s | %11d | %14d\n" "$strategy" "$full_count" "$total_count"
    done
    
    echo ""
    echo "Total strategies: ${#strategies[@]}"
}

# Submit evaluation job for a strategy
submit_strategy_job() {
    local strategy="$1"
    local include_clearday="$2"
    local queue="$3"
    local gpu_mem="$4"
    local gpu_mode="$5"
    local num_cpus="$6"
    local max_time="$7"
    local dry_run="$8"
    
    local strategy_dir="${WEIGHTS_ROOT}/${strategy}"
    
    # Check if strategy directory exists
    if [[ ! -d "$strategy_dir" ]]; then
        echo "  SKIP: Strategy directory not found: $strategy_dir"
        return 1
    fi
    
    # Count available checkpoints
    local model_count=$(count_strategy_models "$strategy" "$include_clearday")
    if [[ "$model_count" -eq 0 ]]; then
        echo "  SKIP: No checkpoints found for $strategy"
        return 1
    fi
    
    # Job configuration
    local jobname="da_eval_${strategy}"
    local log_dir="${PROJECT_ROOT}/logs/domain_adaptation_strategies"
    mkdir -p "$log_dir"
    
    # Determine which variants to evaluate
    local variants=("\"\"")  # Always include full model
    if [[ "$include_clearday" == "true" ]]; then
        variants+=("\"_clear_day\"")
    fi
    local variants_str=$(IFS=,; echo "${variants[*]}")
    
    # Build Python evaluation script
    # This runs sequentially through all models for the strategy
    local python_script="
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from tools.evaluate_domain_adaptation import run_evaluation, get_checkpoint_path

STRATEGY = '${strategy}'
WEIGHTS_ROOT = '${WEIGHTS_ROOT}'
SOURCE_DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas']
BASE_MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
VARIANTS = [${variants_str}]

print(f'\\n{\"=\"*70}')
print(f'Strategy Domain Adaptation Evaluation: {STRATEGY}')
print(f'{\"=\"*70}')
print(f'Evaluating on Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy)')
print(f'{\"=\"*70}\\n')

results_count = 0
errors = []

for source in SOURCE_DATASETS:
    for model in BASE_MODELS:
        for variant in VARIANTS:
            full_model = model + variant
            
            # Build checkpoint path manually for strategy
            from pathlib import Path
            ckpt_dir = Path(WEIGHTS_ROOT) / STRATEGY / source.lower() / full_model
            ckpt_path = ckpt_dir / 'iter_80000.pth'
            
            if not ckpt_path.exists():
                print(f'SKIP: No checkpoint for {STRATEGY}/{source}/{full_model}')
                continue
            
            print(f'\\n{\"=\"*70}')
            print(f'Evaluating: {STRATEGY} / {source} / {full_model}')
            print(f'{\"=\"*70}')
            
            try:
                result = run_evaluation(
                    source_dataset=source,
                    model=model,
                    checkpoint_path=str(ckpt_path),
                    variant=variant
                )
                
                if result:
                    results_count += 1
                    # Save result to strategy-specific location
                    output_dir = Path('${WEIGHTS_ROOT}') / 'domain_adaptation_ablation' / STRATEGY / source.lower() / full_model
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    import json
                    result_file = output_dir / 'domain_adaptation_evaluation.json'
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f'Saved to: {result_file}')
                    
            except Exception as e:
                print(f'ERROR: {e}')
                errors.append(f'{source}/{full_model}: {e}')

print(f'\\n{\"=\"*70}')
print(f'EVALUATION COMPLETE')
print(f'{\"=\"*70}')
print(f'Strategy: {STRATEGY}')
print(f'Successful evaluations: {results_count}')
print(f'Errors: {len(errors)}')
if errors:
    print('Error details:')
    for err in errors:
        print(f'  - {err}')
"

    # LSF submission command
    local submit_cmd="bsub -gpu \"num=1:mode=${gpu_mode}:gmem=${gpu_mem}\" \
        -q ${queue} \
        -W ${max_time} \
        -R \"span[hosts=1]\" \
        -n ${num_cpus} \
        -oo \"${log_dir}/${jobname}_%J.log\" \
        -eo \"${log_dir}/${jobname}_%J.err\" \
        -L /bin/bash \
        -J \"${jobname}\" \
        \"conda activate ${CONDA_ENV}; python -c '${python_script}'\""
    
    if [[ "$dry_run" == "true" ]]; then
        echo "  [DRY-RUN] Would submit: $jobname"
        echo "    Strategy: $strategy"
        echo "    Models: $model_count"
        echo "    Include clear_day: $include_clearday"
        echo ""
    else
        echo "  Submitting: $jobname"
        echo "    Strategy: $strategy"
        echo "    Models: $model_count"
        eval "$submit_cmd"
    fi
    
    return 0
}

# ============================================================================
# Parse Arguments
# ============================================================================

STRATEGY=""
ALL_MODE=false
LIST_MODE=false
DRY_RUN=false
INCLUDE_CLEARDAY=false
QUEUE="$DEFAULT_QUEUE"
GPU_MEM="$DEFAULT_GPU_MEM"
GPU_MODE="$DEFAULT_GPU_MODE"
NUM_CPUS="$DEFAULT_NUM_CPUS"
MAX_TIME="$DEFAULT_TIME"

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --all)
            ALL_MODE=true
            shift
            ;;
        --list)
            LIST_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --include-clear-day)
            INCLUDE_CLEARDAY=true
            shift
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
            MAX_TIME="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Execution
# ============================================================================

if [[ "$LIST_MODE" == "true" ]]; then
    list_strategies
    exit 0
fi

if [[ "$ALL_MODE" == "false" ]] && [[ -z "$STRATEGY" ]]; then
    print_usage
    echo "ERROR: Specify --strategy <name> or --all"
    exit 1
fi

# Determine strategies to process
if [[ "$ALL_MODE" == "true" ]]; then
    mapfile -t STRATEGIES < <(get_all_strategies)
else
    STRATEGIES=("$STRATEGY")
fi

echo "========================================================================"
echo "PROVE Strategy Domain Adaptation Evaluation"
echo "========================================================================"
echo ""
echo "Target: Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy)"
echo "Include clear_day variants: $INCLUDE_CLEARDAY"
echo "Queue: $QUEUE"
echo "GPU Memory: $GPU_MEM"
echo "Max Time: $MAX_TIME"
echo "Dry Run: $DRY_RUN"
echo ""
echo "Strategies to evaluate: ${#STRATEGIES[@]}"
echo ""

submitted=0
skipped=0

for strategy in "${STRATEGIES[@]}"; do
    echo "Processing: $strategy"
    
    if submit_strategy_job "$strategy" "$INCLUDE_CLEARDAY" "$QUEUE" "$GPU_MEM" "$GPU_MODE" "$NUM_CPUS" "$MAX_TIME" "$DRY_RUN"; then
        ((submitted++))
    else
        ((skipped++))
    fi
done

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "  Submitted: $submitted"
echo "  Skipped: $skipped"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "This was a DRY RUN. No jobs were actually submitted."
    echo "Remove --dry-run to submit jobs."
fi

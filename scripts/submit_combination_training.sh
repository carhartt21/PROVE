#!/bin/bash
# =============================================================================
# PROVE - Combination Strategy Ablation Study Submission Script
# =============================================================================
#
# This script submits training jobs for the combination ablation study,
# which investigates whether combining multiple augmentation strategies
# provides synergistic benefits over using individual strategies alone.
#
# Combination Types:
#   1. Generative + Standard: gen_* + std_* (e.g., gen_cycleGAN+std_mixup)
#   2. Standard + Standard: std_* + std_* (e.g., std_randaugment+std_cutmix)
#
# Output directory: /scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATIONS_chge7185/
#
# Usage:
#   ./scripts/submit_combination_training.sh --list          # List all combinations
#   ./scripts/submit_combination_training.sh --dry-run       # Preview without submitting
#   ./scripts/submit_combination_training.sh                 # Submit all jobs
#   ./scripts/submit_combination_training.sh --limit 10      # Submit first 10 jobs
#
# See docs/COMBINATION_ABLATION.md for detailed documentation.
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

# Generative strategies for combination study
GEN_STRATEGIES=(
    "gen_step1x_new"
    "gen_flux_kontext"
    "gen_Qwen_Image_Edit"
    "gen_stargan_v2"
    "gen_Attribute_Hallucination"
)

# Standard strategies for combination study
STD_STRATEGIES=(
    "std_randaugment"
    "std_mixup"
    "std_cutmix"
    "std_autoaugment"
    "photometric_distort"
)

# Standard-only combinations (all pairs of std strategies)
STD_COMBINATIONS=(
    "std_randaugment+std_mixup"
    "std_randaugment+std_cutmix"
    "std_randaugment+std_autoaugment"
    "std_randaugment+photometric_distort"
    "std_mixup+std_cutmix"
    "std_mixup+std_autoaugment"
    "std_mixup+photometric_distort"
    "std_cutmix+std_autoaugment"
    "std_cutmix+photometric_distort"
    "std_autoaugment+photometric_distort"
)

# Datasets for combination ablation
DATASETS=(
    "MapillaryVistas"
    "IDD-AW"
)

# Models for combination ablation
MODELS=(
    "segformer_mit-b5"
    "pspnet_r50"
)

# LSF Configuration
LSF_QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=10
WALL_TIME="48:00"

# Output directory
WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATIONS_chge7185"
LOG_DIR="${PROJECT_ROOT}/logs"

# Real-to-generated ratio for generative strategies
RATIO=0.5

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Combination Strategy Ablation Study Submission"
    echo "===================================================="
    echo ""
    echo "This script submits training jobs for combination ablation study."
    echo "Combinations of generative+standard and standard+standard strategies."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  --list            List all combinations that would be submitted"
    echo "  --dry-run         Show bsub commands without submitting"
    echo "  --limit N         Submit only first N jobs"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Options:"
    echo "  --gen-only        Submit only generative+standard combinations"
    echo "  --std-only        Submit only standard+standard combinations"
    echo "  --dataset NAME    Submit only for specific dataset"
    echo "  --model NAME      Submit only for specific model"
    echo "  --ratio RATIO     Real-to-generated ratio (default: 0.5)"
    echo "  --queue NAME      LSF queue (default: BatchGPU)"
    echo "  --gpu-mem SIZE    GPU memory (default: 24G)"
    echo "  --delay N         Delay in seconds between submissions (default: 1)"
    echo "  --no-test         Skip automatic testing after training"
    echo ""
    echo "Examples:"
    echo "  $0 --list"
    echo "  $0 --dry-run"
    echo "  $0 --limit 10"
    echo "  $0 --gen-only --dataset MapillaryVistas"
    echo "  $0 --std-only --model segformer_mit-b5"
    echo "  $0 --no-test     # Training only, no testing"
    echo ""
    echo "Default Configuration:"
    echo "  Gen strategies: ${GEN_STRATEGIES[*]}"
    echo "  Std strategies: ${STD_STRATEGIES[*]}"
    echo "  Datasets: ${DATASETS[*]}"
    echo "  Models: ${MODELS[*]}"
    local gen_std_count=$((${#GEN_STRATEGIES[@]} * ${#STD_STRATEGIES[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))
    local std_std_count=$((${#STD_COMBINATIONS[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))
    echo "  Total gen+std: ${gen_std_count}"
    echo "  Total std+std: ${std_std_count}"
    echo "  Total: $((gen_std_count + std_std_count))"
}

# ============================================================================
# Helper Functions
# ============================================================================

submit_gen_std_job() {
    local gen_strategy=$1
    local std_strategy=$2
    local dataset=$3
    local model=$4
    local dry_run=$5
    local skip_test=$6
    
    local combo_name="${gen_strategy}+${std_strategy}"
    local ratio_str="ratio0p$(echo "$RATIO" | sed 's/0\.//')"
    local model_suffix="${model}_${ratio_str}"
    local dataset_lower="${dataset,,}"
    
    # Replace IDD-AW with idd-aw
    dataset_lower="${dataset_lower//idd-aw/idd-aw}"
    
    local jobname="combo_${gen_strategy}_${std_strategy}_${dataset}_${model}"
    
    # Construct explicit work directory path
    local work_dir="${WEIGHTS_DIR}/${combo_name}/${dataset_lower}/${model_suffix}"
    
    # Check if already completed
    local checkpoint="${work_dir}/iter_80000.pth"
    if [[ -f "$checkpoint" ]]; then
        echo "[SKIP] Already exists: ${combo_name}/${dataset_lower}/${model_suffix}"
        return 0
    fi
    
    # Build training command with explicit work-dir
    local train_cmd="cd ${PROJECT_ROOT} && source ~/.bashrc && mamba activate prove && python unified_training.py"
    train_cmd="${train_cmd} --dataset ${dataset}"
    train_cmd="${train_cmd} --model ${model}"
    train_cmd="${train_cmd} --strategy ${gen_strategy}"
    train_cmd="${train_cmd} --std-strategy ${std_strategy}"
    train_cmd="${train_cmd} --real-gen-ratio ${RATIO}"
    train_cmd="${train_cmd} --work-dir ${work_dir}"
    
    # Build test command (run after training)
    local test_cmd=""
    if [[ "$skip_test" != "true" ]]; then
        test_cmd=" && python fine_grained_test.py"
        test_cmd="${test_cmd} --config ${work_dir}/training_config.py"
        test_cmd="${test_cmd} --checkpoint ${work_dir}/iter_80000.pth"
        test_cmd="${test_cmd} --dataset ${dataset}"
        test_cmd="${test_cmd} --output-dir ${work_dir}/test_results_detailed"
    fi
    
    # Combine commands
    local full_cmd="${train_cmd}${test_cmd}"
    
    # Build bsub command
    local bsub_cmd="bsub -J \"${jobname}\" \
        -q ${LSF_QUEUE} \
        -gpu \"num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}\" \
        -n ${NUM_CPUS} \
        -W ${WALL_TIME} \
        -R \"span[hosts=1]\" \
        -oo \"${LOG_DIR}/${jobname}_%J.out\" \
        -eo \"${LOG_DIR}/${jobname}_%J.err\" \
        \"${full_cmd}\""
    
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] ${combo_name} | ${dataset} | ${model}"
        echo "  Work dir: ${work_dir}"
        echo "  Train: ${train_cmd}"
        if [[ -n "$test_cmd" ]]; then
            echo "  Test:  python fine_grained_test.py --config ${work_dir}/training_config.py --checkpoint ${work_dir}/iter_80000.pth --dataset ${dataset} --output-dir ${work_dir}/test_results_detailed"
        fi
        echo ""
    else
        echo "Submitting: ${combo_name} | ${dataset} | ${model}"
        eval $bsub_cmd
    fi
}

submit_std_std_job() {
    local std_combo=$1
    local dataset=$2
    local model=$3
    local dry_run=$4
    local skip_test=$5
    
    # Parse the combination (e.g., std_randaugment+std_mixup)
    local std1="${std_combo%+*}"
    local std2="${std_combo#*+}"
    
    local dataset_lower="${dataset,,}"
    dataset_lower="${dataset_lower//idd-aw/idd-aw}"
    
    local jobname="combo_${std_combo}_${dataset}_${model}"
    jobname="${jobname//+/_}"  # Replace + with _ for job name
    
    # Construct explicit work directory path (no ratio suffix for std+std)
    local work_dir="${WEIGHTS_DIR}/${std_combo}/${dataset_lower}/${model}"
    
    # Check if already completed
    local checkpoint="${work_dir}/iter_80000.pth"
    if [[ -f "$checkpoint" ]]; then
        echo "[SKIP] Already exists: ${std_combo}/${dataset_lower}/${model}"
        return 0
    fi
    
    # Build training command with explicit work-dir
    # First std strategy as main, second as --std-strategy
    local train_cmd="cd ${PROJECT_ROOT} && source ~/.bashrc && mamba activate prove && python unified_training.py"
    train_cmd="${train_cmd} --dataset ${dataset}"
    train_cmd="${train_cmd} --model ${model}"
    train_cmd="${train_cmd} --strategy ${std1}"
    train_cmd="${train_cmd} --std-strategy ${std2}"
    train_cmd="${train_cmd} --work-dir ${work_dir}"
    
    # Build test command (run after training)
    local test_cmd=""
    if [[ "$skip_test" != "true" ]]; then
        test_cmd=" && python fine_grained_test.py"
        test_cmd="${test_cmd} --config ${work_dir}/training_config.py"
        test_cmd="${test_cmd} --checkpoint ${work_dir}/iter_80000.pth"
        test_cmd="${test_cmd} --dataset ${dataset}"
        test_cmd="${test_cmd} --output-dir ${work_dir}/test_results_detailed"
    fi
    
    # Combine commands
    local full_cmd="${train_cmd}${test_cmd}"
    
    # Build bsub command
    local bsub_cmd="bsub -J \"${jobname}\" \
        -q ${LSF_QUEUE} \
        -gpu \"num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}\" \
        -n ${NUM_CPUS} \
        -W ${WALL_TIME} \
        -R \"span[hosts=1]\" \
        -oo \"${LOG_DIR}/${jobname}_%J.out\" \
        -eo \"${LOG_DIR}/${jobname}_%J.err\" \
        \"${full_cmd}\""
    
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] ${std_combo} | ${dataset} | ${model}"
        echo "  Work dir: ${work_dir}"
        echo "  Train: ${train_cmd}"
        if [[ -n "$test_cmd" ]]; then
            echo "  Test:  python fine_grained_test.py --config ${work_dir}/training_config.py --checkpoint ${work_dir}/iter_80000.pth --dataset ${dataset} --output-dir ${work_dir}/test_results_detailed"
        fi
        echo ""
    else
        echo "Submitting: ${std_combo} | ${dataset} | ${model}"
        eval $bsub_cmd
    fi
}

# ============================================================================
# Main Commands
# ============================================================================

list_combinations() {
    echo "=============================================="
    echo "Combination Ablation Study - All Combinations"
    echo "=============================================="
    echo ""
    
    local count=0
    
    echo "=== Generative + Standard Combinations ==="
    echo "Gen strategies: ${GEN_STRATEGIES[*]}"
    echo "Std strategies: ${STD_STRATEGIES[*]}"
    echo ""
    
    for gen in "${GEN_STRATEGIES[@]}"; do
        for std in "${STD_STRATEGIES[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for model in "${MODELS[@]}"; do
                    echo "  ${gen}+${std} | ${dataset} | ${model}"
                    count=$((count + 1))
                done
            done
        done
    done
    
    echo ""
    echo "=== Standard + Standard Combinations ==="
    echo "Combinations: ${STD_COMBINATIONS[*]}"
    echo ""
    
    for combo in "${STD_COMBINATIONS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do
                echo "  ${combo} | ${dataset} | ${model}"
                count=$((count + 1))
            done
        done
    done
    
    echo ""
    echo "=============================================="
    echo "Total combinations: ${count}"
    echo "=============================================="
}

submit_all() {
    local dry_run=$1
    local limit=$2
    local gen_only=$3
    local std_only=$4
    local filter_dataset=$5
    local filter_model=$6
    local delay=$7
    local skip_test=$8
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    local count=0
    local submitted=0
    
    echo "=============================================="
    echo "Combination Ablation Study - Job Submission"
    echo "=============================================="
    echo "Dry run: ${dry_run}"
    echo "Auto-test after training: $([[ "$skip_test" == "true" ]] && echo "No" || echo "Yes")"
    [[ -n "$limit" ]] && echo "Limit: ${limit}"
    [[ -n "$filter_dataset" ]] && echo "Dataset filter: ${filter_dataset}"
    [[ -n "$filter_model" ]] && echo "Model filter: ${filter_model}"
    echo "Output directory: ${WEIGHTS_DIR}"
    echo "=============================================="
    echo ""
    
    # Submit gen+std combinations
    if [[ "$std_only" != "true" ]]; then
        echo "=== Submitting Generative + Standard Combinations ==="
        echo ""
        
        for gen in "${GEN_STRATEGIES[@]}"; do
            for std in "${STD_STRATEGIES[@]}"; do
                for dataset in "${DATASETS[@]}"; do
                    # Apply dataset filter
                    if [[ -n "$filter_dataset" ]] && [[ "$dataset" != "$filter_dataset" ]]; then
                        continue
                    fi
                    
                    for model in "${MODELS[@]}"; do
                        # Apply model filter
                        if [[ -n "$filter_model" ]] && [[ "$model" != "$filter_model" ]]; then
                            continue
                        fi
                        
                        # Check limit
                        if [[ -n "$limit" ]] && [[ $submitted -ge $limit ]]; then
                            echo ""
                            echo "Reached limit of ${limit} submissions"
                            break 4
                        fi
                        
                        submit_gen_std_job "$gen" "$std" "$dataset" "$model" "$dry_run" "$skip_test"
                        submitted=$((submitted + 1))
                        count=$((count + 1))
                        
                        # Add delay between submissions
                        if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]] && [[ $delay -gt 0 ]]; then
                            sleep "$delay"
                        fi
                    done
                done
            done
        done
    fi
    
    # Check if we've reached the limit
    if [[ -n "$limit" ]] && [[ $submitted -ge $limit ]]; then
        echo ""
        echo "=============================================="
        echo "Submitted ${submitted} jobs (limit reached)"
        echo "=============================================="
        return
    fi
    
    # Submit std+std combinations
    if [[ "$gen_only" != "true" ]]; then
        echo ""
        echo "=== Submitting Standard + Standard Combinations ==="
        echo ""
        
        for combo in "${STD_COMBINATIONS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                # Apply dataset filter
                if [[ -n "$filter_dataset" ]] && [[ "$dataset" != "$filter_dataset" ]]; then
                    continue
                fi
                
                for model in "${MODELS[@]}"; do
                    # Apply model filter
                    if [[ -n "$filter_model" ]] && [[ "$model" != "$filter_model" ]]; then
                        continue
                    fi
                    
                    # Check limit
                    if [[ -n "$limit" ]] && [[ $submitted -ge $limit ]]; then
                        echo ""
                        echo "Reached limit of ${limit} submissions"
                        break 3
                    fi
                    
                    submit_std_std_job "$combo" "$dataset" "$model" "$dry_run" "$skip_test"
                    submitted=$((submitted + 1))
                    count=$((count + 1))
                    
                    # Add delay between submissions
                    if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]] && [[ $delay -gt 0 ]]; then
                        sleep "$delay"
                    fi
                done
            done
        done
    fi
    
    echo ""
    echo "=============================================="
    if [[ "$dry_run" == "true" ]]; then
        echo "Would submit ${submitted} jobs"
    else
        echo "Submitted ${submitted} jobs"
    fi
    echo "=============================================="
}

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN="false"
LIMIT=""
LIST_ONLY="false"
GEN_ONLY="false"
STD_ONLY="false"
FILTER_DATASET=""
FILTER_MODEL=""
DELAY=1
SKIP_TEST="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --list)
            LIST_ONLY="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --gen-only)
            GEN_ONLY="true"
            shift
            ;;
        --std-only)
            STD_ONLY="true"
            shift
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --queue)
            LSF_QUEUE="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --no-test)
            SKIP_TEST="true"
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Execute
# ============================================================================

if [[ "$LIST_ONLY" == "true" ]]; then
    list_combinations
else
    submit_all "$DRY_RUN" "$LIMIT" "$GEN_ONLY" "$STD_ONLY" "$FILTER_DATASET" "$FILTER_MODEL" "$DELAY" "$SKIP_TEST"
fi

#!/bin/bash
# PROVE - Submit Standard Augmentation Combination Training Jobs to LSF
#
# This script submits batch training jobs that combine multiple standard
# augmentation strategies (std_* + std_*) to explore synergies between them.
#
# Standard Augmentations:
#   - std_randaugment: RandAugment augmentation policy
#   - std_mixup: MixUp augmentation (blends images)
#   - std_cutmix: CutMix augmentation (cuts and pastes patches)
#   - std_autoaugment: AutoAugment learned augmentation policy
#
# Example combinations:
#   - std_mixup + std_cutmix
#   - std_randaugment + std_mixup
#   - etc.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration - Customize these based on your needs
# ============================================================================

# All standard augmentation strategies
STD_STRATEGIES=(
    "std_randaugment"
    "std_mixup"
    "std_cutmix"
    "std_autoaugment"
)

# Datasets to train on
DATASETS=(
    "ACDC"
    "BDD10k"
    "IDD-AW"
    "MapillaryVistas"
    "OUTSIDE15k"
)

# Models to train
MODELS=(
    "deeplabv3plus_r50"
    "pspnet_r50"
    "segformer_mit-b5"
)

# LSF Configuration
LSF_QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=8

# Domain variants for training (empty string = no filter, clear_day = clear_day only)
DOMAIN_VARIANTS=("" "clear_day")

# ============================================================================
# Generate all unique 2-way combinations of std strategies
# ============================================================================

generate_std_combinations() {
    local combinations=()
    local n=${#STD_STRATEGIES[@]}
    
    for ((i=0; i<n; i++)); do
        for ((j=i+1; j<n; j++)); do
            combinations+=("${STD_STRATEGIES[$i]}+${STD_STRATEGIES[$j]}")
        done
    done
    
    echo "${combinations[@]}"
}

# Get all std+std combinations
STD_COMBINATIONS=($(generate_std_combinations))

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Standard Augmentation Combination Batch Submission"
    echo "========================================================="
    echo ""
    echo "This script submits training jobs for std+std combinations"
    echo "to evaluate synergies between standard augmentation strategies."
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  submit-all      Submit all std+std combinations for all datasets/models"
    echo "  submit-single   Submit single combination (requires --std1 and --std2)"
    echo "  submit-dataset  Submit all combinations for one dataset"
    echo "  submit-model    Submit all combinations for one model"
    echo "  submit-combo    Submit all models/datasets for one std+std combination"
    echo "  list            List all combinations that would be submitted"
    echo "  estimate        Estimate total jobs and resources"
    echo "  help            Show this help"
    echo ""
    echo "Options:"
    echo "  --std1 <strategy>         First std_* strategy"
    echo "  --std2 <strategy>         Second std_* strategy"
    echo "  --dataset <name>          Specific dataset"
    echo "  --model <name>            Specific model"
    echo "  --queue <name>            LSF queue (default: BatchGPU)"
    echo "  --gpu-mem <size>          GPU memory (default: 24G)"
    echo "  --gpu-mode <mode>         GPU mode: shared/exclusive_process (default: shared)"
    echo "  --dry-run                 Show commands without executing"
    echo "  --delay <seconds>         Delay between job submissions (default: 1)"
    echo "  --domain-filter <domain>  Add domain filter (e.g., clear_day)"
    echo "  --with-domain-variants    Submit both regular AND clear_day variants"
    echo ""
    echo "Std+Std Combinations (${#STD_COMBINATIONS[@]} total):"
    for combo in "${STD_COMBINATIONS[@]}"; do
        echo "  - $combo"
    done
    echo ""
    echo "Examples:"
    echo "  $0 submit-all --dry-run"
    echo "  $0 submit-all --with-domain-variants"
    echo "  $0 submit-single --std1 std_mixup --std2 std_cutmix --dataset ACDC --model deeplabv3plus_r50"
    echo "  $0 submit-dataset --dataset ACDC"
    echo "  $0 submit-model --model segformer_mit-b5"
    echo "  $0 submit-combo --std1 std_mixup --std2 std_cutmix"
    echo "  $0 list"
    echo "  $0 estimate"
}

submit_job() {
    local dataset=$1
    local model=$2
    local std1=$3
    local std2=$4
    local dry_run=$5
    local domain_filter=$6
    
    # Job name: prove_ACDC_deeplabv3plus_r50_std_mixup+std_cutmix
    local jobname="prove_${dataset}_${model}_${std1}+${std2}"
    if [[ -n "$domain_filter" ]]; then
        jobname="${jobname}_${domain_filter}"
    fi
    
    # Build the training command
    # Use first std as main strategy, second std as --std-strategy
    local train_cmd="$SCRIPT_DIR/train_unified.sh single --dataset ${dataset} --model ${model} --strategy ${std1} --std-strategy ${std2}"
    if [[ -n "$domain_filter" ]]; then
        train_cmd="${train_cmd} --domain-filter ${domain_filter}"
    fi
    
    # Build bsub command
    local bsub_cmd="bsub -gpu \"num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}\" \
        -q ${LSF_QUEUE} \
        -R \"span[hosts=1]\" \
        -n ${NUM_CPUS} \
        -oo \"logs/${jobname}_%J.out\" \
        -eo \"logs/${jobname}_%J.err\" \
        -L /bin/bash \
        -J \"${jobname}\" \
        \"${train_cmd}\""
    
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] $bsub_cmd"
    else
        echo "Submitting: ${jobname}"
        eval $bsub_cmd
    fi
}

# ============================================================================
# Commands
# ============================================================================

cmd_submit_all() {
    local dry_run=$1
    local delay=$2
    local domain_filter=$3
    local with_domain_variants=$4
    
    # Determine domain variants to process
    local domain_variants=("")
    if [[ "$with_domain_variants" == "true" ]]; then
        domain_variants=("" "clear_day")
    elif [[ -n "$domain_filter" ]]; then
        domain_variants=("$domain_filter")
    fi
    
    echo "=============================================="
    echo "Submitting ALL std+std combinations"
    echo "=============================================="
    echo "Std combinations: ${#STD_COMBINATIONS[@]}"
    for combo in "${STD_COMBINATIONS[@]}"; do
        echo "  - $combo"
    done
    echo "Datasets: ${#DATASETS[@]}"
    echo "Models: ${#MODELS[@]}"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    local total_jobs=$((${#STD_COMBINATIONS[@]} * ${#DATASETS[@]} * ${#MODELS[@]} * ${#domain_variants[@]}))
    echo "Total jobs: $total_jobs"
    echo "=============================================="
    
    # Create logs directory
    mkdir -p logs
    
    local count=0
    for combo in "${STD_COMBINATIONS[@]}"; do
        # Split combo into std1 and std2
        local std1="${combo%+*}"
        local std2="${combo#*+}"
        
        for dataset in "${DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do
                for variant_filter in "${domain_variants[@]}"; do
                    submit_job "$dataset" "$model" "$std1" "$std2" "$dry_run" "$variant_filter"
                    count=$((count + 1))
                    if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]]; then
                        sleep "$delay"
                    fi
                done
            done
        done
    done
    
    echo ""
    echo "=============================================="
    echo "Submitted $count jobs"
    echo "=============================================="
}

cmd_submit_single() {
    local std1=$1
    local std2=$2
    local dataset=$3
    local model=$4
    local dry_run=$5
    local domain_filter=$6
    
    if [[ -z "$std1" ]] || [[ -z "$std2" ]] || [[ -z "$dataset" ]] || [[ -z "$model" ]]; then
        echo "Error: --std1, --std2, --dataset, and --model are required for submit-single"
        echo ""
        print_usage
        exit 1
    fi
    
    echo "=============================================="
    echo "Submitting single std+std combination"
    echo "=============================================="
    echo "Combination: ${std1}+${std2}"
    echo "Dataset: $dataset"
    echo "Model: $model"
    if [[ -n "$domain_filter" ]]; then
        echo "Domain filter: $domain_filter"
    fi
    echo "=============================================="
    
    mkdir -p logs
    submit_job "$dataset" "$model" "$std1" "$std2" "$dry_run" "$domain_filter"
}

cmd_submit_dataset() {
    local dataset=$1
    local dry_run=$2
    local domain_filter=$3
    local with_domain_variants=$4
    
    if [[ -z "$dataset" ]]; then
        echo "Error: --dataset is required for submit-dataset"
        exit 1
    fi
    
    # Determine domain variants to process
    local domain_variants=("")
    if [[ "$with_domain_variants" == "true" ]]; then
        domain_variants=("" "clear_day")
    elif [[ -n "$domain_filter" ]]; then
        domain_variants=("$domain_filter")
    fi
    
    local total_jobs=$((${#STD_COMBINATIONS[@]} * ${#MODELS[@]} * ${#domain_variants[@]}))
    
    echo "=============================================="
    echo "Submitting all std+std combinations for dataset: $dataset"
    echo "=============================================="
    echo "Std combinations: ${#STD_COMBINATIONS[@]}"
    echo "Models: ${#MODELS[@]}"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    echo "Total jobs: $total_jobs"
    echo "=============================================="
    
    mkdir -p logs
    
    local count=0
    for combo in "${STD_COMBINATIONS[@]}"; do
        local std1="${combo%+*}"
        local std2="${combo#*+}"
        
        for model in "${MODELS[@]}"; do
            for variant_filter in "${domain_variants[@]}"; do
                submit_job "$dataset" "$model" "$std1" "$std2" "$dry_run" "$variant_filter"
                count=$((count + 1))
            done
        done
    done
    
    echo ""
    echo "Submitted $count jobs for dataset: $dataset"
}

cmd_submit_model() {
    local model=$1
    local dry_run=$2
    local domain_filter=$3
    local with_domain_variants=$4
    
    if [[ -z "$model" ]]; then
        echo "Error: --model is required for submit-model"
        exit 1
    fi
    
    # Determine domain variants to process
    local domain_variants=("")
    if [[ "$with_domain_variants" == "true" ]]; then
        domain_variants=("" "clear_day")
    elif [[ -n "$domain_filter" ]]; then
        domain_variants=("$domain_filter")
    fi
    
    local total_jobs=$((${#STD_COMBINATIONS[@]} * ${#DATASETS[@]} * ${#domain_variants[@]}))
    
    echo "=============================================="
    echo "Submitting all std+std combinations for model: $model"
    echo "=============================================="
    echo "Std combinations: ${#STD_COMBINATIONS[@]}"
    echo "Datasets: ${#DATASETS[@]}"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    echo "Total jobs: $total_jobs"
    echo "=============================================="
    
    mkdir -p logs
    
    local count=0
    for combo in "${STD_COMBINATIONS[@]}"; do
        local std1="${combo%+*}"
        local std2="${combo#*+}"
        
        for dataset in "${DATASETS[@]}"; do
            for variant_filter in "${domain_variants[@]}"; do
                submit_job "$dataset" "$model" "$std1" "$std2" "$dry_run" "$variant_filter"
                count=$((count + 1))
            done
        done
    done
    
    echo ""
    echo "Submitted $count jobs for model: $model"
}

cmd_submit_combo() {
    local std1=$1
    local std2=$2
    local dry_run=$3
    local domain_filter=$4
    local with_domain_variants=$5
    
    if [[ -z "$std1" ]] || [[ -z "$std2" ]]; then
        echo "Error: --std1 and --std2 are required for submit-combo"
        exit 1
    fi
    
    # Determine domain variants to process
    local domain_variants=("")
    if [[ "$with_domain_variants" == "true" ]]; then
        domain_variants=("" "clear_day")
    elif [[ -n "$domain_filter" ]]; then
        domain_variants=("$domain_filter")
    fi
    
    local total_jobs=$((${#DATASETS[@]} * ${#MODELS[@]} * ${#domain_variants[@]}))
    
    echo "=============================================="
    echo "Submitting all datasets/models for: ${std1}+${std2}"
    echo "=============================================="
    echo "Datasets: ${#DATASETS[@]}"
    echo "Models: ${#MODELS[@]}"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    echo "Total jobs: $total_jobs"
    echo "=============================================="
    
    mkdir -p logs
    
    local count=0
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for variant_filter in "${domain_variants[@]}"; do
                submit_job "$dataset" "$model" "$std1" "$std2" "$dry_run" "$variant_filter"
                count=$((count + 1))
            done
        done
    done
    
    echo ""
    echo "Submitted $count jobs for ${std1}+${std2}"
}

cmd_list() {
    echo "=============================================="
    echo "Std+Std Combination Training Jobs"
    echo "=============================================="
    echo ""
    echo "Standard Augmentation Strategies: ${#STD_STRATEGIES[@]}"
    for std in "${STD_STRATEGIES[@]}"; do
        echo "  - $std"
    done
    echo ""
    echo "Unique 2-way Combinations: ${#STD_COMBINATIONS[@]}"
    for combo in "${STD_COMBINATIONS[@]}"; do
        echo "  - $combo"
    done
    echo ""
    echo "Datasets: ${#DATASETS[@]}"
    for ds in "${DATASETS[@]}"; do
        echo "  - $ds"
    done
    echo ""
    echo "Models: ${#MODELS[@]}"
    for model in "${MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "=============================================="
    local total_jobs=$((${#STD_COMBINATIONS[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))
    echo "Total jobs (without domain variants): $total_jobs"
    echo "Total jobs (with domain variants): $((total_jobs * 2))"
    echo "=============================================="
}

cmd_estimate() {
    echo "=============================================="
    echo "Resource Estimation"
    echo "=============================================="
    echo ""
    echo "Configuration:"
    echo "  Std combinations: ${#STD_COMBINATIONS[@]}"
    echo "  Datasets: ${#DATASETS[@]}"
    echo "  Models: ${#MODELS[@]}"
    echo ""
    
    local total_jobs=$((${#STD_COMBINATIONS[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))
    local total_with_variants=$((total_jobs * 2))
    
    echo "Job counts:"
    echo "  Without domain variants: $total_jobs jobs"
    echo "  With domain variants: $total_with_variants jobs"
    echo ""
    
    # Estimate time (assuming ~6 hours per job)
    local hours_per_job=6
    local total_gpu_hours=$((total_jobs * hours_per_job))
    local total_gpu_hours_variants=$((total_with_variants * hours_per_job))
    
    echo "Estimated GPU hours (@ ${hours_per_job}h/job):"
    echo "  Without domain variants: $total_gpu_hours GPU-hours"
    echo "  With domain variants: $total_gpu_hours_variants GPU-hours"
    echo ""
    
    echo "LSF Configuration:"
    echo "  Queue: $LSF_QUEUE"
    echo "  GPU memory: $GPU_MEM"
    echo "  GPU mode: $GPU_MODE"
    echo "  CPUs per job: $NUM_CPUS"
    echo "=============================================="
}

# ============================================================================
# Main
# ============================================================================

main() {
    local command="${1:-help}"
    shift || true
    
    # Parse global options
    local std1=""
    local std2=""
    local dataset=""
    local model=""
    local dry_run="false"
    local delay="1"
    local domain_filter=""
    local with_domain_variants="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --std1) std1="$2"; shift 2 ;;
            --std2) std2="$2"; shift 2 ;;
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --queue) LSF_QUEUE="$2"; shift 2 ;;
            --gpu-mem) GPU_MEM="$2"; shift 2 ;;
            --gpu-mode) GPU_MODE="$2"; shift 2 ;;
            --dry-run) dry_run="true"; shift ;;
            --delay) delay="$2"; shift 2 ;;
            --domain-filter) domain_filter="$2"; shift 2 ;;
            --with-domain-variants) with_domain_variants="true"; shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    case $command in
        submit-all)
            cmd_submit_all "$dry_run" "$delay" "$domain_filter" "$with_domain_variants"
            ;;
        submit-single)
            cmd_submit_single "$std1" "$std2" "$dataset" "$model" "$dry_run" "$domain_filter"
            ;;
        submit-dataset)
            cmd_submit_dataset "$dataset" "$dry_run" "$domain_filter" "$with_domain_variants"
            ;;
        submit-model)
            cmd_submit_model "$model" "$dry_run" "$domain_filter" "$with_domain_variants"
            ;;
        submit-combo)
            cmd_submit_combo "$std1" "$std2" "$dry_run" "$domain_filter" "$with_domain_variants"
            ;;
        list)
            cmd_list
            ;;
        estimate)
            cmd_estimate
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            echo "Unknown command: $command"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"

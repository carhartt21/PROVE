#!/bin/bash
# PROVE - Submit Gen+Std Combined Strategy Training Jobs to LSF
#
# This script submits batch training jobs that combine the top-performing
# generative augmentation strategies with standard augmentation strategies.
#
# Based on test_result_analyzer.py comprehensive analysis:
# - Top gen_* strategies (by avg mIoU improvement over baseline):
#   1. gen_StyleID (+4.14%)
#   2. gen_cycleGAN (+3.83%)
#   3. gen_LANIT (+3.58%)
#   4. gen_CUT (+3.56%)
#   5. gen_step1x_new (+3.56%)
#   6. gen_automold (+3.48%)
#
# - Top std_* strategies (by avg mIoU):
#   1. std_randaugment (56.44%)
#   2. std_mixup (56.19%)
#   3. std_cutmix (55.93%)
#   4. std_autoaugment (55.36%)
#
# Generated on: $(date)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# Configuration - Customize these based on your needs
# ============================================================================

# Top-performing generative strategies (ranked by improvement over baseline)
TOP_GEN_STRATEGIES=(
    "gen_StyleID"
    "gen_cycleGAN"
    "gen_LANIT"
    "gen_CUT"
    "gen_step1x_new"
    "gen_automold"
)

# All standard augmentation strategies to combine with
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
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Gen+Std Combined Strategy Batch Submission"
    echo "================================================="
    echo ""
    echo "This script submits training jobs for promising gen_*+std_* combinations"
    echo "based on test result analysis."
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  submit-all      Submit all gen+std combinations for all datasets/models"
    echo "  submit-top      Submit only top 3 gen + top 2 std combinations (reduced)"
    echo "  submit-single   Submit single combination (requires --gen and --std)"
    echo "  submit-dataset  Submit all combinations for one dataset"
    echo "  submit-model    Submit all combinations for one model"
    echo "  list            List all combinations that would be submitted"
    echo "  estimate        Estimate total jobs and resources"
    echo "  help            Show this help"
    echo ""
    echo "Options:"
    echo "  --gen <strategy>          Specific gen_* strategy"
    echo "  --std <strategy>          Specific std_* strategy"
    echo "  --dataset <name>          Specific dataset"
    echo "  --model <name>            Specific model"
    echo "  --queue <name>            LSF queue (default: BatchGPU)"
    echo "  --gpu-mem <size>          GPU memory (default: 24G)"
    echo "  --gpu-mode <mode>         GPU mode: shared/exclusive_process (default: shared)"
    echo "  --dry-run                 Show commands without executing"
    echo "  --delay <seconds>         Delay between job submissions (default: 1)"
    echo "  --domain-filter <domain>  Add domain filter (e.g., clear_day)"
    echo "  --with-domain-variants    Submit both regular AND clear_day variants"
    echo "  --with-baseline-std       Also submit baseline+std combinations"
    echo ""
    echo "Examples:"
    echo "  $0 submit-all --dry-run"
    echo "  $0 submit-top"
    echo "  $0 submit-single --gen gen_cycleGAN --std std_cutmix --dataset ACDC --model deeplabv3plus_r50"
    echo "  $0 submit-dataset --dataset ACDC"
    echo "  $0 submit-model --model segformer_mit-b5"
    echo "  $0 list"
    echo "  $0 estimate"
}

submit_job() {
    local dataset=$1
    local model=$2
    local strategy=$3
    local std_strategy=$4
    local dry_run=$5
    local domain_filter=$6
    
    local jobname="prove_${dataset}_${model}_${strategy}+${std_strategy}"
    if [[ -n "$domain_filter" ]]; then
        jobname="${jobname}_${domain_filter}"
    fi
    
    # Build the training command
    local train_cmd="$SCRIPT_DIR/train_unified.sh single --dataset ${dataset} --model ${model} --strategy ${strategy} --std-strategy ${std_strategy}"
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

submit_baseline_std_job() {
    local dataset=$1
    local model=$2
    local std_strategy=$3
    local dry_run=$4
    local domain_filter=$5
    
    local jobname="prove_${dataset}_${model}_baseline+${std_strategy}"
    if [[ -n "$domain_filter" ]]; then
        jobname="${jobname}_${domain_filter}"
    fi
    
    # Build the training command
    local train_cmd="$SCRIPT_DIR/train_unified.sh single --dataset ${dataset} --model ${model} --strategy baseline --std-strategy ${std_strategy}"
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
    local with_baseline_std=$4
    local with_domain_variants=$5
    
    # Determine domain variants to process
    local domain_variants=("")
    if [[ "$with_domain_variants" == "true" ]]; then
        domain_variants=("" "clear_day")
    elif [[ -n "$domain_filter" ]]; then
        domain_variants=("$domain_filter")
    fi
    
    echo "=============================================="
    echo "Submitting ALL gen+std combinations"
    echo "=============================================="
    echo "Gen strategies: ${#TOP_GEN_STRATEGIES[@]}"
    echo "Std strategies: ${#STD_STRATEGIES[@]}"
    echo "Datasets: ${#DATASETS[@]}"
    echo "Models: ${#MODELS[@]}"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    local total_jobs=$((${#TOP_GEN_STRATEGIES[@]} * ${#STD_STRATEGIES[@]} * ${#DATASETS[@]} * ${#MODELS[@]} * ${#domain_variants[@]}))
    if [[ "$with_baseline_std" == "true" ]]; then
        total_jobs=$((total_jobs + ${#STD_STRATEGIES[@]} * ${#DATASETS[@]} * ${#MODELS[@]} * ${#domain_variants[@]}))
    fi
    echo "Total jobs: $total_jobs"
    echo "=============================================="
    
    local count=0
    
    # Submit baseline+std combinations if requested
    if [[ "$with_baseline_std" == "true" ]]; then
        echo ""
        echo "--- Baseline + Std Combinations ---"
        for dataset in "${DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do
                for std_strategy in "${STD_STRATEGIES[@]}"; do
                    for variant_filter in "${domain_variants[@]}"; do
                        submit_baseline_std_job "$dataset" "$model" "$std_strategy" "$dry_run" "$variant_filter"
                        count=$((count + 1))
                        if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]]; then
                            sleep "$delay"
                        fi
                    done
                done
            done
        done
    fi
    
    # Submit gen+std combinations
    echo ""
    echo "--- Gen + Std Combinations ---"
    for gen_strategy in "${TOP_GEN_STRATEGIES[@]}"; do
        for std_strategy in "${STD_STRATEGIES[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for model in "${MODELS[@]}"; do
                    for variant_filter in "${domain_variants[@]}"; do
                        submit_job "$dataset" "$model" "$gen_strategy" "$std_strategy" "$dry_run" "$variant_filter"
                        count=$((count + 1))
                        if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]]; then
                            sleep "$delay"
                        fi
                    done
                done
            done
        done
    done
    
    echo ""
    echo "=============================================="
    echo "Submitted $count jobs"
    echo "=============================================="
}

cmd_submit_top() {
    local dry_run=$1
    local delay=$2
    local domain_filter=$3
    local with_domain_variants=$4
    
    # Top 3 gen strategies
    local top_gen=("gen_StyleID" "gen_cycleGAN" "gen_CUT")
    # Top 2 std strategies
    local top_std=("std_randaugment" "std_mixup")
    
    # Determine domain variants to process
    local domain_variants=("")
    if [[ "$with_domain_variants" == "true" ]]; then
        domain_variants=("" "clear_day")
    elif [[ -n "$domain_filter" ]]; then
        domain_variants=("$domain_filter")
    fi
    
    echo "=============================================="
    echo "Submitting TOP gen+std combinations"
    echo "=============================================="
    echo "Gen strategies: ${top_gen[*]}"
    echo "Std strategies: ${top_std[*]}"
    echo "Datasets: ${#DATASETS[@]}"
    echo "Models: ${#MODELS[@]}"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    local total_jobs=$((${#top_gen[@]} * ${#top_std[@]} * ${#DATASETS[@]} * ${#MODELS[@]} * ${#domain_variants[@]}))
    echo "Total jobs: $total_jobs"
    echo "=============================================="
    
    local count=0
    for gen_strategy in "${top_gen[@]}"; do
        for std_strategy in "${top_std[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for model in "${MODELS[@]}"; do
                    for variant_filter in "${domain_variants[@]}"; do
                        submit_job "$dataset" "$model" "$gen_strategy" "$std_strategy" "$dry_run" "$variant_filter"
                        count=$((count + 1))
                        if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]]; then
                            sleep "$delay"
                        fi
                    done
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
    local gen_strategy=$1
    local std_strategy=$2
    local dataset=$3
    local model=$4
    local dry_run=$5
    local domain_filter=$6
    
    if [[ -z "$gen_strategy" ]] || [[ -z "$std_strategy" ]] || [[ -z "$dataset" ]] || [[ -z "$model" ]]; then
        echo "Error: --gen, --std, --dataset, and --model are required for submit-single"
        exit 1
    fi
    
    submit_job "$dataset" "$model" "$gen_strategy" "$std_strategy" "$dry_run" "$domain_filter"
}

cmd_submit_dataset() {
    local dataset=$1
    local dry_run=$2
    local delay=$3
    local domain_filter=$4
    local with_domain_variants=$5
    
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
    
    echo "=============================================="
    echo "Submitting all combinations for dataset: $dataset"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    echo "=============================================="
    
    local count=0
    for gen_strategy in "${TOP_GEN_STRATEGIES[@]}"; do
        for std_strategy in "${STD_STRATEGIES[@]}"; do
            for model in "${MODELS[@]}"; do
                for variant_filter in "${domain_variants[@]}"; do
                    submit_job "$dataset" "$model" "$gen_strategy" "$std_strategy" "$dry_run" "$variant_filter"
                    count=$((count + 1))
                    if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]]; then
                        sleep "$delay"
                    fi
                done
            done
        done
    done
    
    echo ""
    echo "Submitted $count jobs for $dataset"
}

cmd_submit_model() {
    local model=$1
    local dry_run=$2
    local delay=$3
    local domain_filter=$4
    local with_domain_variants=$5
    
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
    
    echo "=============================================="
    echo "Submitting all combinations for model: $model"
    echo "Domain variants: ${domain_variants[*]:-'(none)'}"
    echo "=============================================="
    
    local count=0
    for gen_strategy in "${TOP_GEN_STRATEGIES[@]}"; do
        for std_strategy in "${STD_STRATEGIES[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for variant_filter in "${domain_variants[@]}"; do
                    submit_job "$dataset" "$model" "$gen_strategy" "$std_strategy" "$dry_run" "$variant_filter"
                    count=$((count + 1))
                    if [[ "$dry_run" != "true" ]] && [[ -n "$delay" ]]; then
                        sleep "$delay"
                    fi
                done
            done
        done
    done
    
    echo ""
    echo "Submitted $count jobs for $model"
}

cmd_list() {
    echo "=============================================="
    echo "All gen+std combinations to be submitted"
    echo "=============================================="
    echo ""
    
    local count=0
    for gen_strategy in "${TOP_GEN_STRATEGIES[@]}"; do
        for std_strategy in "${STD_STRATEGIES[@]}"; do
            echo "  ${gen_strategy} + ${std_strategy}"
            count=$((count + 1))
        done
    done
    
    echo ""
    echo "Total combinations: $count"
    echo ""
    echo "With datasets and models:"
    echo "  Datasets: ${DATASETS[*]}"
    echo "  Models: ${MODELS[*]}"
    echo ""
    local total_jobs=$((count * ${#DATASETS[@]} * ${#MODELS[@]}))
    local total_jobs_with_variants=$((total_jobs * 2))
    echo "Total training jobs (no domain variants): $total_jobs"
    echo "Total training jobs (with --with-domain-variants): $total_jobs_with_variants"
    echo ""
    echo "Note: Use --with-domain-variants to submit both regular AND clear_day variants"
}

cmd_estimate() {
    echo "=============================================="
    echo "Resource Estimation"
    echo "=============================================="
    echo ""
    
    local total_combinations=$((${#TOP_GEN_STRATEGIES[@]} * ${#STD_STRATEGIES[@]}))
    local total_jobs=$((total_combinations * ${#DATASETS[@]} * ${#MODELS[@]}))
    local total_jobs_with_variants=$((total_jobs * 2))
    
    echo "Configuration:"
    echo "  Gen strategies: ${#TOP_GEN_STRATEGIES[@]} (${TOP_GEN_STRATEGIES[*]})"
    echo "  Std strategies: ${#STD_STRATEGIES[@]} (${STD_STRATEGIES[*]})"
    echo "  Datasets: ${#DATASETS[@]} (${DATASETS[*]})"
    echo "  Models: ${#MODELS[@]} (${MODELS[*]})"
    echo ""
    echo "Calculations:"
    echo "  Total strategy combinations: $total_combinations"
    echo "  Total training jobs (no domain variants): $total_jobs"
    echo "  Total training jobs (with domain variants): $total_jobs_with_variants"
    echo ""
    echo "Estimated Resources (assuming ~4-8 hours per job):"
    echo "  GPU hours (no variants): $((total_jobs * 6)) hours (avg 6h per job)"
    echo "  GPU hours (with variants): $((total_jobs_with_variants * 6)) hours"
    echo "  Sequential runtime (no variants): $((total_jobs * 6)) hours"
    echo "  Sequential runtime (with variants): $((total_jobs_with_variants * 6)) hours"
    echo "  With 10 parallel jobs: $((total_jobs_with_variants * 6 / 10)) hours"
    echo "  With 20 parallel jobs: $((total_jobs_with_variants * 6 / 20)) hours"
    echo ""
    echo "LSF Settings:"
    echo "  Queue: $LSF_QUEUE"
    echo "  GPU Memory: $GPU_MEM"
    echo "  GPU Mode: $GPU_MODE"
    echo "  CPUs per job: $NUM_CPUS"
    echo ""
    echo "Commands to submit:"
    echo "  All jobs:                    $0 submit-all"
    echo "  All jobs (w/ variants):      $0 submit-all --with-domain-variants"
    echo "  Top combinations:            $0 submit-top ($(( 3 * 2 * ${#DATASETS[@]} * ${#MODELS[@]} )) jobs)"
    echo "  Top combinations (w/ vars):  $0 submit-top --with-domain-variants ($(( 3 * 2 * ${#DATASETS[@]} * ${#MODELS[@]} * 2 )) jobs)"
    echo "  Dry run:                     $0 submit-all --dry-run"
}

# ============================================================================
# Main
# ============================================================================

if [[ $# -eq 0 ]]; then
    print_usage
    exit 0
fi

COMMAND=$1
shift

# Parse arguments
DRY_RUN="false"
DELAY="1"
DOMAIN_FILTER=""
WITH_BASELINE_STD="false"
WITH_DOMAIN_VARIANTS="false"
GEN_STRATEGY=""
STD_STRATEGY=""
ARG_DATASET=""
ARG_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gen)
            GEN_STRATEGY=$2
            shift 2
            ;;
        --std)
            STD_STRATEGY=$2
            shift 2
            ;;
        --dataset)
            ARG_DATASET=$2
            shift 2
            ;;
        --model)
            ARG_MODEL=$2
            shift 2
            ;;
        --queue)
            LSF_QUEUE=$2
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM=$2
            shift 2
            ;;
        --gpu-mode)
            GPU_MODE=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --delay)
            DELAY=$2
            shift 2
            ;;
        --domain-filter)
            DOMAIN_FILTER=$2
            shift 2
            ;;
        --with-domain-variants)
            WITH_DOMAIN_VARIANTS="true"
            shift
            ;;
        --with-baseline-std)
            WITH_BASELINE_STD="true"
            shift
            ;;
        --help|-h)
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

# Execute command
case $COMMAND in
    submit-all)
        cmd_submit_all "$DRY_RUN" "$DELAY" "$DOMAIN_FILTER" "$WITH_BASELINE_STD" "$WITH_DOMAIN_VARIANTS"
        ;;
    submit-top)
        cmd_submit_top "$DRY_RUN" "$DELAY" "$DOMAIN_FILTER" "$WITH_DOMAIN_VARIANTS"
        ;;
    submit-single)
        cmd_submit_single "$GEN_STRATEGY" "$STD_STRATEGY" "$ARG_DATASET" "$ARG_MODEL" "$DRY_RUN" "$DOMAIN_FILTER"
        ;;
    submit-dataset)
        cmd_submit_dataset "$ARG_DATASET" "$DRY_RUN" "$DELAY" "$DOMAIN_FILTER" "$WITH_DOMAIN_VARIANTS"
        ;;
    submit-model)
        cmd_submit_model "$ARG_MODEL" "$DRY_RUN" "$DELAY" "$DOMAIN_FILTER" "$WITH_DOMAIN_VARIANTS"
        ;;
    list)
        cmd_list
        ;;
    estimate)
        cmd_estimate
        ;;
    help)
        print_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac

#!/bin/bash
# PROVE Unified Training - Example Usage Script
#
# This script demonstrates how to use the unified training system
# for various training scenarios.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# Configuration
# ============================================================================

# Available datasets
SEGMENTATION_DATASETS="ACDC BDD10k IDD-AW MapillaryVistas OUTSIDE15k"
DETECTION_DATASETS="BDD100k"

# Available models
SEGMENTATION_MODELS="deeplabv3plus_r50 pspnet_r50 segformer_mit-b5"
DETECTION_MODELS="faster_rcnn_r50_fpn_1x yolox_l rtmdet_l"

# Available strategies
# - Base strategies: baseline, photometric_distort
# - Standard augmentations: std_cutmix, std_mixup, std_autoaugment, std_randaugment
# - Generative models: gen_cycleGAN, gen_CUT, gen_stargan_v2, gen_SUSTechGAN, gen_EDICT,
#                      gen_Img2Img, gen_IP2P, gen_UniControl, gen_step1x_new, gen_StyleID,
#                      gen_NST, gen_albumentations, gen_automold, gen_imgaug_weather,
#                      gen_Weather_Effect_Generator, gen_Attribute_Hallucination, 
#                      gen_cnet_seg, gen_tunit, gen_flux1_kontext
STRATEGIES="baseline photometric_distort std_cutmix std_mixup std_autoaugment std_randaugment gen_cycleGAN gen_CUT gen_stargan_v2 gen_SUSTechGAN gen_EDICT gen_Img2Img gen_IP2P gen_UniControl gen_step1x_new gen_StyleID gen_NST gen_albumentations gen_automold gen_imgaug_weather gen_Weather_Effect_Generator gen_Attribute_Hallucination gen_cnet_seg gen_tunit gen_flux1_kontext"

# Real-to-generated ratios to try
RATIOS="1.0 0.875 0.625 0.5 0.375 0.25 0.125 0.0"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Unified Training Script"
    echo "=============================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  single      Train a single configuration"
    echo "  batch       Train multiple configurations"
    echo "  ratio-exp   Run ratio experiment for one model"
    echo "  generate    Generate configs without training"
    echo "  list        List available options"
    echo "  help        Show this help message"
    echo ""
    echo "Single Training Options:"
    echo "  --dataset <name>          Dataset name (ACDC, BDD10k, BDD100k, IDD-AW, MapillaryVistas, OUTSIDE15k)"
    echo "  --model <name>            Model name (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5, etc.)"
    echo "  --strategy <name>         Augmentation strategy (see below)"
    echo "  --real-gen-ratio <ratio>  Ratio of real to generated images (0.0 to 1.0)"
    echo "  --domain-filter <domain>  Filter training data to specific domain (e.g., clear_day)"
    echo "  --cache-dir <path>        Directory for caching pretrained weights"
    echo "  --no-early-stop           Disable early stopping (enabled by default)"
    echo "  --early-stop-patience <n> Number of validations without improvement before stopping (default: 5)"
    echo ""
    echo "Batch Training Options:"
    echo "  --datasets <names...>     List of datasets for batch training"
    echo "  --models <names...>       List of models for batch training"
    echo "  --all-seg-datasets        Use all segmentation datasets"
    echo "  --all-det-datasets        Use all detection datasets"
    echo "  --all-seg-models          Use all segmentation models"
    echo "  --all-det-models          Use all detection models"
    echo "  --parallel                Run jobs in parallel"
    echo "  --dry-run                 Show commands without executing"
    echo ""
    echo "Strategies:"
    echo "  Base:        baseline, photometric_distort"
    echo "  Standard:    std_cutmix, std_mixup, std_autoaugment, std_randaugment"
    echo "  Generative:  gen_cycleGAN, gen_CUT, gen_stargan_v2, gen_SUSTechGAN, gen_EDICT,"
    echo "               gen_Img2Img, gen_IP2P, gen_UniControl, gen_step1x_new, gen_StyleID,"
    echo "               gen_NST, gen_albumentations, gen_automold, gen_imgaug_weather,"
    echo "               gen_Weather_Effect_Generator, gen_Attribute_Hallucination,"
    echo "               gen_cnet_seg, gen_tunit, gen_flux1_kontext"
    echo ""
    echo "Examples:"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --domain-filter clear_day"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --cache-dir /data/pretrained"
    echo "  $0 batch --all-seg-datasets --all-seg-models --strategy baseline"
    echo "  $0 batch --all-det-datasets --all-det-models --strategy baseline --dry-run"
    echo "  $0 batch --datasets ACDC BDD10k --strategy gen_cycleGAN"
    echo "  $0 ratio-exp --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN"
    echo "  $0 generate --strategy baseline --all"
    echo ""
    echo "Run '$0 list' for detailed available options."
}

train_single() {
    local dataset=$1
    local model=$2
    local strategy=$3
    local ratio=${4:-1.0}
    local domain_filter=${5:-}
    
    echo "Training: $dataset / $model / $strategy (ratio=$ratio)"
    if [ -n "$domain_filter" ]; then
        echo "Domain filter: $domain_filter"
        python unified_training.py \
            --dataset "$dataset" \
            --model "$model" \
            --strategy "$strategy" \
            --real-gen-ratio "$ratio" \
            --domain-filter "$domain_filter"
    else
        python unified_training.py \
            --dataset "$dataset" \
            --model "$model" \
            --strategy "$strategy" \
            --real-gen-ratio "$ratio"
    fi
}

# ============================================================================
# Commands
# ============================================================================

cmd_single() {
    # Parse arguments
    local dataset=""
    local model=""
    local strategy="baseline"
    local ratio="1.0"
    local domain_filter=""
    local cache_dir=""
    local no_early_stop=false
    local early_stop_patience=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --domain-filter) domain_filter="$2"; shift 2 ;;
            --cache-dir) cache_dir="$2"; shift 2 ;;
            --no-early-stop) no_early_stop=true; shift ;;
            --early-stop-patience) early_stop_patience="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "Error: --dataset and --model are required"
        exit 1
    fi
    
    # Build command with optional parameters
    local cmd="python unified_training.py --dataset $dataset --model $model --strategy $strategy --real-gen-ratio $ratio"
    
    if [ -n "$domain_filter" ]; then
        cmd="$cmd --domain-filter $domain_filter"
    fi
    if [ -n "$cache_dir" ]; then
        cmd="$cmd --cache-dir $cache_dir"
    fi
    if [ "$no_early_stop" = true ]; then
        cmd="$cmd --no-early-stop"
    fi
    if [ -n "$early_stop_patience" ]; then
        cmd="$cmd --early-stop-patience $early_stop_patience"
    fi
    
    echo "Training: $dataset / $model / $strategy (ratio=$ratio)"
    if [ -n "$domain_filter" ]; then
        echo "Domain filter: $domain_filter"
    fi
    if [ -n "$cache_dir" ]; then
        echo "Cache dir: $cache_dir"
    fi
    
    eval $cmd
}

cmd_batch() {
    local datasets=""
    local models=""
    local strategy="baseline"
    local ratio="1.0"
    local parallel=false
    local all_seg_datasets=false
    local all_det_datasets=false
    local all_seg_models=false
    local all_det_models=false
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --datasets) shift; 
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    datasets="$datasets $1"
                    shift
                done
                ;;
            --models) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    models="$models $1"
                    shift
                done
                ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --parallel) parallel=true; shift ;;
            --all-seg-datasets) all_seg_datasets=true; shift ;;
            --all-det-datasets) all_det_datasets=true; shift ;;
            --all-seg-models) all_seg_models=true; shift ;;
            --all-det-models) all_det_models=true; shift ;;
            --dry-run) dry_run=true; shift ;;
            *) shift ;;
        esac
    done
    
    # Build command arguments
    local cmd="python unified_training.py --batch"
    
    if [ -n "$datasets" ]; then
        cmd="$cmd --datasets $datasets"
    fi
    if [ -n "$models" ]; then
        cmd="$cmd --models $models"
    fi
    if [ "$all_seg_datasets" = true ]; then
        cmd="$cmd --all-seg-datasets"
    fi
    if [ "$all_det_datasets" = true ]; then
        cmd="$cmd --all-det-datasets"
    fi
    if [ "$all_seg_models" = true ]; then
        cmd="$cmd --all-seg-models"
    fi
    if [ "$all_det_models" = true ]; then
        cmd="$cmd --all-det-models"
    fi
    cmd="$cmd --strategies $strategy --ratios $ratio"
    if [ "$parallel" = true ]; then
        cmd="$cmd --parallel"
    fi
    if [ "$dry_run" = true ]; then
        cmd="$cmd --dry-run"
    fi
    
    echo "Executing: $cmd"
    eval $cmd
}

cmd_ratio_experiment() {
    local dataset=""
    local model=""
    local strategy="gen_cycleGAN"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "Error: --dataset and --model are required"
        exit 1
    fi
    
    echo "Ratio Experiment"
    echo "================"
    echo "Dataset: $dataset"
    echo "Model: $model"
    echo "Strategy: $strategy"
    echo "Ratios: $RATIOS"
    echo ""
    
    for ratio in $RATIOS; do
        echo ""
        echo ">>> Training with ratio=$ratio"
        train_single "$dataset" "$model" "$strategy" "$ratio"
    done
    
    echo ""
    echo "Ratio experiment complete!"
}

cmd_generate() {
    local strategy="baseline"
    local all=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --strategy) strategy="$2"; shift 2 ;;
            --all) all=true; shift ;;
            *) shift ;;
        esac
    done
    
    if $all; then
        echo "Generating all configs for strategy: $strategy"
        python unified_training_config.py --generate-all --strategy "$strategy"
    else
        echo "Generating configs for strategy: $strategy"
        python unified_training_config.py --generate-all --strategy "$strategy"
    fi
}

cmd_list() {
    python unified_training_config.py --list
}

# ============================================================================
# Main
# ============================================================================

case "${1:-help}" in
    single)
        shift
        cmd_single "$@"
        ;;
    batch)
        shift
        cmd_batch "$@"
        ;;
    ratio-exp)
        shift
        cmd_ratio_experiment "$@"
        ;;
    generate)
        shift
        cmd_generate "$@"
        ;;
    list)
        cmd_list
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "Unknown command: $1"
        print_usage
        exit 1
        ;;
esac

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
STRATEGIES="baseline photometric_distort gen_cycleGAN gen_CUT gen_stargan_v2"

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
    echo "Options:"
    echo "  --domain-filter <domain>  Filter training data to specific domain (e.g., clear_day)"
    echo ""
    echo "Examples:"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --domain-filter clear_day"
    echo "  $0 batch --datasets ACDC BDD10k --strategy gen_cycleGAN"
    echo "  $0 ratio-exp --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN"
    echo "  $0 generate --strategy baseline --all"
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
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --domain-filter) domain_filter="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "Error: --dataset and --model are required"
        exit 1
    fi
    
    train_single "$dataset" "$model" "$strategy" "$ratio" "$domain_filter"
}

cmd_batch() {
    local datasets=""
    local models=""
    local strategy="baseline"
    local ratio="1.0"
    local parallel=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --datasets) shift; datasets="$@"; break ;;
            --models) shift; models="$@"; break ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --parallel) parallel=true; shift ;;
            *) shift ;;
        esac
    done
    
    # Default to all segmentation datasets
    if [ -z "$datasets" ]; then
        datasets="$SEGMENTATION_DATASETS"
    fi
    
    # Default to all segmentation models
    if [ -z "$models" ]; then
        models="$SEGMENTATION_MODELS"
    fi
    
    echo "Batch Training"
    echo "=============="
    echo "Datasets: $datasets"
    echo "Models: $models"
    echo "Strategy: $strategy"
    echo "Ratio: $ratio"
    echo ""
    
    for dataset in $datasets; do
        for model in $models; do
            if $parallel; then
                train_single "$dataset" "$model" "$strategy" "$ratio" &
            else
                train_single "$dataset" "$model" "$strategy" "$ratio"
            fi
        done
    done
    
    if $parallel; then
        wait
    fi
    
    echo ""
    echo "Batch training complete!"
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

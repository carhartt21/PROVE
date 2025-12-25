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
UNIFIED_SEG_DATASET="multi_ACDC+MapillaryVistas+IDD-AW+BDD10k"

# Available models
SEGMENTATION_MODELS="deeplabv3plus_r50 pspnet_r50 segformer_mit-b5"
DETECTION_MODELS="faster_rcnn_r50_fpn_1x yolox_l rtmdet_l"

# Available strategies
# - Base strategies: baseline, photometric_distort
# - Standard augmentations: std_cutmix, std_mixup, std_autoaugment, std_randaugment
# - Generative models: gen_albumentations_weather, gen_AOD_Net, gen_Attribute_Hallucination,
#                      gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cyclediffusion,
#                      gen_cycleGAN, gen_EDICT, gen_flux2, gen_flux_kontext, gen_Img2Img,
#                      gen_IP2P, gen_LANIT, gen_NST, gen_Qwen_Image_Edit, gen_stargan_v2,
#                      gen_step1x_new, gen_step1x_v1p2, gen_StyleID, gen_SUSTechGAN, gen_TSIT,
#                      gen_tunit, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator
STRATEGIES="baseline photometric_distort std_cutmix std_mixup std_autoaugment std_randaugment gen_albumentations_weather gen_AOD_Net gen_Attribute_Hallucination gen_augmenters gen_automold gen_CNetSeg gen_CUT gen_cyclediffusion gen_cycleGAN gen_EDICT gen_flux2 gen_flux_kontext gen_Img2Img gen_IP2P gen_LANIT gen_NST gen_Qwen_Image_Edit gen_stargan_v2 gen_step1x_new gen_step1x_v1p2 gen_StyleID gen_SUSTechGAN gen_TSIT gen_tunit gen_UniControl gen_VisualCloze gen_Weather_Effect_Generator"

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
    echo "  single       Train a single configuration"
    echo "  single-multi Train jointly on multiple datasets"
    echo "  batch        Train multiple configurations"
    echo "  submit       Submit single job to LSF cluster"
    echo "  submit-batch Submit multiple jobs to LSF cluster"
    echo "  ratio-exp   Run ratio experiment for one model"
    echo "  generate    Generate configs without training"
    echo "  list        List available options"
    echo "  help        Show this help message"
    echo ""
    echo "Single Training Options:"
    echo "  --dataset <name>          Dataset name (ACDC, BDD10k, BDD100k, IDD-AW, MapillaryVistas, OUTSIDE15k)"
    echo "  --model <name>            Model name (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5, etc.)"
    echo "  --strategy <name>         Augmentation strategy (see below)"
    echo "  --std-strategy <name>     Standard augmentation to combine with main strategy"
    echo "                            (std_cutmix, std_mixup, std_autoaugment, std_randaugment)"
    echo "  --real-gen-ratio <ratio>  Ratio of real to generated images (0.0 to 1.0)"
    echo "  --domain-filter <domain>  Filter training data to specific domain (e.g., clear_day)"
    echo "  --cache-dir <path>        Directory for caching pretrained weights"
    echo "  --no-early-stop           Disable early stopping (enabled by default)"
    echo "  --early-stop-patience <n> Number of validations without improvement before stopping (default: 5)"
    echo ""
    echo "Multi-Dataset Training Options (single-multi command):"
    echo "  --datasets <names...>     List of datasets to train jointly (e.g., ACDC MapillaryVistas)"
    echo "  --weights <floats...>     Optional sampling weights per dataset (must sum to 1.0)"
    echo "  --model <name>            Model name"
    echo "  --strategy <name>         Augmentation strategy"
    echo "  --std-strategy <name>     Standard augmentation to combine with main strategy"
    echo "  --config-only             Only generate config, do not train"
    echo ""
    echo "Single Training Options:"
    echo "  --dataset <name>          Dataset name (ACDC, BDD10k, BDD100k, IDD-AW, MapillaryVistas, OUTSIDE15k)"
    echo "  --model <name>            Model name (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5, etc.)"
    echo "  --strategy <name>         Augmentation strategy (see below)"
    echo "  --std-strategy <name>     Standard augmentation to combine with main strategy"
    echo "  --real-gen-ratio <ratio>  Ratio of real to generated images (0.0 to 1.0)"
    echo "  --domain-filter <domain>  Filter training data to specific domain (e.g., clear_day)"
    echo "  --cache-dir <path>        Directory for caching pretrained weights"
    echo "  --no-early-stop           Disable early stopping (enabled by default)"
    echo "  --early-stop-patience <n> Number of validations without improvement before stopping (default: 5)"
    echo ""
    echo "LSF Submit Options:"
    echo "  --queue <name>            LSF queue name (default: BatchGPU)"
    echo "  --gpu-mem <size>          GPU memory requirement (default: 24G)"
    echo "  --gpu-mode <mode>         GPU mode: shared or exclusive_process (default: shared)"
    echo "  --num-cpus <n>            Number of CPUs per job (default: 8)"
    echo "  --domain-filter <domain>  Filter training data to specific domain (e.g., clear_day)"
    echo "  --dry-run                 Show bsub command without executing"
    echo ""
    echo "Batch Training Options:"
    echo "  --datasets <names...>     List of datasets for batch training"
    echo "  --models <names...>       List of models for batch training"
    echo "  --all-seg-datasets        Use all segmentation datasets"
    echo "  --all-det-datasets        Use all detection datasets"
    echo "  --unified-seg-dataset     Use unified segmentation dataset (ACDC+BDD10k+MapillaryVistas+IDD-AW)"
    echo "  --all-seg-models          Use all segmentation models"
    echo "  --all-det-models          Use all detection models"
    echo "  --parallel                Run jobs in parallel"
    echo "  --dry-run                 Show commands without executing"
    echo ""
    echo "Strategies:"
    echo "  Base:        baseline, photometric_distort"
    echo "  Standard:    std_cutmix, std_mixup, std_autoaugment, std_randaugment"
    echo "  Generative:  gen_albumentations_weather, gen_AOD_Net, gen_Attribute_Hallucination,"
    echo "               gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cyclediffusion,"
    echo "               gen_cycleGAN, gen_EDICT, gen_flux2, gen_flux_kontext, gen_Img2Img,"
    echo "               gen_IP2P, gen_LANIT, gen_NST, gen_Qwen_Image_Edit, gen_stargan_v2,"
    echo "               gen_step1x_new, gen_step1x_v1p2, gen_StyleID, gen_SUSTechGAN, gen_TSIT,"
    echo "               gen_tunit, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator"
    echo ""
    echo "Combined Strategies (--std-strategy):"
    echo "  Use --std-strategy to combine standard augmentations with gen_* or baseline strategies."
    echo "  Example: --strategy gen_cycleGAN --std-strategy std_cutmix"
    echo ""
    echo "Examples:"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --std-strategy std_cutmix"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --domain-filter clear_day"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --cache-dir /data/pretrained"
    echo "  $0 submit --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 submit --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --std-strategy std_mixup"
    echo "  $0 submit-batch --all-seg-datasets --all-seg-models --strategy baseline"
    echo "  $0 submit-batch --unified-seg-dataset --all-seg-models --strategy baseline"
    echo "  $0 submit-batch --datasets ACDC BDD10k --models deeplabv3plus_r50 --strategy gen_cycleGAN --dry-run"
    echo "  $0 submit-batch --unified-seg-dataset --all-seg-models --strategy baseline --with-domain-variants"
    
    echo "  $0 batch --all-seg-datasets --all-seg-models --strategy baseline"
    echo "  $0 batch --unified-seg-dataset --models deeplabv3plus_r50 --strategy baseline"
    echo "  $0 batch --all-det-datasets --all-det-models --strategy baseline --dry-run"
    echo "  $0 batch --datasets ACDC BDD10k --strategy gen_cycleGAN"
    echo "  $0 single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50"
    echo "  $0 single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50 --strategy gen_CUT --std-strategy std_autoaugment"
    echo "  $0 single-multi --datasets ACDC MapillaryVistas --weights 0.7 0.3 --model deeplabv3plus_r50"
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
        mamba run -n prove python unified_training.py \
            --dataset "$dataset" \
            --model "$model" \
            --strategy "$strategy" \
            --real-gen-ratio "$ratio" \
            --domain-filter "$domain_filter"
    else
        mamba run -n prove python unified_training.py \
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
    local std_strategy=""
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
            --std-strategy) std_strategy="$2"; shift 2 ;;
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
    
    # Handle multi-dataset names (from submit-batch command)
    # Multi-dataset format: multi_ds1+ds2+ds3 (e.g., multi_acdc+bdd10k+mapillaryvistas)
    if [[ "$dataset" == multi_* ]]; then
        # Extract datasets from multi_ds1+ds2+ds3 format
        local datasets_str="${dataset#multi_}"  # Remove "multi_" prefix
        # Convert + separated list to space separated for cmd_single_multi
        local datasets_space="${datasets_str//+/ }"
        
        # Build arguments for cmd_single_multi
        local multi_args="--datasets $datasets_space --model $model --strategy $strategy --ratio $ratio"
        if [ -n "$std_strategy" ]; then
            multi_args="$multi_args --std-strategy $std_strategy"
        fi
        if [ -n "$cache_dir" ]; then
            multi_args="$multi_args --cache-dir $cache_dir"
        fi
        if [ "$no_early_stop" = true ]; then
            multi_args="$multi_args --no-early-stop"
        fi
        if [ -n "$early_stop_patience" ]; then
            multi_args="$multi_args --early-stop-patience $early_stop_patience"
        fi
        
        echo "Detected multi-dataset format, redirecting to single-multi command..."
        cmd_single_multi $multi_args
        return $?
    fi
    
    # Build command with optional parameters
    local cmd="mamba run -n prove python unified_training.py --dataset $dataset --model $model --strategy $strategy --real-gen-ratio $ratio"
    
    if [ -n "$std_strategy" ]; then
        cmd="$cmd --std-strategy $std_strategy"
    fi
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
    
    if [ -n "$std_strategy" ]; then
        echo "Training: $dataset / $model / $strategy + $std_strategy (ratio=$ratio)"
    else
        echo "Training: $dataset / $model / $strategy (ratio=$ratio)"
    fi
    if [ -n "$domain_filter" ]; then
        echo "Domain filter: $domain_filter"
    fi
    if [ -n "$cache_dir" ]; then
        echo "Cache dir: $cache_dir"
    fi
    
    eval $cmd
}

cmd_single_multi() {
    # Parse arguments for multi-dataset training
    local datasets=""
    local weights=""
    local model=""
    local strategy="baseline"
    local std_strategy=""
    local ratio="1.0"
    local cache_dir=""
    local no_early_stop=false
    local early_stop_patience=""
    local config_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --datasets) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    datasets="$datasets $1"
                    shift
                done
                ;;
            --weights) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    weights="$weights $1"
                    shift
                done
                ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --std-strategy) std_strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --cache-dir) cache_dir="$2"; shift 2 ;;
            --no-early-stop) no_early_stop=true; shift ;;
            --early-stop-patience) early_stop_patience="$2"; shift 2 ;;
            --config-only) config_only=true; shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    # Trim leading spaces
    datasets=$(echo $datasets | xargs)
    weights=$(echo $weights | xargs)
    
    if [ -z "$datasets" ] || [ -z "$model" ]; then
        echo "Error: --datasets and --model are required for multi-dataset training"
        echo "Example: $0 single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50"
        exit 1
    fi
    
    # Build command
    local cmd="mamba run -n prove python unified_training.py --multi-dataset --datasets $datasets --model $model --strategy $strategy --real-gen-ratio $ratio"
    
    if [ -n "$std_strategy" ]; then
        cmd="$cmd --std-strategy $std_strategy"
    fi
    if [ -n "$weights" ]; then
        cmd="$cmd --weights $weights"
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
    if [ "$config_only" = true ]; then
        cmd="$cmd --config-only"
    fi
    
    echo "Multi-Dataset Training"
    echo "====================="
    echo "Datasets: $datasets"
    if [ -n "$weights" ]; then
        echo "Weights: $weights"
    fi
    echo "Model: $model"
    if [ -n "$std_strategy" ]; then
        echo "Strategy: $strategy + $std_strategy"
    else
        echo "Strategy: $strategy"
    fi
    echo ""
    
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
    local unified_seg_dataset=false
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
            --unified-seg-dataset) unified_seg_dataset=true; shift ;;
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
    if [ "$unified_seg_dataset" = true ]; then
        cmd="$cmd --unified-seg-dataset"
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

cmd_submit_batch() {
    # Submit multiple jobs to LSF cluster
    local datasets=""
    local models=""
    local strategy="baseline"
    local std_strategy=""
    local ratio="1.0"
    local queue="BatchGPU"
    local gpu_mem="12G"
    local gpu_mode="shared"
    local num_cpus="8"
    local dry_run=false
    local all_seg_datasets=false
    local all_det_datasets=false
    local unified_seg_dataset=false
    local all_seg_models=false
    local all_det_models=false
    local cache_dir=""
    local no_early_stop=true
    local early_stop_patience=""
    local domain_filter=""
    local with_domain_variants=false
    
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
            --std-strategy) std_strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --domain-filter) domain_filter="$2"; shift 2 ;;
            --with-domain-variants) with_domain_variants=true; shift ;;
            --queue) queue="$2"; shift 2 ;;
            --gpu-mem) gpu_mem="$2"; shift 2 ;;
            --gpu-mode) gpu_mode="$2"; shift 2 ;;
            --num-cpus) num_cpus="$2"; shift 2 ;;
            --cache-dir) cache_dir="$2"; shift 2 ;;
            --no-early-stop) no_early_stop=true; shift ;;
            --early-stop-patience) early_stop_patience="$2"; shift 2 ;;
            --all-seg-datasets) all_seg_datasets=true; shift ;;
            --all-det-datasets) all_det_datasets=true; shift ;;
            --unified-seg-dataset) unified_seg_dataset=true; shift ;;
            --all-seg-models) all_seg_models=true; shift ;;
            --all-det-models) all_det_models=true; shift ;;
            --dry-run) dry_run=true; shift ;;
            *) shift ;;
        esac
    done
    
    # Expand all-* options
    if [ "$all_seg_datasets" = true ]; then
        datasets="$datasets $SEGMENTATION_DATASETS"
    fi
    if [ "$all_det_datasets" = true ]; then
        datasets="$datasets $DETECTION_DATASETS"
    fi
    if [ "$unified_seg_dataset" = true ]; then
        datasets="$datasets $UNIFIED_SEG_DATASET"
    fi
    if [ "$all_seg_models" = true ]; then
        models="$models $SEGMENTATION_MODELS"
    fi
    if [ "$all_det_models" = true ]; then
        models="$models $DETECTION_MODELS"
    fi
    
    # Trim leading spaces
    datasets=$(echo $datasets | xargs)
    models=$(echo $models | xargs)
    
    if [ -z "$datasets" ] || [ -z "$models" ]; then
        echo "Error: Must specify datasets and models (use --datasets/--models or --all-seg-datasets/--all-seg-models etc.)"
        exit 1
    fi
    
    # Create logs directory
    mkdir -p logs
    
    echo "LSF Batch Job Submission"
    echo "========================"
    echo "Datasets:  $datasets"
    echo "Models:    $models"
    if [ -n "$std_strategy" ]; then
        echo "Strategy:  $strategy + $std_strategy"
    else
        echo "Strategy:  $strategy"
    fi
    echo "Ratio:     $ratio"
    echo "Queue:     $queue"
    echo "GPU mem:   $gpu_mem"
    echo "GPU mode:  $gpu_mode"
    echo "CPUs:      $num_cpus"
    echo ""
    
    local job_count=0
    for dataset in $datasets; do
        for model in $models; do
            # Determine domain filter variants to process
            local domain_variants=("")
            if [ "$with_domain_variants" = true ]; then
                domain_variants=("" "clear_day")
            elif [ -n "$domain_filter" ]; then
                domain_variants=("$domain_filter")
            fi
            
            for variant_filter in "${domain_variants[@]}"; do
                job_count=$((job_count + 1))
                
                # Build job name
                local job_name="prove_${dataset}_${model}_${strategy}"
                if [ -n "$std_strategy" ]; then
                    job_name="${job_name}+${std_strategy}"
                fi
                if [ -n "$variant_filter" ]; then
                    job_name="${job_name}_${variant_filter}"
                fi            
                if [[ "$ratio" != "1.0" ]]; then
                    local ratio_int=$(echo "$ratio * 100" | bc | cut -d. -f1)
                    job_name="${job_name}_r${ratio_int}"
                fi
                
                # Build training command
                local train_cmd="./train_unified.sh single --dataset $dataset --model $model --strategy $strategy --ratio $ratio"
                
                if [ -n "$std_strategy" ]; then
                    train_cmd="$train_cmd --std-strategy $std_strategy"
                fi
                if [ -n "$variant_filter" ]; then
                    train_cmd="$train_cmd --domain-filter $variant_filter"
                fi
                if [ -n "$cache_dir" ]; then
                    train_cmd="$train_cmd --cache-dir $cache_dir"
                fi
                if [ "$no_early_stop" = true ]; then
                    train_cmd="$train_cmd --no-early-stop"
                fi
                if [ -n "$early_stop_patience" ]; then
                    train_cmd="$train_cmd --early-stop-patience $early_stop_patience"
                fi
            
            # Build bsub command
            local bsub_cmd="bsub -gpu \"num=1:mode=${gpu_mode}:gmem=${gpu_mem}\" \
                -q ${queue} \
                -R \"span[hosts=1]\" \
                -n ${num_cpus} \
                -oo \"logs/${job_name}_%J.log\" \
                -eo \"logs/${job_name}_%J.err\" \
                -L /bin/bash \
                -J \"${job_name}\" \
                \"${train_cmd}\""
            
            if [ "$dry_run" = true ]; then
                echo "[$job_count] [DRY RUN] $job_name"
                echo "    Command: $train_cmd"
            else
                echo "[$job_count] Submitting: $job_name"
                eval $bsub_cmd
            fi
            done
        done
    done
    
    echo ""
    if [ "$dry_run" = true ]; then
        echo "[DRY RUN] Would submit $job_count jobs"
    else
        echo "Submitted $job_count jobs to LSF"
    fi
}

cmd_submit() {
    # Submit job to LSF cluster
    local dataset=""
    local model=""
    local strategy="baseline"
    local std_strategy=""
    local ratio="1.0"
    local domain_filter=""
    local cache_dir=""
    local no_early_stop=true
    local early_stop_patience=""
    local queue="BatchGPU"
    local gpu_mem="12G"
    local gpu_mode="shared"
    local num_cpus="8"
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --std-strategy) std_strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --domain-filter) domain_filter="$2"; shift 2 ;;
            --cache-dir) cache_dir="$2"; shift 2 ;;
            --no-early-stop) no_early_stop=true; shift ;;
            --early-stop-patience) early_stop_patience="$2"; shift 2 ;;
            --queue) queue="$2"; shift 2 ;;
            --gpu-mem) gpu_mem="$2"; shift 2 ;;
            --gpu-mode) gpu_mode="$2"; shift 2 ;;
            --num-cpus) num_cpus="$2"; shift 2 ;;
            --dry-run) dry_run=true; shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "Error: --dataset and --model are required"
        exit 1
    fi
    
    # Build job name
    local job_name="prove_${dataset}_${model}_${strategy}"
    if [ -n "$std_strategy" ]; then
        job_name="${job_name}+${std_strategy}"
    fi
    if [ -n "$domain_filter" ]; then
        job_name="${job_name}_${domain_filter}"
    fi
    if [[ "$ratio" != "1.0" ]]; then
        local ratio_int=$(echo "$ratio * 100" | bc | cut -d. -f1)
        job_name="${job_name}_r${ratio_int}"
    fi
    
    # Build training command
    local train_cmd="./train_unified.sh single --dataset $dataset --model $model --strategy $strategy --ratio $ratio"
    
    if [ -n "$std_strategy" ]; then
        train_cmd="$train_cmd --std-strategy $std_strategy"
    fi
    if [ -n "$domain_filter" ]; then
        train_cmd="$train_cmd --domain-filter $domain_filter"
    fi
    if [ -n "$cache_dir" ]; then
        train_cmd="$train_cmd --cache-dir $cache_dir"
    fi
    if [ "$no_early_stop" = true ]; then
        train_cmd="$train_cmd --no-early-stop"
    fi
    if [ -n "$early_stop_patience" ]; then
        train_cmd="$train_cmd --early-stop-patience $early_stop_patience"
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Build bsub command
    local bsub_cmd="bsub -gpu \"num=1:mode=${gpu_mode}:gmem=${gpu_mem}\" \
        -q ${queue} \
        -R \"span[hosts=1]\" \
        -n ${num_cpus} \
        -oo \"logs/${job_name}_%J.log\" \
        -eo \"logs/${job_name}_%J.err\" \
        -L /bin/bash \
        -J \"${job_name}\" \
        \"${train_cmd}\""
    
    echo "LSF Job Submission"
    echo "=================="
    echo "Job name:  $job_name"
    echo "Dataset:   $dataset"
    echo "Model:     $model"
    if [ -n "$std_strategy" ]; then
        echo "Strategy:  $strategy + $std_strategy"
    else
        echo "Strategy:  $strategy"
    fi
    echo "Ratio:     $ratio"
    echo "Queue:     $queue"
    echo "GPU mem:   $gpu_mem"
    echo "GPU mode:  $gpu_mode"
    echo "CPUs:      $num_cpus"
    echo ""
    echo "Command:   $train_cmd"
    echo ""
    
    if [ "$dry_run" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "$bsub_cmd"
    else
        echo "Submitting job..."
        eval $bsub_cmd
    fi
}

# ============================================================================
# Main
# ============================================================================

case "${1:-help}" in
    single)
        shift
        cmd_single "$@"
        ;;
    single-multi)
        shift
        cmd_single_multi "$@"
        ;;
    batch)
        shift
        cmd_batch "$@"
        ;;
    submit)
        shift
        cmd_submit "$@"
        ;;
    submit-batch)
        shift
        cmd_submit_batch "$@"
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

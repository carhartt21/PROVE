#!/bin/bash
# PROVE Unified Testing - Test Script
#
# This script provides a unified interface for testing trained models
# using checkpoints created by the training pipeline.
#
# Analogous to train_unified.sh but for evaluation/testing.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

# Default work directory root (same as training)
DEFAULT_WEIGHTS_ROOT="${PROVE_WEIGHTS_ROOT:-/scratch/aaa_exchange/AWARE/WEIGHTS}"

# Available datasets
SEGMENTATION_DATASETS="BDD10k IDD-AW MapillaryVistas OUTSIDE15k"
DETECTION_DATASETS="BDD100k"

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

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Unified Testing Script"
    echo "============================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  single              Test a single checkpoint"
    echo "  single-multi        Test checkpoint trained on multiple datasets"
    echo "  detailed            Fine-grained test with per-domain and per-class results"
    echo "  batch               Test multiple checkpoints"
    echo "  detailed-batch      Fine-grained test for multiple checkpoints"
    echo "  submit              Submit single test job to LSF cluster"
    echo "  submit-batch        Submit multiple test jobs to LSF cluster"
    echo "  submit-detailed     Submit single detailed test job to LSF cluster"
    echo "  submit-detailed-batch Submit multiple detailed test jobs to LSF cluster"
    echo "  find                Find available checkpoints"
    echo "  results             Show test results summary"
    echo "  list                List available options"
    echo "  help                Show this help message"
    echo ""
    echo "Multi-Dataset Test Options (single-multi command):"
    echo "  --datasets <names...>     List of datasets the model was trained on"
    echo "  --model <name>            Model name"
    echo "  --strategy <name>         Augmentation strategy used during training"
    echo "  --eval-dataset <name>     Dataset to evaluate on (default: first in --datasets)"
    echo ""
    echo "Single Test Options:"
    echo "  --dataset <name>          Dataset name (ACDC, BDD10k, BDD100k, IDD-AW, MapillaryVistas, OUTSIDE15k)"
    echo "  --model <name>            Model name (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5, etc.)"
    echo "  --strategy <name>         Augmentation strategy used during training"
    echo "  --ratio <ratio>           Real-to-generated ratio used during training (default: 1.0)"
    echo "  --checkpoint <path>       Path to checkpoint file (auto-detected if not specified)"
    echo "  --checkpoint-type <type>  Checkpoint type: best, latest (default: best)"
    echo "  --work-dir <path>         Work directory root (default: \$PROVE_WEIGHTS_ROOT)"
    echo "  --output-dir <path>       Output directory for results (default: work_dir/test_results)"
    echo "  --test-split <split>      Test split to evaluate: val, test (default: test)"
    echo "  --show                    Visualize results"
    echo "  --show-dir <path>         Directory to save visualizations"
    echo ""
    echo "Detailed Test Options (per-domain, per-class):"
    echo "  --data-root <path>        Data root directory (default: \$PROVE_DATA_ROOT)"
    echo ""
    echo "Batch Test Options:"
    echo "  --datasets <names...>     List of datasets"
    echo "  --models <names...>       List of models"
    echo "  --strategies <names...>   List of strategies"
    echo "  --all-seg-datasets        Use all segmentation datasets"
    echo "  --all-det-datasets        Use all detection datasets"
    echo "  --all-seg-models          Use all segmentation models"
    echo "  --all-det-models          Use all detection models"
    echo "  --dry-run                 Show commands without executing"
    echo ""
    echo "LSF Submit Options:"
    echo "  --queue <name>            LSF queue name (default: BatchGPU)"
    echo "  --gpu-mem <size>          GPU memory requirement (default: 16G)"
    echo "  --gpu-mode <mode>         GPU mode: shared or exclusive_process (default: shared)"
    echo "  --num-cpus <n>            Number of CPUs per job (default: 4)"
    echo ""
    echo "Examples:"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 single --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --ratio 0.5"
    echo "  $0 single --checkpoint /path/to/checkpoint.pth --dataset ACDC --model deeplabv3plus_r50"
    echo "  $0 detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 detailed-batch --all-seg-datasets --all-seg-models --strategy baseline --dry-run"
    echo "  $0 batch --all-seg-datasets --all-seg-models --strategy baseline --dry-run"
    echo "  $0 find --dataset ACDC --model deeplabv3plus_r50"
    echo "  $0 find --all"
    echo "  $0 submit --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 submit-detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline"
    echo "  $0 single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50"
    echo "  $0 results --dataset ACDC"
    echo ""
}

# Get checkpoint path based on training configuration
get_checkpoint_path() {
    local work_dir="$1"
    local dataset="$2"
    local model="$3"
    local strategy="$4"
    local ratio="$5"
    local checkpoint_type="${6:-best}"
    
    # Build work directory path (matches train_unified.sh structure)
    local ratio_str=""
    if [[ "$ratio" != "1.0" ]]; then
        ratio_str="_ratio${ratio//.p/p}"
        ratio_str="${ratio_str//./_}"
    fi
    
    local checkpoint_dir="${work_dir}/${strategy}/${dataset,,}/${model}${ratio_str}"
    
    if [ "$checkpoint_type" = "best" ]; then
        # Find best checkpoint (e.g., best_mIoU_iter_*.pth)
        local best_ckpt=$(find "$checkpoint_dir" -maxdepth 1 -name "best_*.pth" 2>/dev/null | head -n 1)
        if [ -n "$best_ckpt" ]; then
            echo "$best_ckpt"
            return
        fi
        # Fall back to latest iteration checkpoint (highest iter number)
        local latest_iter=$(find "$checkpoint_dir" -maxdepth 1 -name "iter_*.pth" 2>/dev/null | sort -t_ -k2 -n | tail -n 1)
        if [ -n "$latest_iter" ]; then
            echo "$latest_iter"
            return
        fi
        # Fall back to latest.pth
        if [ -f "${checkpoint_dir}/latest.pth" ]; then
            echo "${checkpoint_dir}/latest.pth"
            return
        fi
    else
        # Latest checkpoint requested
        if [ -f "${checkpoint_dir}/latest.pth" ]; then
            echo "${checkpoint_dir}/latest.pth"
            return
        fi
        # Fall back to highest iteration
        local latest_iter=$(find "$checkpoint_dir" -maxdepth 1 -name "iter_*.pth" 2>/dev/null | sort -t_ -k2 -n | tail -n 1)
        if [ -n "$latest_iter" ]; then
            echo "$latest_iter"
            return
        fi
    fi
    
    # Return empty if nothing found
    echo ""
}

# Get config path for a checkpoint
get_config_path() {
    local work_dir="$1"
    local dataset="$2"
    local model="$3"
    local strategy="$4"
    local ratio="$5"
    
    local ratio_str=""
    if [[ "$ratio" != "1.0" ]]; then
        ratio_str="_ratio${ratio//.p/p}"
        ratio_str="${ratio_str//./_}"
    fi
    
    local config_dir="${work_dir}/${strategy}/${dataset,,}/${model}${ratio_str}/configs"
    local config_file=$(find "$config_dir" -name "*.py" 2>/dev/null | head -n 1)
    
    echo "$config_file"
}

# Detect task type based on model name
get_task_type() {
    local model="$1"
    
    case "$model" in
        deeplabv3plus_*|pspnet_*|segformer_*)
            echo "segmentation"
            ;;
        faster_rcnn_*|yolox_*|rtmdet_*)
            echo "detection"
            ;;
        *)
            echo "segmentation"
            ;;
    esac
}

# Run test for segmentation model
run_segmentation_test() {
    local config_path="$1"
    local checkpoint_path="$2"
    local output_dir="$3"
    local test_split="$4"
    local show="$5"
    local show_dir="$6"
    
    echo "Running segmentation test..."
    echo "  Config: $config_path"
    echo "  Checkpoint: $checkpoint_path"
    echo "  Output: $output_dir"
    
    mkdir -p "$output_dir"
    
    # Use MMEngine Runner for testing (MMSeg 1.x compatible)
    mamba run -n prove python -c "
import os
import sys
import json

# Add project root to path and import custom transforms
sys.path.insert(0, '$SCRIPT_DIR')
import custom_transforms  # Registers ReduceToSingleChannel, CityscapesLabelIdToTrainId, FWIoUMetric

from mmengine.config import Config
from mmengine.runner import Runner

# Load config
cfg = Config.fromfile('$config_path')

# Override for testing
cfg.work_dir = '$output_dir'
cfg.load_from = '$checkpoint_path'

# Ensure test evaluator is configured
if 'test_evaluator' not in cfg:
    cfg.test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])

# Build runner and run test
runner = Runner.from_cfg(cfg)
metrics = runner.test()

# Convert any tensor values to Python types for JSON serialization
import torch
def to_python_type(val):
    if isinstance(val, torch.Tensor):
        return val.item() if val.numel() == 1 else val.tolist()
    return val

metrics_serializable = {k: to_python_type(v) for k, v in metrics.items()}

# Save results
results_file = os.path.join('$output_dir', 'metrics.json')
with open(results_file, 'w') as f:
    json.dump(metrics_serializable, f, indent=2)
print(f'Results saved to: {results_file}')

# Print summary
print('\\n=== Test Results ===')
for key, value in metrics_serializable.items():
    if isinstance(value, float):
        print(f'{key}: {value:.4f}')
    else:
        print(f'{key}: {value}')
"
}

# Run test for detection model
run_detection_test() {
    local config_path="$1"
    local checkpoint_path="$2"
    local output_dir="$3"
    local test_split="$4"
    local show="$5"
    local show_dir="$6"
    
    echo "Running detection test..."
    echo "  Config: $config_path"
    echo "  Checkpoint: $checkpoint_path"
    echo "  Output: $output_dir"
    
    mkdir -p "$output_dir"
    
    # Use MMEngine Runner for testing (MMDet 3.x compatible)
    mamba run -n prove python -c "
import os
import sys
import json

# Add project root to path and import custom transforms
sys.path.insert(0, '$SCRIPT_DIR')
import custom_transforms  # Registers ReduceToSingleChannel, CityscapesLabelIdToTrainId, FWIoUMetric

from mmengine.config import Config
from mmengine.runner import Runner

# Load config
cfg = Config.fromfile('$config_path')

# Override for testing
cfg.work_dir = '$output_dir'
cfg.load_from = '$checkpoint_path'

# Ensure test evaluator is configured
if 'test_evaluator' not in cfg:
    cfg.test_evaluator = dict(type='CocoMetric', metric='bbox')

# Build runner and run test
runner = Runner.from_cfg(cfg)
metrics = runner.test()

# Convert any tensor values to Python types for JSON serialization
import torch
def to_python_type(val):
    if isinstance(val, torch.Tensor):
        return val.item() if val.numel() == 1 else val.tolist()
    return val

metrics_serializable = {k: to_python_type(v) for k, v in metrics.items()}

# Save results
results_file = os.path.join('$output_dir', 'metrics.json')
with open(results_file, 'w') as f:
    json.dump(metrics_serializable, f, indent=2)
print(f'Results saved to: {results_file}')

# Print summary
print('\\n=== Test Results ===')
for key, value in metrics_serializable.items():
    if isinstance(value, float):
        print(f'{key}: {value:.4f}')
    else:
        print(f'{key}: {value}')
"
}

# ============================================================================
# Commands
# ============================================================================

cmd_single() {
    local dataset=""
    local model=""
    local strategy="baseline"
    local ratio="1.0"
    local checkpoint=""
    local checkpoint_type="best"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local output_dir=""
    local test_split="test"
    local show=false
    local show_dir=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --checkpoint) checkpoint="$2"; shift 2 ;;
            --checkpoint-type) checkpoint_type="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --output-dir) output_dir="$2"; shift 2 ;;
            --test-split) test_split="$2"; shift 2 ;;
            --show) show=true; shift ;;
            --show-dir) show_dir="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "Error: --dataset and --model are required"
        exit 1
    fi
    
    # Get checkpoint path if not specified
    if [ -z "$checkpoint" ]; then
        checkpoint=$(get_checkpoint_path "$work_dir" "$dataset" "$model" "$strategy" "$ratio" "$checkpoint_type")
    fi
    
    if [ ! -f "$checkpoint" ]; then
        echo "Error: Checkpoint not found: $checkpoint"
        echo "Use '$0 find' to list available checkpoints"
        exit 1
    fi
    
    # Get config path
    local config_path=$(get_config_path "$work_dir" "$dataset" "$model" "$strategy" "$ratio")
    
    if [ -z "$config_path" ] || [ ! -f "$config_path" ]; then
        echo "Warning: Config not found in work_dir, will generate config dynamically"
        # Generate config dynamically
        config_path="/tmp/prove_test_config_$$.py"
        mamba run -n prove python unified_training_config.py \
            --dataset "$dataset" \
            --model "$model" \
            --strategy "$strategy" \
            --real-gen-ratio "$ratio" \
            --save-config "$config_path"
    fi
    
    # Set output directory
    if [ -z "$output_dir" ]; then
        output_dir="$(dirname $checkpoint)/test_results/${test_split}"
    fi
    
    echo "PROVE Unified Testing"
    echo "====================="
    echo "Dataset:     $dataset"
    echo "Model:       $model"
    echo "Strategy:    $strategy"
    echo "Ratio:       $ratio"
    echo "Checkpoint:  $checkpoint"
    echo "Config:      $config_path"
    echo "Output:      $output_dir"
    echo "Test split:  $test_split"
    echo ""
    
    # Detect task type and run appropriate test
    local task_type=$(get_task_type "$model")
    
    if [ "$task_type" = "segmentation" ]; then
        run_segmentation_test "$config_path" "$checkpoint" "$output_dir" "$test_split" "$show" "$show_dir"
    else
        run_detection_test "$config_path" "$checkpoint" "$output_dir" "$test_split" "$show" "$show_dir"
    fi
    
    echo ""
    echo "Testing complete. Results saved to: $output_dir"
}

# Multi-dataset test command
cmd_single_multi() {
    local datasets=""
    local model=""
    local strategy="baseline"
    local ratio="1.0"
    local checkpoint=""
    local checkpoint_type="best"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local output_dir=""
    local test_split="test"
    local eval_dataset=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --datasets) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    datasets="$datasets $1"
                    shift
                done
                ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --checkpoint) checkpoint="$2"; shift 2 ;;
            --checkpoint-type) checkpoint_type="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --output-dir) output_dir="$2"; shift 2 ;;
            --test-split) test_split="$2"; shift 2 ;;
            --eval-dataset) eval_dataset="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    # Trim leading spaces
    datasets=$(echo $datasets | xargs)
    
    if [ -z "$datasets" ] || [ -z "$model" ]; then
        echo "Error: --datasets and --model are required"
        exit 1
    fi
    
    # Build multi-dataset directory name (e.g., multi_acdc+mapillaryvistas)
    local datasets_lower=$(echo "$datasets" | tr ' ' '+' | tr '[:upper:]' '[:lower:]')
    local datasets_dir="multi_${datasets_lower}"
    
    # Build ratio string for directory
    local ratio_str=""
    if [[ "$ratio" != "1.0" ]]; then
        ratio_str="_ratio${ratio//.p/p}"
        ratio_str="${ratio_str//./_}"
    fi
    
    local checkpoint_dir="${work_dir}/${strategy}/${datasets_dir}/${model}${ratio_str}"
    
    # Auto-detect checkpoint if not specified
    if [ -z "$checkpoint" ]; then
        if [ "$checkpoint_type" = "best" ]; then
            checkpoint=$(find "$checkpoint_dir" -maxdepth 1 -name "best_*.pth" 2>/dev/null | head -n 1)
        fi
        if [ -z "$checkpoint" ]; then
            checkpoint=$(find "$checkpoint_dir" -maxdepth 1 -name "iter_*.pth" 2>/dev/null | sort -t_ -k2 -n | tail -n 1)
        fi
        if [ -z "$checkpoint" ] && [ -f "${checkpoint_dir}/latest.pth" ]; then
            checkpoint="${checkpoint_dir}/latest.pth"
        fi
    fi
    
    if [ ! -f "$checkpoint" ]; then
        echo "Error: Checkpoint not found in: $checkpoint_dir"
        echo "Use '$0 find' to list available checkpoints"
        exit 1
    fi
    
    # Get config path
    local config_dir="${checkpoint_dir}/configs"
    local config_path="${config_dir}/training_config.py"
    
    if [ ! -f "$config_path" ]; then
        echo "Warning: Config not found at $config_path"
        # Try to generate config
        echo "Generating config dynamically..."
        mamba run -n prove python -c "
from unified_training_config import UnifiedTrainingConfig
config = UnifiedTrainingConfig()
datasets = '$datasets'.split()
cfg = config.build_multi_dataset(datasets=datasets, model='$model', strategy='$strategy', real_gen_ratio=$ratio)
config.save_config(cfg, '$config_path')
print('Config generated successfully')
"
    fi
    
    # Set output directory
    if [ -z "$output_dir" ]; then
        output_dir="$(dirname $checkpoint)/test_results/${test_split}"
    fi
    
    # Use first dataset for evaluation if not specified
    if [ -z "$eval_dataset" ]; then
        eval_dataset=$(echo $datasets | awk '{print $1}')
    fi
    
    echo "PROVE Multi-Dataset Testing"
    echo "============================"
    echo ""
    echo "Datasets:    $datasets"
    echo "Model:       $model"
    echo "Strategy:    $strategy"
    echo "Ratio:       $ratio"
    echo "Eval on:     $eval_dataset"
    echo "Checkpoint:  $checkpoint"
    echo "Config:      $config_path"
    echo "Output:      $output_dir"
    echo ""
    
    # Run segmentation test
    run_segmentation_test "$config_path" "$checkpoint" "$output_dir" "$test_split" "false" ""
    
    echo ""
    echo "Testing complete. Results saved to: $output_dir"
}

# Detailed test command (per-domain, per-class)
cmd_detailed() {
    local dataset=""
    local model=""
    local strategy="baseline"
    local ratio="1.0"
    local checkpoint=""
    local checkpoint_type="best"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local output_dir=""
    local test_split="test"
    local data_root="${PROVE_DATA_ROOT:-/scratch/aaa_exchange/AWARE/FINAL_SPLITS}"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --checkpoint) checkpoint="$2"; shift 2 ;;
            --checkpoint-type) checkpoint_type="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --output-dir) output_dir="$2"; shift 2 ;;
            --test-split) test_split="$2"; shift 2 ;;
            --data-root) data_root="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    # Validate required options
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "Error: --dataset and --model are required for detailed test"
        exit 1
    fi
    
    # Get checkpoint path if not specified
    if [ -z "$checkpoint" ]; then
        checkpoint=$(get_checkpoint_path "$work_dir" "$dataset" "$model" "$strategy" "$ratio" "$checkpoint_type")
        if [ -z "$checkpoint" ] || [ ! -f "$checkpoint" ]; then
            echo "Error: Could not find checkpoint for $dataset/$model/$strategy"
            echo "Searched in: $work_dir"
            exit 1
        fi
    fi
    
    # Get config path
    local config_dir=$(dirname "$checkpoint")/configs
    local config_path="$config_dir/training_config.py"
    
    if [ ! -f "$config_path" ]; then
        echo "Error: Config file not found at: $config_path"
        exit 1
    fi
    
    # Default output directory
    if [ -z "$output_dir" ]; then
        output_dir="$(dirname "$checkpoint")/test_results_detailed"
    fi
    
    echo "PROVE Fine-Grained Testing"
    echo "=========================="
    echo ""
    echo "Dataset:     $dataset"
    echo "Model:       $model"
    echo "Strategy:    $strategy"
    echo "Ratio:       $ratio"
    echo "Checkpoint:  $checkpoint"
    echo "Config:      $config_path"
    echo "Output:      $output_dir"
    echo "Test split:  $test_split"
    echo "Data root:   $data_root"
    echo ""
    
    # Run fine-grained test
    mamba run -n prove python "$PROJECT_ROOT/fine_grained_test.py" \
        --config "$config_path" \
        --checkpoint "$checkpoint" \
        --output-dir "$output_dir" \
        --dataset "$dataset" \
        --data-root "$data_root" \
        --test-split "$test_split"
    
    echo ""
    echo "Fine-grained testing complete. Results saved to: $output_dir"
}

cmd_detailed_batch() {
    local datasets=""
    local models=""
    local strategies=""
    local ratio="1.0"
    local checkpoint_type="best"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local test_split="test"
    local data_root="${PROVE_DATA_ROOT:-/scratch/aaa_exchange/AWARE/FINAL_SPLITS}"
    local all_seg_datasets=false
    local all_det_datasets=false
    local all_seg_models=false
    local all_det_models=false
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) datasets="$datasets $2"; shift 2 ;;
            --datasets) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    datasets="$datasets $1"
                    shift
                done
                ;;
            --model) models="$models $2"; shift 2 ;;
            --models) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    models="$models $1"
                    shift
                done
                ;;
            --strategies) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    strategies="$strategies $1"
                    shift
                done
                ;;
            --strategy) strategies="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --checkpoint-type) checkpoint_type="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --test-split) test_split="$2"; shift 2 ;;
            --data-root) data_root="$2"; shift 2 ;;
            --all-seg-datasets) all_seg_datasets=true; shift ;;
            --all-det-datasets) all_det_datasets=true; shift ;;
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
    if [ "$all_seg_models" = true ]; then
        models="$models $SEGMENTATION_MODELS"
    fi
    if [ "$all_det_models" = true ]; then
        models="$models $DETECTION_MODELS"
    fi
    
    # Set default strategy if not specified
    if [ -z "$strategies" ]; then
        strategies="baseline"
    fi
    
    # Trim leading spaces
    datasets=$(echo $datasets | xargs)
    models=$(echo $models | xargs)
    strategies=$(echo $strategies | xargs)
    
    if [ -z "$datasets" ] || [ -z "$models" ]; then
        echo "Error: Must specify datasets and models"
        exit 1
    fi
    
    echo "PROVE Batch Detailed Testing"
    echo "============================"
    echo "Datasets:   $datasets"
    echo "Models:     $models"
    echo "Strategies: $strategies"
    echo "Ratio:      $ratio"
    echo ""
    
    local test_count=0
    local success_count=0
    local fail_count=0
    
    for dataset in $datasets; do
        for model in $models; do
            for strategy in $strategies; do
                test_count=$((test_count + 1))
                
                local checkpoint=$(get_checkpoint_path "$work_dir" "$dataset" "$model" "$strategy" "$ratio" "$checkpoint_type")
                
                if [ "$dry_run" = true ]; then
                    if [ -f "$checkpoint" ]; then
                        echo "[$test_count] [DRY RUN] Would run detailed test: $dataset / $model / $strategy"
                        echo "    Checkpoint: $checkpoint"
                    else
                        echo "[$test_count] [SKIP] Checkpoint not found: $dataset / $model / $strategy"
                    fi
                else
                    if [ -f "$checkpoint" ]; then
                        echo ""
                        echo "[$test_count] Detailed testing: $dataset / $model / $strategy"
                        if cmd_detailed --dataset "$dataset" --model "$model" --strategy "$strategy" \
                                        --ratio "$ratio" --checkpoint "$checkpoint" \
                                        --work-dir "$work_dir" --test-split "$test_split" \
                                        --data-root "$data_root"; then
                            success_count=$((success_count + 1))
                        else
                            fail_count=$((fail_count + 1))
                        fi
                    else
                        echo "[$test_count] [SKIP] Checkpoint not found: $dataset / $model / $strategy"
                        fail_count=$((fail_count + 1))
                    fi
                fi
            done
        done
    done
    
    echo ""
    echo "Batch detailed testing complete."
    if [ "$dry_run" = false ]; then
        echo "  Success: $success_count"
        echo "  Failed/Skipped: $fail_count"
        echo "  Total: $test_count"
    fi
}

cmd_batch() {
    local datasets=""
    local models=""
    local strategies=""
    local ratio="1.0"
    local checkpoint_type="best"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local output_dir=""
    local test_split="test"
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
            --strategies) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    strategies="$strategies $1"
                    shift
                done
                ;;
            --strategy) strategies="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --checkpoint-type) checkpoint_type="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --output-dir) output_dir="$2"; shift 2 ;;
            --test-split) test_split="$2"; shift 2 ;;
            --all-seg-datasets) all_seg_datasets=true; shift ;;
            --all-det-datasets) all_det_datasets=true; shift ;;
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
    if [ "$all_seg_models" = true ]; then
        models="$models $SEGMENTATION_MODELS"
    fi
    if [ "$all_det_models" = true ]; then
        models="$models $DETECTION_MODELS"
    fi
    
    # Set default strategy if not specified
    if [ -z "$strategies" ]; then
        strategies="baseline"
    fi
    
    # Trim leading spaces
    datasets=$(echo $datasets | xargs)
    models=$(echo $models | xargs)
    strategies=$(echo $strategies | xargs)
    
    if [ -z "$datasets" ] || [ -z "$models" ]; then
        echo "Error: Must specify datasets and models"
        exit 1
    fi
    
    echo "PROVE Batch Testing"
    echo "==================="
    echo "Datasets:   $datasets"
    echo "Models:     $models"
    echo "Strategies: $strategies"
    echo "Ratio:      $ratio"
    echo ""
    
    local test_count=0
    local success_count=0
    local fail_count=0
    
    for dataset in $datasets; do
        for model in $models; do
            for strategy in $strategies; do
                test_count=$((test_count + 1))
                
                local checkpoint=$(get_checkpoint_path "$work_dir" "$dataset" "$model" "$strategy" "$ratio" "$checkpoint_type")
                
                if [ "$dry_run" = true ]; then
                    if [ -f "$checkpoint" ]; then
                        echo "[$test_count] [DRY RUN] Would test: $dataset / $model / $strategy"
                        echo "    Checkpoint: $checkpoint"
                    else
                        echo "[$test_count] [SKIP] Checkpoint not found: $dataset / $model / $strategy"
                    fi
                else
                    if [ -f "$checkpoint" ]; then
                        echo ""
                        echo "[$test_count] Testing: $dataset / $model / $strategy"
                        if cmd_single --dataset "$dataset" --model "$model" --strategy "$strategy" \
                                      --ratio "$ratio" --checkpoint "$checkpoint" \
                                      --work-dir "$work_dir" --test-split "$test_split"; then
                            success_count=$((success_count + 1))
                        else
                            fail_count=$((fail_count + 1))
                        fi
                    else
                        echo "[$test_count] [SKIP] Checkpoint not found: $dataset / $model / $strategy"
                        fail_count=$((fail_count + 1))
                    fi
                fi
            done
        done
    done
    
    echo ""
    echo "Batch testing complete."
    if [ "$dry_run" = false ]; then
        echo "  Success: $success_count"
        echo "  Failed/Skipped: $fail_count"
        echo "  Total: $test_count"
    fi
}

cmd_find() {
    local dataset=""
    local model=""
    local strategy=""
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local show_all=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --all) show_all=true; shift ;;
            *) shift ;;
        esac
    done
    
    echo "Searching for checkpoints in: $work_dir"
    echo ""
    
    local search_path="$work_dir"
    
    if [ -n "$strategy" ]; then
        search_path="$search_path/$strategy"
    fi
    if [ -n "$dataset" ]; then
        search_path="$search_path/${dataset,,}"
    fi
    if [ -n "$model" ]; then
        search_path="$search_path/$model*"
    fi
    
    if [ "$show_all" = true ]; then
        search_path="$work_dir"
    fi
    
    echo "Available checkpoints:"
    echo "======================"
    
    # Find all .pth files
    local checkpoints=$(find $search_path -name "*.pth" 2>/dev/null | sort)
    
    if [ -z "$checkpoints" ]; then
        echo "No checkpoints found."
        echo ""
        echo "Make sure training has completed and checkpoints exist in:"
        echo "  $work_dir/{strategy}/{dataset}/{model}/"
    else
        local count=0
        for ckpt in $checkpoints; do
            count=$((count + 1))
            # Extract info from path
            local rel_path="${ckpt#$work_dir/}"
            local size=$(du -h "$ckpt" | cut -f1)
            echo "[$count] $rel_path ($size)"
        done
        echo ""
        echo "Found $count checkpoint(s)"
    fi
}

cmd_results() {
    local dataset=""
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    echo "Test Results Summary"
    echo "===================="
    
    local search_path="$work_dir"
    if [ -n "$dataset" ]; then
        search_path="$work_dir/*/${dataset,,}"
    fi
    
    # Find all results.pkl files
    local results=$(find $search_path -path "*/test_results/*" -name "*.pkl" 2>/dev/null | sort)
    
    if [ -z "$results" ]; then
        echo "No test results found."
        echo "Run testing first with: $0 single --dataset <dataset> --model <model>"
    else
        for result in $results; do
            echo ""
            echo "Result: ${result#$work_dir/}"
            # Try to parse and display key metrics
            mamba run -n prove python -c "
import pickle
import os

try:
    with open('$result', 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        if 'mIoU' in data:
            print(f'  mIoU: {data[\"mIoU\"]:.2f}%')
        if 'mDice' in data:
            print(f'  mDice: {data[\"mDice\"]:.2f}%')
        if 'fwIoU' in data:
            print(f'  fwIoU: {data[\"fwIoU\"]:.2f}%')
        if 'bbox_mAP' in data:
            print(f'  bbox_mAP: {data[\"bbox_mAP\"]:.4f}')
    else:
        print('  (Results format not recognized)')
except Exception as e:
    print(f'  Error reading results: {e}')
" 2>/dev/null || echo "  (Could not parse results)"
        done
    fi
}

cmd_submit() {
    local dataset=""
    local model=""
    local strategy="baseline"
    local ratio="1.0"
    local checkpoint_type="best"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
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
            --ratio) ratio="$2"; shift 2 ;;
            --checkpoint-type) checkpoint_type="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
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
    
    local job_name="prove_test_${dataset}_${model}_${strategy}"
    local test_cmd="$SCRIPT_DIR/test_unified.sh single --dataset $dataset --model $model --strategy $strategy --ratio $ratio --work-dir $work_dir"
    
    mkdir -p logs
    
    local bsub_cmd="bsub -gpu \"num=1:mode=${gpu_mode}:gmem=${gpu_mem}\" \
        -q ${queue} \
        -R \"span[hosts=1]\" \
        -n ${num_cpus} \
        -oo \"logs/${job_name}_%J.log\" \
        -eo \"logs/${job_name}_%J.err\" \
        -L /bin/bash \
        -J \"${job_name}\" \
        \"${test_cmd}\""
    
    echo "LSF Test Job Submission"
    echo "======================="
    echo "Job name:  $job_name"
    echo "Dataset:   $dataset"
    echo "Model:     $model"
    echo "Strategy:  $strategy"
    echo "Queue:     $queue"
    echo "GPU mem:   $gpu_mem"
    echo "GPU mode:  $gpu_mode"
    echo ""
    
    if [ "$dry_run" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "$bsub_cmd"
    else
        echo "Submitting job..."
        eval $bsub_cmd
    fi
}

cmd_submit_batch() {
    local datasets=""
    local models=""
    local strategies=""
    local ratio="1.0"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local queue="BatchGPU"
    local gpu_mem="12G"
    local gpu_mode="shared"
    local num_cpus="8"
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
            --strategies) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    strategies="$strategies $1"
                    shift
                done
                ;;
            --strategy) strategies="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --queue) queue="$2"; shift 2 ;;
            --gpu-mem) gpu_mem="$2"; shift 2 ;;
            --gpu-mode) gpu_mode="$2"; shift 2 ;;
            --num-cpus) num_cpus="$2"; shift 2 ;;
            --all-seg-datasets) all_seg_datasets=true; shift ;;
            --all-det-datasets) all_det_datasets=true; shift ;;
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
    if [ "$all_seg_models" = true ]; then
        models="$models $SEGMENTATION_MODELS"
    fi
    if [ "$all_det_models" = true ]; then
        models="$models $DETECTION_MODELS"
    fi
    
    if [ -z "$strategies" ]; then
        strategies="baseline"
    fi
    
    datasets=$(echo $datasets | xargs)
    models=$(echo $models | xargs)
    strategies=$(echo $strategies | xargs)
    
    if [ -z "$datasets" ] || [ -z "$models" ]; then
        echo "Error: Must specify datasets and models"
        exit 1
    fi
    
    mkdir -p logs
    
    echo "LSF Batch Test Submission"
    echo "========================="
    echo "Datasets:   $datasets"
    echo "Models:     $models"
    echo "Strategies: $strategies"
    echo "Queue:      $queue"
    echo "GPU mem:    $gpu_mem"
    echo "GPU mode:   $gpu_mode"
    echo ""
    
    local job_count=0
    for dataset in $datasets; do
        for model in $models; do
            for strategy in $strategies; do
                job_count=$((job_count + 1))
                
                local checkpoint=$(get_checkpoint_path "$work_dir" "$dataset" "$model" "$strategy" "$ratio" "best")
                
                if [ -f "$checkpoint" ] || [ "$dry_run" = true ]; then
                    local job_name="prove_test_${dataset}_${model}_${strategy}"
                    local test_cmd="$SCRIPT_DIR/test_unified.sh single --dataset $dataset --model $model --strategy $strategy --ratio $ratio --work-dir $work_dir"
                    
                    local gpu_spec="num=1"
                    if [ "$gpu_mode" = "exclusive_process" ]; then
                        gpu_spec="${gpu_spec}:mode=exclusive_process"
                    fi
                    gpu_spec="${gpu_spec}:gmem=${gpu_mem}"
                    
                    local bsub_cmd="bsub -gpu \"${gpu_spec}\" \
                        -q ${queue} \
                        -R \"span[hosts=1]\" \
                        -n ${num_cpus} \
                        -oo \"logs/${job_name}_%J.log\" \
                        -eo \"logs/${job_name}_%J.err\" \
                        -L /bin/bash \
                        -J \"${job_name}\" \
                        \"${test_cmd}\""
                    
                    if [ "$dry_run" = true ]; then
                        echo "[$job_count] [DRY RUN] $job_name"
                        echo "    Command: $test_cmd"
                    else
                        echo "[$job_count] Submitting: $job_name"
                        eval $bsub_cmd
                    fi
                else
                    echo "[$job_count] [SKIP] Checkpoint not found: $dataset / $model / $strategy"
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

cmd_submit_detailed() {
    local dataset=""
    local model=""
    local strategy="baseline"
    local ratio="1.0"
    local checkpoint_type="best"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local queue="BatchGPU"
    local gpu_mem="12G"
    local gpu_mode="shared"
    local num_cpus="4"
    local data_root="${PROVE_DATA_ROOT:-/scratch/aaa_exchange/AWARE/FINAL_SPLITS}"
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) dataset="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --strategy) strategy="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --checkpoint-type) checkpoint_type="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --queue) queue="$2"; shift 2 ;;
            --gpu-mem) gpu_mem="$2"; shift 2 ;;
            --gpu-mode) gpu_mode="$2"; shift 2 ;;
            --num-cpus) num_cpus="$2"; shift 2 ;;
            # --mode) mode="$2"; shift 2 ;;
            --data-root) data_root="$2"; shift 2 ;;
            --dry-run) dry_run=true; shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "Error: --dataset and --model are required"
        exit 1
    fi
    
    local job_name="prove_detailed_${dataset}_${model}_${strategy}"
    local test_cmd="$SCRIPT_DIR/test_unified.sh detailed --dataset $dataset --model $model --strategy $strategy --ratio $ratio --work-dir $work_dir --data-root $data_root"
    
    mkdir -p logs
    
    local gpu_spec="num=1"
    if [ "$gpu_mode" = "exclusive_process" ]; then
        gpu_spec="${gpu_spec}:mode=exclusive_process"
    fi
    gpu_spec="${gpu_spec}:gmem=${gpu_mem}"
    
    local bsub_cmd="bsub -gpu \"${gpu_spec}\" \
        -q ${queue} \
        -R \"span[hosts=1]\" \
        -n ${num_cpus} \
        -oo \"logs/${job_name}_%J.log\" \
        -eo \"logs/${job_name}_%J.err\" \
        -L /bin/bash \
        -J \"${job_name}\" \
        \"${test_cmd}\""
    
    echo "LSF Detailed Test Job Submission"
    echo "================================="
    echo "Job name:  $job_name"
    echo "Dataset:   $dataset"
    echo "Model:     $model"
    echo "Strategy:  $strategy"
    echo "Queue:     $queue"
    echo "GPU mem:   $gpu_mem"
    echo "GPU mode:  $gpu_mode"
    echo ""
    
    if [ "$dry_run" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "$bsub_cmd"
    else
        echo "Submitting job..."
        eval $bsub_cmd
    fi
}

cmd_submit_detailed_batch() {
    local datasets=""
    local models=""
    local strategies=""
    local ratio="1.0"
    local work_dir="$DEFAULT_WEIGHTS_ROOT"
    local queue="BatchGPU"
    local gpu_mem="12G"
    local gpu_mode="shared"
    local num_cpus="8"

    local data_root="${PROVE_DATA_ROOT:-/scratch/aaa_exchange/AWARE/FINAL_SPLITS}"
    local all_seg_datasets=false
    local all_det_datasets=false
    local all_seg_models=false
    local all_det_models=false
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) datasets="$datasets $2"; shift 2 ;;
            --datasets) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    datasets="$datasets $1"
                    shift
                done
                ;;
            --model) models="$models $2"; shift 2 ;;
            --models) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    models="$models $1"
                    shift
                done
                ;;
            --strategies) shift;
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    strategies="$strategies $1"
                    shift
                done
                ;;
            --strategy) strategies="$2"; shift 2 ;;
            --ratio) ratio="$2"; shift 2 ;;
            --work-dir) work_dir="$2"; shift 2 ;;
            --queue) queue="$2"; shift 2 ;;
            --gpu-mem) gpu_mem="$2"; shift 2 ;;
            --gpu-mode) gpu_mode="$2"; shift 2 ;;
            --num-cpus) num_cpus="$2"; shift 2 ;;
            # --mode) mode="$2"; shift 2 ;;
            --data-root) data_root="$2"; shift 2 ;;
            --all-seg-datasets) all_seg_datasets=true; shift ;;
            --all-det-datasets) all_det_datasets=true; shift ;;
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
    if [ "$all_seg_models" = true ]; then
        models="$models $SEGMENTATION_MODELS"
    fi
    if [ "$all_det_models" = true ]; then
        models="$models $DETECTION_MODELS"
    fi
    
    if [ -z "$strategies" ]; then
        strategies="baseline"
    fi
    
    datasets=$(echo $datasets | xargs)
    models=$(echo $models | xargs)
    strategies=$(echo $strategies | xargs)
    
    if [ -z "$datasets" ] || [ -z "$models" ]; then
        echo "Error: Must specify datasets and models"
        exit 1
    fi
    
    mkdir -p logs
    
    echo "LSF Batch Detailed Test Submission"
    echo "==================================="
    echo "Datasets:   $datasets"
    echo "Models:     $models"
    echo "Strategies: $strategies"
    echo "Mode:       $mode"
    echo "Queue:      $queue"
    echo "GPU mem:    $gpu_mem"
    echo "GPU mode:   $gpu_mode"
    echo ""
    
    local job_count=0
    for dataset in $datasets; do
        for model in $models; do
            for strategy in $strategies; do
                job_count=$((job_count + 1))
                
                local checkpoint=$(get_checkpoint_path "$work_dir" "$dataset" "$model" "$strategy" "$ratio" "best")
                
                if [ -f "$checkpoint" ] || [ "$dry_run" = true ]; then
                    local job_name="prove_detailed_${dataset}_${model}_${strategy}"
                    local test_cmd="$SCRIPT_DIR/test_unified.sh detailed --dataset $dataset --model $model --strategy $strategy --ratio $ratio --work-dir $work_dir --mode $mode --data-root $data_root"
                    
                    local gpu_spec="num=1"
                    if [ "$gpu_mode" = "exclusive_process" ]; then
                        gpu_spec="${gpu_spec}:mode=exclusive_process"
                    fi
                    gpu_spec="${gpu_spec}:gmem=${gpu_mem}"
                    
                    local bsub_cmd="bsub -gpu \"${gpu_spec}\" \
                        -q ${queue} \
                        -R \"span[hosts=1]\" \
                        -n ${num_cpus} \
                        -oo \"logs/${job_name}_%J.log\" \
                        -eo \"logs/${job_name}_%J.err\" \
                        -L /bin/bash \
                        -J \"${job_name}\" \
                        \"${test_cmd}\""
                    
                    if [ "$dry_run" = true ]; then
                        echo "[$job_count] [DRY RUN] $job_name"
                        echo "    Command: $test_cmd"
                    else
                        echo "[$job_count] Submitting: $job_name"
                        eval $bsub_cmd
                    fi
                else
                    echo "[$job_count] [SKIP] Checkpoint not found: $dataset / $model / $strategy"
                fi
            done
        done
    done
    
    echo ""
    if [ "$dry_run" = true ]; then
        echo "[DRY RUN] Would submit $job_count jobs"
    else
        echo "Submitted $job_count detailed test jobs to LSF"
    fi
}

cmd_list() {
    echo "Available Options"
    echo "================="
    echo ""
    echo "Segmentation Datasets: $SEGMENTATION_DATASETS"
    echo "Detection Datasets:    $DETECTION_DATASETS"
    echo ""
    echo "Segmentation Models:   $SEGMENTATION_MODELS"
    echo "Detection Models:      $DETECTION_MODELS"
    echo ""
    echo "Strategies:"
    echo "  $STRATEGIES" | fold -s -w 70 | sed 's/^/  /'
    echo ""
    echo "Work Directory Root: $DEFAULT_WEIGHTS_ROOT"
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
    detailed)
        shift
        cmd_detailed "$@"
        ;;
    batch)
        shift
        cmd_batch "$@"
        ;;
    detailed-batch)
        shift
        cmd_detailed_batch "$@"
        ;;
    submit)
        shift
        cmd_submit "$@"
        ;;
    submit-batch)
        shift
        cmd_submit_batch "$@"
        ;;
    submit-detailed)
        shift
        cmd_submit_detailed "$@"
        ;;
    submit-detailed-batch)
        shift
        cmd_submit_detailed_batch "$@"
        ;;
    find)
        shift
        cmd_find "$@"
        ;;
    results)
        shift
        cmd_results "$@"
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

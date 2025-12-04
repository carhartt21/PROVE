#!/bin/bash
# PROVE Training Script: gen_cycleGAN
# Auto-generated - trains all models for this augmentation strategy

set -e

STRATEGY="gen_cycleGAN"
CONFIG_DIR="./multi_model_configs/$STRATEGY"
WORK_DIR_BASE="/scratch/aaa_exchange/AWARE/WEIGHTS/$STRATEGY"
LOG_DIR="./logs/$STRATEGY"

mkdir -p "$LOG_DIR"

echo "PROVE Training: $STRATEGY"
echo "=================================================="

# Function to train a model
train_model() {
    local dataset=$1
    local model=$2
    local config="$CONFIG_DIR/${dataset^^}/${dataset,,}_${model}_config.py"
    local work_dir="$WORK_DIR_BASE/${dataset,,}/$model"
    local log_file="$LOG_DIR/${dataset,,}_${model}.log"
    
    if [ ! -f "$config" ]; then
        echo "Config not found: $config"
        return 1
    fi
    
    echo "Training: $dataset / $model"
    echo "Config: $config"
    echo "Work dir: $work_dir"
    
    mkdir -p "$work_dir"
    
    python tools/train.py "$config" --work-dir "$work_dir" > "$log_file" 2>&1
    
    echo "✓ Completed: $dataset / $model"
}

# Train all datasets

# ACDC
train_model "ACDC" "deeplabv3plus_r50"
train_model "ACDC" "pspnet_r50"
train_model "ACDC" "segformer_mit-b5"

# BDD10k
train_model "BDD10k" "deeplabv3plus_r50"
train_model "BDD10k" "pspnet_r50"
train_model "BDD10k" "segformer_mit-b5"

# BDD100k
train_model "BDD100k" "faster_rcnn_r50_fpn_1x"
train_model "BDD100k" "yolox_l"
train_model "BDD100k" "rtmdet_l"

# IDD-AW
train_model "IDD-AW" "deeplabv3plus_r50"
train_model "IDD-AW" "pspnet_r50"
train_model "IDD-AW" "segformer_mit-b5"

# MapillaryVistas
train_model "MapillaryVistas" "deeplabv3plus_r50"
train_model "MapillaryVistas" "pspnet_r50"
train_model "MapillaryVistas" "segformer_mit-b5"

# OUTSIDE15k
train_model "OUTSIDE15k" "deeplabv3plus_r50"
train_model "OUTSIDE15k" "pspnet_r50"
train_model "OUTSIDE15k" "segformer_mit-b5"

echo
echo "Training complete for strategy: $STRATEGY"
echo "Results saved in: $WORK_DIR_BASE"

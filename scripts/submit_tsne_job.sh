#!/bin/bash
#BSUB -J tsne_domain_gap
#BSUB -q BatchGPU
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /home/mima2416/repositories/PROVE/logs/tsne_domain_gap_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/tsne_domain_gap_%J.err
#BSUB -W 04:00

# t-SNE Domain Gap Visualization Job
cd /home/mima2416/repositories/PROVE

# Ensure scikit-learn is installed
echo "Checking scikit-learn installation..."
mamba run -n prove python3 -c "import sklearn" 2>/dev/null || {
    echo "Installing scikit-learn..."
    mamba run -n prove pip install scikit-learn --quiet
}

WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"
OUTPUT_DIR="/home/mima2416/repositories/PROVE/result_figures/tsne"
MODEL_TYPE="deeplabv3plus"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Baseline checkpoint
BASELINE_CKPT="${WEIGHTS_DIR}/baseline/acdc/deeplabv3plus_r50/iter_80000.pth"

echo "========================================"
echo "t-SNE Domain Gap Visualization"
echo "========================================"
echo "Baseline: $BASELINE_CKPT"
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo ""

# Top 3 generative strategies + top std + bottom performer
STRATEGIES=(
    "gen_automold"
    "gen_NST"
    "gen_SUSTechGAN"
    "std_randaugment"
    "gen_Qwen_Image_Edit"
)

for strategy in "${STRATEGIES[@]}"; do
    echo "========================================"
    echo "Processing: $strategy"
    echo "========================================"
    
    CKPT="${WEIGHTS_DIR}/${strategy}/acdc/deeplabv3plus_r50/iter_80000.pth"
    
    if [[ ! -f "$CKPT" ]]; then
        echo "WARNING: Checkpoint not found: $CKPT"
        continue
    fi
    
    OUTPUT_SUBDIR="${OUTPUT_DIR}/${strategy}"
    mkdir -p "$OUTPUT_SUBDIR"
    
    echo "Using checkpoint: $CKPT"
    echo "Output dir: $OUTPUT_SUBDIR"
    
    mamba run -n prove python3 tools/tsne_domain_gap.py \
        --checkpoint-baseline "$BASELINE_CKPT" \
        --checkpoint-augmented "$CKPT" \
        --data-root "$DATA_ROOT" \
        --model-type "$MODEL_TYPE" \
        --output "$OUTPUT_SUBDIR" \
        --num-samples 75000 \
        --max-images-per-domain 50 \
        --split test
    
    echo ""
done

echo "========================================"
echo "All visualizations complete!"
echo "Results in: $OUTPUT_DIR"
echo "========================================"

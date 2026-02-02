#!/bin/bash
#BSUB -J test_baseline_bdd10k_segformer_mit_b5_100k
#BSUB -o /home/mima2416/repositories/PROVE/logs/test_baseline_bdd10k_segformer_mit_b5_100k_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/test_baseline_bdd10k_segformer_mit_b5_100k_%J.err
#BSUB -q BatchGPU
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"

# Test baseline extended training checkpoint
# Dataset: BDD10k
# Model: segformer_mit-b5
# Iteration: 100000

source /home/mima2416/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd /home/mima2416/repositories/PROVE

echo "=========================================="
echo "Testing baseline extended checkpoint"
echo "=========================================="
echo "Dataset: BDD10k"
echo "Model: segformer_mit-b5"
echo "Iteration: 100000"
echo "Checkpoint: /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/bdd10k/segformer_mit-b5/iter_100000.pth"
echo "Output: /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/bdd10k/segformer_mit-b5/test_results_detailed/iter_100000"
echo "=========================================="

# Create output directory
mkdir -p "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/bdd10k/segformer_mit-b5/test_results_detailed/iter_100000"

python fine_grained_test.py \
    --config "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/bdd10k/segformer_mit-b5/training_config.py" \
    --checkpoint "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/bdd10k/segformer_mit-b5/iter_100000.pth" \
    --dataset BDD10k \
    --output-dir "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/bdd10k/segformer_mit-b5/test_results_detailed/iter_100000"

echo "Testing complete!"

#!/bin/bash
#BSUB -J test_baseline_iddaw_pspnet_r50_120k
#BSUB -o /home/mima2416/repositories/PROVE/logs/test_baseline_iddaw_pspnet_r50_120k_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/test_baseline_iddaw_pspnet_r50_120k_%J.err
#BSUB -q BatchGPU
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"

# Test baseline extended training checkpoint
# Dataset: IDD-AW
# Model: pspnet_r50
# Iteration: 120000

source /home/mima2416/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd /home/mima2416/repositories/PROVE

echo "=========================================="
echo "Testing baseline extended checkpoint"
echo "=========================================="
echo "Dataset: IDD-AW"
echo "Model: pspnet_r50"
echo "Iteration: 120000"
echo "Checkpoint: /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/iter_120000.pth"
echo "Output: /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/test_results_detailed/iter_120000"
echo "=========================================="

# Create output directory
mkdir -p "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/test_results_detailed/iter_120000"

python fine_grained_test.py \
    --config "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/training_config.py" \
    --checkpoint "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/iter_120000.pth" \
    --dataset IDD-AW \
    --output-dir "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/test_results_detailed/iter_120000"

echo "Testing complete!"

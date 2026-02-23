#!/bin/bash
#BSUB -J test_baseline_iddaw_segformer_mit_b5_90k
#BSUB -o ${HOME}/repositories/PROVE/logs/test_baseline_iddaw_segformer_mit_b5_90k_%J.out
#BSUB -e ${HOME}/repositories/PROVE/logs/test_baseline_iddaw_segformer_mit_b5_90k_%J.err
#BSUB -q BatchGPU
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"

# Test baseline extended training checkpoint
# Dataset: IDD-AW
# Model: segformer_mit-b5
# Iteration: 90000

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd ${HOME}/repositories/PROVE

echo "=========================================="
echo "Testing baseline extended checkpoint"
echo "=========================================="
echo "Dataset: IDD-AW"
echo "Model: segformer_mit-b5"
echo "Iteration: 90000"
echo "Checkpoint: ${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/segformer_mit-b5/iter_90000.pth"
echo "Output: ${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/segformer_mit-b5/test_results_detailed/iter_90000"
echo "=========================================="

# Create output directory
mkdir -p "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/segformer_mit-b5/test_results_detailed/iter_90000"

python fine_grained_test.py \
    --config "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/segformer_mit-b5/training_config.py" \
    --checkpoint "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/segformer_mit-b5/iter_90000.pth" \
    --dataset IDD-AW \
    --output-dir "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/segformer_mit-b5/test_results_detailed/iter_90000"

echo "Testing complete!"

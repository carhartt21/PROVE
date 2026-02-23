#!/bin/bash
#BSUB -J test_baseline_iddaw_pspnet_r50_110k
#BSUB -o ${HOME}/repositories/PROVE/logs/test_baseline_iddaw_pspnet_r50_110k_%J.out
#BSUB -e ${HOME}/repositories/PROVE/logs/test_baseline_iddaw_pspnet_r50_110k_%J.err
#BSUB -q BatchGPU
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"

# Test baseline extended training checkpoint
# Dataset: IDD-AW
# Model: pspnet_r50
# Iteration: 110000

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd ${HOME}/repositories/PROVE

echo "=========================================="
echo "Testing baseline extended checkpoint"
echo "=========================================="
echo "Dataset: IDD-AW"
echo "Model: pspnet_r50"
echo "Iteration: 110000"
echo "Checkpoint: ${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/iter_110000.pth"
echo "Output: ${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/test_results_detailed/iter_110000"
echo "=========================================="

# Create output directory
mkdir -p "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/test_results_detailed/iter_110000"

python fine_grained_test.py \
    --config "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/training_config.py" \
    --checkpoint "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/iter_110000.pth" \
    --dataset IDD-AW \
    --output-dir "${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/iddaw/pspnet_r50/test_results_detailed/iter_110000"

echo "Testing complete!"

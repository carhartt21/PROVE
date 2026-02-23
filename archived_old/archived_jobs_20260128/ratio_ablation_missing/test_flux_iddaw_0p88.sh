#!/bin/bash
#BSUB -J test_flux_iddaw_0p88
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p88/test_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p88/test_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 4:00

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd ${HOME}/repositories/PROVE

python fine_grained_test.py \
    --config "${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p88/training_config.py" \
    --checkpoint "${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p88/iter_80000.pth" \
    --dataset IDD-AW \
    --output-dir "${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p88/test_results_detailed"

echo "Testing completed: test_flux_iddaw_0p88"

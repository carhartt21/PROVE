#!/bin/bash
#BSUB -J da_gen_step1x_new_idd-aw_pspnet_r50
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50/da_test_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50/da_test_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 4:00

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd ${HOME}/repositories/PROVE

# Create output directory
mkdir -p "${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50"

# Run domain adaptation evaluation (test on ACDC)
python fine_grained_test.py \
    --config "${AWARE_DATA_ROOT}/WEIGHTS/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50/training_config.py" \
    --checkpoint "${AWARE_DATA_ROOT}/WEIGHTS/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50/iter_80000.pth" \
    --dataset ACDC \
    --output-dir "${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50"

# Rename results to domain_adaptation_evaluation.json
if [ -f "${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50/results.json" ]; then
    mv "${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50/results.json" "${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/gen_step1x_new/idd-aw/pspnet_r50_ratio0p50/domain_adaptation_evaluation.json"
fi

echo "Domain adaptation test completed: da_gen_step1x_new_idd-aw_pspnet_r50"

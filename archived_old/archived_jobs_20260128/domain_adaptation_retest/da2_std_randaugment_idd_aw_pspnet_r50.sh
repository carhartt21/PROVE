#!/bin/bash
#BSUB -J da2_std_randaugment_idd_aw_pspnet_r50
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/std_randaugment/idd-aw/pspnet_r50/da_test_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/std_randaugment/idd-aw/pspnet_r50/da_test_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 4:00

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd ${HOME}/repositories/PROVE

# Create output directory
mkdir -p "${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/std_randaugment/idd-aw/pspnet_r50"

# Run domain adaptation evaluation using the correct evaluation script
python tools/evaluate_domain_adaptation.py \
    --source-dataset IDD-AW \
    --model pspnet_r50 \
    --strategy std_randaugment \
    --checkpoint "${AWARE_DATA_ROOT}/WEIGHTS/std_randaugment/idd-aw/pspnet_r50/iter_80000.pth" \
    --device cuda

echo "Domain adaptation test completed: da2_std_randaugment_idd_aw_pspnet_r50"

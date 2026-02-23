#!/bin/bash
#BSUB -J da2_baseline_idd_aw_segformer_mit_b5
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/baseline/idd-aw/segformer_mit-b5/da_test_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/baseline/idd-aw/segformer_mit-b5/da_test_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 4:00

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd ${HOME}/repositories/PROVE

# Create output directory
mkdir -p "${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/baseline/idd-aw/segformer_mit-b5"

# Run domain adaptation evaluation using the correct evaluation script
python tools/evaluate_domain_adaptation.py \
    --source-dataset IDD-AW \
    --model segformer_mit-b5 \
    --strategy baseline \
    --checkpoint "${AWARE_DATA_ROOT}/WEIGHTS/baseline/idd-aw/segformer_mit-b5/iter_80000.pth" \
    --device cuda

echo "Domain adaptation test completed: da2_baseline_idd_aw_segformer_mit_b5"

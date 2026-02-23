#!/bin/bash
#BSUB -J train_gen_flux_kontext_BDD10k_0p62
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/BDD10k/segformer_mit-b5_ratio0p62/train_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/BDD10k/segformer_mit-b5_ratio0p62/train_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=48000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 24:00

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd ${HOME}/repositories/PROVE

python unified_training.py \
    --dataset BDD10k \
    --model segformer_mit-b5 \
    --strategy gen_flux_kontext \
    --real-gen-ratio 0.62 \
    --domain-filter clear_day \
    --work-dir "${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/BDD10k/segformer_mit-b5_ratio0p62" \
    

echo "Training completed: train_gen_flux_kontext_BDD10k_0p62"

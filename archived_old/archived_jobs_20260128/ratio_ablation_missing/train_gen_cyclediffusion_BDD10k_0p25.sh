#!/bin/bash
#BSUB -J train_gen_cyclediffusion_BDD10k_0p25
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_cyclediffusion/BDD10k/segformer_mit-b5_ratio0p25/train_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_cyclediffusion/BDD10k/segformer_mit-b5_ratio0p25/train_%J.err
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
    --strategy gen_cyclediffusion \
    --real-gen-ratio 0.25 \
    --domain-filter clear_day \
    --work-dir "${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/stage1/gen_cyclediffusion/BDD10k/segformer_mit-b5_ratio0p25" \
    

echo "Training completed: train_gen_cyclediffusion_BDD10k_0p25"

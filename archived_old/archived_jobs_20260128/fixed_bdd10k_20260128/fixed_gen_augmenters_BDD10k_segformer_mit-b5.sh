#!/bin/bash
#BSUB -J fixed_gen_augmenters_BDD10k_segformer_mit-b5
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS_FIXED_BDD10K/gen_augmenters/bdd10k/segformer_mit-b5_ratio0p5/train_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS_FIXED_BDD10K/gen_augmenters/bdd10k/segformer_mit-b5_ratio0p5/train_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=48000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 24:00

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd ${HOME}/repositories/PROVE

echo "=============================================="
echo "FIXED BDD10k Training: gen_augmenters"
echo "Bug fix commit: ecb9721"
echo "Batch size: 2 (default)"
echo "=============================================="

python unified_training.py     --dataset BDD10k     --model segformer_mit-b5     --strategy gen_augmenters     --real-gen-ratio 0.5     --domain-filter clear_day     --work-dir "${AWARE_DATA_ROOT}/WEIGHTS_FIXED_BDD10K/gen_augmenters/bdd10k/segformer_mit-b5_ratio0p5"

echo "Training completed: fixed_gen_augmenters_BDD10k_segformer_mit-b5"

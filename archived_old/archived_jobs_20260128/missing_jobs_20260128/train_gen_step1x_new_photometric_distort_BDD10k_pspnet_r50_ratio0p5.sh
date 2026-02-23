#!/bin/bash
#BSUB -J train_gen_step1x_new_photometric_distort_BDD10k_pspnet_r50_ratio0p5
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/gen_step1x_new+photometric_distort/BDD10k/pspnet_r50_ratio0p5/train_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/gen_step1x_new+photometric_distort/BDD10k/pspnet_r50_ratio0p5/train_%J.err
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
    --model pspnet_r50 \
    --strategy gen_step1x_new+photometric_distort \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --work-dir "${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/gen_step1x_new+photometric_distort/BDD10k/pspnet_r50_ratio0p5" \
    --resume-from ${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/gen_step1x_new+photometric_distort/bdd10k/pspnet_r50_ratio0p5/iter_10000.pth

echo "Training completed: train_gen_step1x_new_photometric_distort_BDD10k_pspnet_r50_ratio0p5"

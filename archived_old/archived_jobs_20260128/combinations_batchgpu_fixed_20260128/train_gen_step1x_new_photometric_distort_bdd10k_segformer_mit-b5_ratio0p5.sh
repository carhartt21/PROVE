#!/bin/bash
#BSUB -J train_gen_step1x_new_photometric_distort_bdd10k_segformer_mit-b5_ratio0p5
#BSUB -o ${HOME}/repositories/PROVE/logs/combinations_batchgpu_fixed/train_gen_step1x_new_photometric_distort_bdd10k_segformer_mit-b5_ratio0p5_%J.out
#BSUB -e ${HOME}/repositories/PROVE/logs/combinations_batchgpu_fixed/train_gen_step1x_new_photometric_distort_bdd10k_segformer_mit-b5_ratio0p5_%J.err
#BSUB -q BatchGPU
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
    --strategy gen_step1x_new \
    --std-strategy photometric_distort \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
     \
    --work-dir "${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/gen_step1x_new+photometric_distort/bdd10k/segformer_mit-b5_ratio0p5" \
    

echo "Training completed: train_gen_step1x_new_photometric_distort_bdd10k_segformer_mit-b5_ratio0p5"

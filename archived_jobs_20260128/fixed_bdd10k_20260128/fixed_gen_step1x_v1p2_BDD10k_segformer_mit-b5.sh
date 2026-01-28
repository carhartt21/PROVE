#!/bin/bash
#BSUB -J fixed_gen_step1x_v1p2_BDD10k
#BSUB -q BatchGPU
#BSUB -o /scratch/aaa_exchange/AWARE/WEIGHTS_FIXED_BDD10K/gen_step1x_v1p2/bdd10k/segformer_mit-b5_ratio0p5/train_%J.out
#BSUB -e /scratch/aaa_exchange/AWARE/WEIGHTS_FIXED_BDD10K/gen_step1x_v1p2/bdd10k/segformer_mit-b5_ratio0p5/train_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=48000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 24:00

source /home/mima2416/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd /home/mima2416/repositories/PROVE

python unified_training.py     --dataset BDD10k     --model segformer_mit-b5     --strategy gen_step1x_v1p2     --real-gen-ratio 0.5     --domain-filter clear_day     --work-dir "/scratch/aaa_exchange/AWARE/WEIGHTS_FIXED_BDD10K/gen_step1x_v1p2/bdd10k/segformer_mit-b5_ratio0p5"

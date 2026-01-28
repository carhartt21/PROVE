#!/bin/bash
#BSUB -J retest_cyclediff_bdd10k_ratio0p00
#BSUB -q BatchGPU
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 2:00
#BSUB -o /home/mima2416/repositories/PROVE/logs/retest_cyclediff_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retest_cyclediff_%J.err

cd /home/mima2416/repositories/PROVE
source ~/.bashrc
mamba activate prove

python fine_grained_test.py \
    --config "/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_cyclediffusion/BDD10k/segformer_mit-b5_ratio0p00/training_config.py" \
    --checkpoint "/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_cyclediffusion/BDD10k/segformer_mit-b5_ratio0p00/iter_80000.pth" \
    --dataset BDD10k \
    --output-dir "/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_cyclediffusion/BDD10k/segformer_mit-b5_ratio0p00/test_results_detailed"

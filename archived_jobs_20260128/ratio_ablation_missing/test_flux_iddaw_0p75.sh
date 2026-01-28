#!/bin/bash
#BSUB -J test_flux_iddaw_0p75
#BSUB -q BatchGPU
#BSUB -o /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p75/test_%J.out
#BSUB -e /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p75/test_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 4:00

source /home/mima2416/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd /home/mima2416/repositories/PROVE

python fine_grained_test.py \
    --config "/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p75/training_config.py" \
    --checkpoint "/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p75/iter_80000.pth" \
    --dataset IDD-AW \
    --output-dir "/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1/gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p75/test_results_detailed"

echo "Testing completed: test_flux_iddaw_0p75"

#!/bin/bash
#BSUB -J da_std_cutmix_idd-aw_segformer_mit-b5
#BSUB -q BatchGPU
#BSUB -o /scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/std_cutmix/idd-aw/segformer_mit-b5/da_test_%J.out
#BSUB -e /scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/std_cutmix/idd-aw/segformer_mit-b5/da_test_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 4:00

source /home/mima2416/miniconda3/etc/profile.d/conda.sh
conda activate prove
cd /home/mima2416/repositories/PROVE

# Create output directory
mkdir -p "/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/std_cutmix/idd-aw/segformer_mit-b5"

# Run domain adaptation evaluation (test on ACDC)
python fine_grained_test.py \
    --config "/scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/idd-aw/segformer_mit-b5/training_config.py" \
    --checkpoint "/scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/idd-aw/segformer_mit-b5/iter_80000.pth" \
    --dataset ACDC \
    --output-dir "/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/std_cutmix/idd-aw/segformer_mit-b5"

# Rename results to domain_adaptation_evaluation.json
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/std_cutmix/idd-aw/segformer_mit-b5/results.json" ]; then
    mv "/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/std_cutmix/idd-aw/segformer_mit-b5/results.json" "/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/std_cutmix/idd-aw/segformer_mit-b5/domain_adaptation_evaluation.json"
fi

echo "Domain adaptation test completed: da_std_cutmix_idd-aw_segformer_mit-b5"

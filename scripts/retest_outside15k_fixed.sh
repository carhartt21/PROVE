#!/bin/bash
# Re-run OUTSIDE15k Stage 1 tests with correct 24-class configuration

cd /home/mima2416/repositories/PROVE

# List of models that need re-testing (had wrong class count)
declare -A MODELS
MODELS["gen_Qwen_Image_Edit_pspnet"]="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Qwen_Image_Edit/outside15k/pspnet_r50_ratio0p50"
MODELS["gen_stargan_v2_segformer"]="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_stargan_v2/outside15k/segformer_mit-b5_ratio0p50"
MODELS["gen_step1x_new_pspnet"]="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k/pspnet_r50_ratio0p50"
MODELS["gen_step1x_new_segformer"]="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k/segformer_mit-b5_ratio0p50"
MODELS["gen_step1x_v1p2_pspnet"]="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_v1p2/outside15k/pspnet_r50_ratio0p50"
MODELS["gen_step1x_v1p2_segformer"]="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_v1p2/outside15k/segformer_mit-b5_ratio0p50"

for name in "${!MODELS[@]}"; do
    DIR="${MODELS[$name]}"
    CONFIG="$DIR/training_config.py"
    CHECKPOINT="$DIR/iter_80000.pth"
    
    if [ -f "$CONFIG" ] && [ -f "$CHECKPOINT" ]; then
        JOB_NAME="retest_$name"
        
        echo "Submitting: $name"
        bsub -J "$JOB_NAME" \
             -q BatchGPU \
             -gpu "num=1:mode=shared:gmem=16G" \
             -n 6 \
             -W 2:00 \
             -o /home/mima2416/repositories/PROVE/logs/${JOB_NAME}.log \
             -e /home/mima2416/repositories/PROVE/logs/${JOB_NAME}.err \
             "source ~/.bashrc && conda activate prove && cd /home/mima2416/repositories/PROVE && python fine_grained_test.py --config $CONFIG --checkpoint $CHECKPOINT --dataset OUTSIDE15k"
    else
        echo "Skipping $name - missing config or checkpoint"
    fi
done

echo "All jobs submitted!"

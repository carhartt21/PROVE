#!/bin/bash
# Submit missing Stage 2 MapillaryVistas tests (3 models)
# These are causing the "11 models" count in the leaderboard

set -e
cd /home/mima2416/repositories/PROVE

STAGE2_WEIGHTS="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"

# Missing tests identified from leaderboard analysis:
# 1. gen_stargan_v2/mapillaryvistas/deeplabv3plus_r50_ratio0p50
# 2. gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50_ratio0p50
# 3. gen_step1x_new/mapillaryvistas/pspnet_r50_ratio0p50

declare -a MISSING_TESTS=(
    "gen_stargan_v2|mapillaryvistas|deeplabv3plus_r50_ratio0p50"
    "gen_Weather_Effect_Generator|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_step1x_new|mapillaryvistas|pspnet_r50_ratio0p50"
)

for entry in "${MISSING_TESTS[@]}"; do
    IFS='|' read -r strategy dataset model <<< "$entry"
    
    model_dir="${STAGE2_WEIGHTS}/${strategy}/${dataset}/${model}"
    config="${model_dir}/training_config.py"
    checkpoint="${model_dir}/iter_80000.pth"
    output_dir="${model_dir}/test_results_detailed"
    
    if [ ! -f "$checkpoint" ]; then
        echo "ERROR: Checkpoint not found: $checkpoint"
        continue
    fi
    
    if [ ! -f "$config" ]; then
        echo "ERROR: Config not found: $config"
        continue
    fi
    
    job_name="fg_s2_${strategy}_${dataset}_${model}"
    # Truncate job name if too long
    job_name="${job_name:0:50}"
    
    echo "Submitting: $strategy/$dataset/$model"
    
    bsub -J "$job_name" \
         -q BatchGPU \
         -W 2:00 \
         -n 4 \
         -R "rusage[mem=16000]" \
         -gpu "num=1:mode=exclusive_process" \
         -o "/home/mima2416/repositories/PROVE/logs/${job_name}_%J.log" \
         -e "/home/mima2416/repositories/PROVE/logs/${job_name}_%J.err" \
         "source ~/.bashrc && conda activate prove && python /home/mima2416/repositories/PROVE/fine_grained_test.py --config $config --checkpoint $checkpoint --dataset MapillaryVistas --output-dir $output_dir"
done

echo ""
echo "Submitted 3 missing Stage 2 MapillaryVistas tests"

#!/bin/bash
# Submit training jobs for missing IDD-AW gen_Qwen_Image_Edit models
# These models failed on Dec 22 because IDD-AW images weren't in manifest yet
# Now the manifest has 15,349 IDD-AW images available

cd /home/mima2416/repositories/PROVE

# Common settings
STRATEGY="gen_Qwen_Image_Edit"
DATASET="IDD-AW"
QUEUE="BatchGPU"
GPU_TYPE="A100"
WALLTIME="24:00"
MEM="16G"

echo "Submitting training jobs for missing IDD-AW gen_Qwen_Image_Edit models..."

# 1. deeplabv3plus_r50
MODEL="deeplabv3plus_r50"
JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}:gmodel=${GPU_TYPE}" \
     -W $WALLTIME \
     -J "$JOB_NAME" \
     -o "logs/${JOB_NAME}_%J.log" \
     -e "logs/${JOB_NAME}_%J.err" \
     "source ~/.bashrc && conda activate prove && python unified_training.py --strategy $STRATEGY --datasets $DATASET --architectures $MODEL --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"
echo "Submitted: $JOB_NAME"

# 2. deeplabv3plus_r50_clear_day
MODEL="deeplabv3plus_r50_clear_day"
JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}:gmodel=${GPU_TYPE}" \
     -W $WALLTIME \
     -J "$JOB_NAME" \
     -o "logs/${JOB_NAME}_%J.log" \
     -e "logs/${JOB_NAME}_%J.err" \
     "source ~/.bashrc && conda activate prove && python unified_training.py --strategy $STRATEGY --datasets $DATASET --architectures $MODEL --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"
echo "Submitted: $JOB_NAME"

# 3. pspnet_r50
MODEL="pspnet_r50"
JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}:gmodel=${GPU_TYPE}" \
     -W $WALLTIME \
     -J "$JOB_NAME" \
     -o "logs/${JOB_NAME}_%J.log" \
     -e "logs/${JOB_NAME}_%J.err" \
     "source ~/.bashrc && conda activate prove && python unified_training.py --strategy $STRATEGY --datasets $DATASET --architectures $MODEL --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"
echo "Submitted: $JOB_NAME"

# 4. pspnet_r50_clear_day
MODEL="pspnet_r50_clear_day"
JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}:gmodel=${GPU_TYPE}" \
     -W $WALLTIME \
     -J "$JOB_NAME" \
     -o "logs/${JOB_NAME}_%J.log" \
     -e "logs/${JOB_NAME}_%J.err" \
     "source ~/.bashrc && conda activate prove && python unified_training.py --strategy $STRATEGY --datasets $DATASET --architectures $MODEL --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"
echo "Submitted: $JOB_NAME"

echo ""
echo "All 4 IDD-AW gen_Qwen_Image_Edit training jobs submitted!"
echo "Use 'bjobs' to check job status."

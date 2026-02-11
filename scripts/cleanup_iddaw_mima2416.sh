#!/bin/bash
# Script for mima2416 to run to clean up IDD-AW incomplete directories
# This removes old config files and run directories from failed training attempts
# Run this as user mima2416 before resubmitting training jobs

echo "Cleaning up IDD-AW directories with mima2416-owned files..."

# Delete all content from incomplete model directories (no iter_80000.pth)
# These directories have stale configs pointing to old paths

# baseline
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/idd-aw_cd/pspnet_r50/*

# gen_* strategies with incomplete runs
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Attribute_Hallucination/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_automold/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_CNetSeg/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_cyclediffusion/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_cycleGAN/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_cycleGAN/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Img2Img/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Img2Img/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_IP2P/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Qwen_Image_Edit/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Qwen_Image_Edit/idd-aw_cd/segformer_mit-b5_ratio0p50/*

# gen_* strategies with no complete runs (0/3 models)
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_stargan_v2/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_stargan_v2/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_stargan_v2/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_v1p2/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_v1p2/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_v1p2/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_SUSTechGAN/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_SUSTechGAN/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_SUSTechGAN/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_TSIT/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_TSIT/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_TSIT/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_VisualCloze/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_VisualCloze/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_VisualCloze/idd-aw_cd/segformer_mit-b5_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw_cd/deeplabv3plus_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw_cd/pspnet_r50_ratio0p50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw_cd/segformer_mit-b5_ratio0p50/*

# photometric_distort and std_* strategies (no complete non-corrupted runs)
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/idd-aw_cd/deeplabv3plus_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/idd-aw_cd/pspnet_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/idd-aw_cd/segformer_mit-b5/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_autoaugment/idd-aw_cd/deeplabv3plus_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_autoaugment/idd-aw_cd/pspnet_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_autoaugment/idd-aw_cd/segformer_mit-b5/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/idd-aw_cd/deeplabv3plus_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/idd-aw_cd/pspnet_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/idd-aw_cd/segformer_mit-b5/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/idd-aw_cd/deeplabv3plus_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/idd-aw_cd/pspnet_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/idd-aw_cd/segformer_mit-b5/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/idd-aw_cd/deeplabv3plus_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/idd-aw_cd/pspnet_r50/*
rm -rf /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/idd-aw_cd/segformer_mit-b5/*

echo ""
echo "Cleanup complete!"
echo "Directories with completed checkpoints (iter_80000.pth) were preserved."
echo ""
echo "After running this script, chge7185 can resubmit the IDD-AW training jobs."

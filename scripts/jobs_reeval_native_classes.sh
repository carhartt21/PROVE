#!/bin/bash
#BSUB -J reeval_native_classes
#BSUB -o /home/chge7185/repositories/PROVE/logs/retrain/reeval_native_classes_%J.out
#BSUB -e /home/chge7185/repositories/PROVE/logs/retrain/reeval_native_classes_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 4:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
mamba activate prove

cd /home/chge7185/repositories/PROVE

echo "========================================"
echo "Re-evaluation job: reeval_native_classes"
echo "Started: $(date)"
echo "========================================"


echo ""
echo "----------------------------------------"
echo "[1/18] Re-evaluating: photometric_distort/mapillaryvistas_cd/deeplabv3plus_r50"
echo "Dataset: MapillaryVistas"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/mapillaryvistas_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/mapillaryvistas_cd/deeplabv3plus_r50/training_config.py \
    --dataset MapillaryVistas \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/mapillaryvistas_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: photometric_distort/mapillaryvistas_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[2/18] Re-evaluating: photometric_distort/mapillaryvistas_cd/pspnet_r50"
echo "Dataset: MapillaryVistas"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/mapillaryvistas_cd/pspnet_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/mapillaryvistas_cd/pspnet_r50/training_config.py \
    --dataset MapillaryVistas \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/mapillaryvistas_cd/pspnet_r50/test_results_detailed_fixed

echo "Finished: photometric_distort/mapillaryvistas_cd/pspnet_r50"

echo ""
echo "----------------------------------------"
echo "[3/18] Re-evaluating: std_randaugment/mapillaryvistas_cd/deeplabv3plus_r50"
echo "Dataset: MapillaryVistas"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/mapillaryvistas_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/mapillaryvistas_cd/deeplabv3plus_r50/training_config.py \
    --dataset MapillaryVistas \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/mapillaryvistas_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: std_randaugment/mapillaryvistas_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[4/18] Re-evaluating: baseline/outside15k_cd/deeplabv3plus_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/outside15k_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/outside15k_cd/deeplabv3plus_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/outside15k_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: baseline/outside15k_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[5/18] Re-evaluating: baseline/outside15k_cd/pspnet_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/outside15k_cd/pspnet_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/outside15k_cd/pspnet_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/outside15k_cd/pspnet_r50/test_results_detailed_fixed

echo "Finished: baseline/outside15k_cd/pspnet_r50"

echo ""
echo "----------------------------------------"
echo "[6/18] Re-evaluating: gen_Attribute_Hallucination/outside15k_cd/deeplabv3plus_r50_ratio0p50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Attribute_Hallucination/outside15k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Attribute_Hallucination/outside15k_cd/deeplabv3plus_r50_ratio0p50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Attribute_Hallucination/outside15k_cd/deeplabv3plus_r50_ratio0p50/test_results_detailed_fixed

echo "Finished: gen_Attribute_Hallucination/outside15k_cd/deeplabv3plus_r50_ratio0p50"

echo ""
echo "----------------------------------------"
echo "[7/18] Re-evaluating: gen_CNetSeg/outside15k_cd/deeplabv3plus_r50_ratio0p50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_CNetSeg/outside15k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_CNetSeg/outside15k_cd/deeplabv3plus_r50_ratio0p50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_CNetSeg/outside15k_cd/deeplabv3plus_r50_ratio0p50/test_results_detailed_fixed

echo "Finished: gen_CNetSeg/outside15k_cd/deeplabv3plus_r50_ratio0p50"

echo ""
echo "----------------------------------------"
echo "[8/18] Re-evaluating: gen_CUT/outside15k_cd/deeplabv3plus_r50_ratio0p50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_CUT/outside15k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_CUT/outside15k_cd/deeplabv3plus_r50_ratio0p50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_CUT/outside15k_cd/deeplabv3plus_r50_ratio0p50/test_results_detailed_fixed

echo "Finished: gen_CUT/outside15k_cd/deeplabv3plus_r50_ratio0p50"

echo ""
echo "----------------------------------------"
echo "[9/18] Re-evaluating: gen_IP2P/outside15k_cd/deeplabv3plus_r50_ratio0p50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_IP2P/outside15k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_IP2P/outside15k_cd/deeplabv3plus_r50_ratio0p50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_IP2P/outside15k_cd/deeplabv3plus_r50_ratio0p50/test_results_detailed_fixed

echo "Finished: gen_IP2P/outside15k_cd/deeplabv3plus_r50_ratio0p50"

echo ""
echo "----------------------------------------"
echo "[10/18] Re-evaluating: photometric_distort/outside15k_cd/deeplabv3plus_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/outside15k_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/outside15k_cd/deeplabv3plus_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/outside15k_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: photometric_distort/outside15k_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[11/18] Re-evaluating: photometric_distort/outside15k_cd/pspnet_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/outside15k_cd/pspnet_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/outside15k_cd/pspnet_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/outside15k_cd/pspnet_r50/test_results_detailed_fixed

echo "Finished: photometric_distort/outside15k_cd/pspnet_r50"

echo ""
echo "----------------------------------------"
echo "[12/18] Re-evaluating: std_autoaugment/outside15k_cd/deeplabv3plus_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_autoaugment/outside15k_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_autoaugment/outside15k_cd/deeplabv3plus_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_autoaugment/outside15k_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: std_autoaugment/outside15k_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[13/18] Re-evaluating: std_cutmix/outside15k_cd/deeplabv3plus_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/outside15k_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/outside15k_cd/deeplabv3plus_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/outside15k_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: std_cutmix/outside15k_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[14/18] Re-evaluating: std_cutmix/outside15k_cd/pspnet_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/outside15k_cd/pspnet_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/outside15k_cd/pspnet_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_cutmix/outside15k_cd/pspnet_r50/test_results_detailed_fixed

echo "Finished: std_cutmix/outside15k_cd/pspnet_r50"

echo ""
echo "----------------------------------------"
echo "[15/18] Re-evaluating: std_mixup/outside15k_cd/deeplabv3plus_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/outside15k_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/outside15k_cd/deeplabv3plus_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/outside15k_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: std_mixup/outside15k_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[16/18] Re-evaluating: std_mixup/outside15k_cd/pspnet_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/outside15k_cd/pspnet_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/outside15k_cd/pspnet_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/outside15k_cd/pspnet_r50/test_results_detailed_fixed

echo "Finished: std_mixup/outside15k_cd/pspnet_r50"

echo ""
echo "----------------------------------------"
echo "[17/18] Re-evaluating: std_randaugment/outside15k_cd/deeplabv3plus_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/deeplabv3plus_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/deeplabv3plus_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/deeplabv3plus_r50/test_results_detailed_fixed

echo "Finished: std_randaugment/outside15k_cd/deeplabv3plus_r50"

echo ""
echo "----------------------------------------"
echo "[18/18] Re-evaluating: std_randaugment/outside15k_cd/pspnet_r50"
echo "Dataset: OUTSIDE15k"
echo "----------------------------------------"

python fine_grained_test.py \
    --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/pspnet_r50/iter_80000.pth \
    --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/pspnet_r50/training_config.py \
    --dataset OUTSIDE15k \
    --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/pspnet_r50/test_results_detailed_fixed

echo "Finished: std_randaugment/outside15k_cd/pspnet_r50"

echo ""
echo "========================================"
echo "All re-evaluations completed: $(date)"
echo "========================================"

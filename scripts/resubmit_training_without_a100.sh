#!/bin/bash
# ============================================================================
# Resubmit training jobs without A100 requirement
# Generated automatically - removes gmodel=NVIDIAA100_PCIE_40GB from resource spec
# ============================================================================

set -e
QUEUE="BatchGPU"
GPU_MEM="16G"  
MAX_TIME="24:00"

submitted=0
failed=0


# Job: prove_resume_BDD10k_pspnet_r50_gen_StyleID
echo "Submitting prove_resume_BDD10k_pspnet_r50_gen_StyleID..."
if bsub -J "prove_resume_BDD10k_pspnet_r50_gen_StyleID" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_resume_BDD10k_pspnet_r50_gen_StyleID_%J.log" \
    -e "logs/prove_resume_BDD10k_pspnet_r50_gen_StyleID_%J.err" \
    "mamba run -n prove python unified_ training.py --dataset BDD10k --model pspnet_r50 --str ategy gen_StyleID --resume-from /scratch/aaa_exchange /AWARE/WEIGHTS/gen_StyleID/bdd10k/pspnet_r50/iter_600 00.pth --ratio 1.0 --work-dir /scratch/aaa_exchange/A WARE/WEIGHTS --data-root /scratch/aaa_exchange/AWARE/ FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_resume_BDD10k_pspnet_r50_gen_StyleID"
    ((failed++)) || true
fi


# Job: prove_resume_IDD-AW_deeplabv3plus_r50_clear_day_gen_St
                          yleID
echo "Submitting prove_resume_IDD-AW_deeplabv3plus_r50_clear_day_gen_St
                          yleID..."
if bsub -J "prove_resume_IDD-AW_deeplabv3plus_r50_clear_day_gen_St
                          yleID" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_resume_IDD-AW_deeplabv3plus_r50_clear_day_gen_St
                          yleID_%J.log" \
    -e "logs/prove_resume_IDD-AW_deeplabv3plus_r50_clear_day_gen_St
                          yleID_%J.err" \
    "mamba run -n prov e python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy gen_StyleID --domain-fi lter clear_day --resume-from /scratch/aaa_exchange/AW ARE/WEIGHTS/gen_StyleID/idd-aw/deeplabv3plus_r50_clea r_day/iter_50000.pth --ratio 1.0 --work-dir /scratch/ aaa_exchange/AWARE/WEIGHTS --data-root /scratch/aaa_e xchange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_resume_IDD-AW_deeplabv3plus_r50_clear_day_gen_St
                          yleID"
    ((failed++)) || true
fi


# Job: prove_resume_MapillaryVistas_deeplabv3plus_r50_clear_d
                          ay_photometric_distort
echo "Submitting prove_resume_MapillaryVistas_deeplabv3plus_r50_clear_d
                          ay_photometric_distort..."
if bsub -J "prove_resume_MapillaryVistas_deeplabv3plus_r50_clear_d
                          ay_photometric_distort" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_resume_MapillaryVistas_deeplabv3plus_r50_clear_d
                          ay_photometric_distort_%J.log" \
    -e "logs/prove_resume_MapillaryVistas_deeplabv3plus_r50_clear_d
                          ay_photometric_distort_%J.err" \
    "mamba run -n prove python unified_training.py --datas et MapillaryVistas --model deeplabv3plus_r50 --strate gy photometric_distort --domain-filter clear_day --re sume-from /scratch/aaa_exchange/AWARE/WEIGHTS/photome tric_distort/mapillaryvistas/deeplabv3plus_r50_clear_ day/iter_30000.pth --ratio 1.0 --work-dir /scratch/aa a_exchange/AWARE/WEIGHTS --data-root /scratch/aaa_exc hange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_resume_MapillaryVistas_deeplabv3plus_r50_clear_d
                          ay_photometric_distort"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_clear_day_gen_augmenters
echo "Submitting prove_BDD10k_pspnet_r50_clear_day_gen_augmenters..."
if bsub -J "prove_BDD10k_pspnet_r50_clear_day_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_clear_day_gen_augmenters_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_clear_day_gen_augmenters_%J.err" \
    "mamba run -n prove python un ified_training.py --dataset BDD10k --model pspnet_r50 --strategy gen_augmenters --domain-filter clear_day --ratio 1.0 --work-dir /scratch/aaa_exchange/AWARE/WE IGHTS --data-root /scratch/aaa_exchange/AWARE/FINAL_S PLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_clear_day_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_gen_augmenters
echo "Submitting prove_BDD10k_segformer_mit-b5_gen_augmenters..."
if bsub -J "prove_BDD10k_segformer_mit-b5_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_gen_augmenters_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_gen_augmenters_%J.err" \
    "mamba run -n prove python unifie d_training.py --dataset BDD10k --model segformer_mit- b5 --strategy gen_augmenters --ratio 1.0 --work-dir / scratch/aaa_exchange/AWARE/WEIGHTS --data-root /scrat ch/aaa_exchange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_clear_day_gen_augmenters
                          
echo "Submitting prove_BDD10k_segformer_mit-b5_clear_day_gen_augmenters
                          ..."
if bsub -J "prove_BDD10k_segformer_mit-b5_clear_day_gen_augmenters
                          " \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_augmenters
                          _%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_augmenters
                          _%J.err" \
    "mamba run -n prove pyt hon unified_training.py --dataset BDD10k --model segf ormer_mit-b5 --strategy gen_augmenters --domain-filte r clear_day --ratio 1.0 --work-dir /scratch/aaa_excha nge/AWARE/WEIGHTS --data-root /scratch/aaa_exchange/A WARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_clear_day_gen_augmenters
                          "
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_gen_augmenters
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_gen_augmenters..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_gen_augmenters_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_gen_augmenters_%J.err" \
    "mamba run -n prove python unifi ed_training.py --dataset IDD-AW --model deeplabv3plus _r50 --strategy gen_augmenters --ratio 1.0 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS --data-root /scr atch/aaa_exchange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_augmenter
                          s
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_augmenter
                          s..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_augmenter
                          s" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_augmenter
                          s_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_augmenter
                          s_%J.err" \
    "mamba run -n prove py thon unified_training.py --dataset IDD-AW --model dee plabv3plus_r50 --strategy gen_augmenters --domain-fil ter clear_day --ratio 1.0 --work-dir /scratch/aaa_exc hange/AWARE/WEIGHTS --data-root /scratch/aaa_exchange /AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_augmenter
                          s"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_gen_augmenters
echo "Submitting prove_IDD-AW_pspnet_r50_gen_augmenters..."
if bsub -J "prove_IDD-AW_pspnet_r50_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_gen_augmenters_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_gen_augmenters_%J.err" \
    "mamba run -n prove python unified_trai ning.py --dataset IDD-AW --model pspnet_r50 --strateg y gen_augmenters --ratio 1.0 --work-dir /scratch/aaa_ exchange/AWARE/WEIGHTS --data-root /scratch/aaa_excha nge/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_clear_day_gen_augmenters
echo "Submitting prove_IDD-AW_pspnet_r50_clear_day_gen_augmenters..."
if bsub -J "prove_IDD-AW_pspnet_r50_clear_day_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_augmenters_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_augmenters_%J.err" \
    "mamba run -n prove python un ified_training.py --dataset IDD-AW --model pspnet_r50 --strategy gen_augmenters --domain-filter clear_day --ratio 1.0 --work-dir /scratch/aaa_exchange/AWARE/WE IGHTS --data-root /scratch/aaa_exchange/AWARE/FINAL_S PLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_clear_day_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_gen_augmenters
echo "Submitting prove_IDD-AW_segformer_mit-b5_gen_augmenters..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_gen_augmenters_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_gen_augmenters_%J.err" \
    "mamba run -n prove python unifie d_training.py --dataset IDD-AW --model segformer_mit- b5 --strategy gen_augmenters --ratio 1.0 --work-dir / scratch/aaa_exchange/AWARE/WEIGHTS --data-root /scrat ch/aaa_exchange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_clear_day_gen_augmenters
                          
echo "Submitting prove_IDD-AW_segformer_mit-b5_clear_day_gen_augmenters
                          ..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_clear_day_gen_augmenters
                          " \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_augmenters
                          _%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_augmenters
                          _%J.err" \
    "mamba run -n prove pyt hon unified_training.py --dataset IDD-AW --model segf ormer_mit-b5 --strategy gen_augmenters --domain-filte r clear_day --ratio 1.0 --work-dir /scratch/aaa_excha nge/AWARE/WEIGHTS --data-root /scratch/aaa_exchange/A WARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_clear_day_gen_augmenters
                          "
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_gen_augmenters
                          
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_gen_augmenters
                          ..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_gen_augmenters
                          " \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_augmenters
                          _%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_augmenters
                          _%J.err" \
    "mamba run -n prove pyt hon unified_training.py --dataset MapillaryVistas --m odel deeplabv3plus_r50 --strategy gen_augmenters --ra tio 1.0 --work-dir /scratch/aaa_exchange/AWARE/WEIGHT S --data-root /scratch/aaa_exchange/AWARE/FINAL_SPLIT S --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_gen_augmenters
                          "
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          augmenters
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          augmenters..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          augmenters_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          augmenters_%J.err" \
    "mamba run -n prove python unified_training.py --dataset Mapillary Vistas --model deeplabv3plus_r50 --strategy gen_augme nters --domain-filter clear_day --ratio 1.0 --work-di r /scratch/aaa_exchange/AWARE/WEIGHTS --data-root /sc ratch/aaa_exchange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          augmenters"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_gen_augmenters
echo "Submitting prove_MapillaryVistas_pspnet_r50_gen_augmenters..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_gen_augmenters_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_gen_augmenters_%J.err" \
    "mamba run -n prove python uni fied_training.py --dataset MapillaryVistas --model ps pnet_r50 --strategy gen_augmenters --ratio 1.0 --work -dir /scratch/aaa_exchange/AWARE/WEIGHTS --data-root /scratch/aaa_exchange/AWARE/FINAL_SPLITS --no-early-s top"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_clear_day_gen_augment
                          ers
echo "Submitting prove_MapillaryVistas_pspnet_r50_clear_day_gen_augment
                          ers..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_clear_day_gen_augment
                          ers" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_augment
                          ers_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_augment
                          ers_%J.err" \
    "mamba run -n prove python unified_training.py --dataset MapillaryVistas --model pspnet_r50 --strategy gen_augmenters --domain -filter clear_day --ratio 1.0 --work-dir /scratch/aaa _exchange/AWARE/WEIGHTS --data-root /scratch/aaa_exch ange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_clear_day_gen_augment
                          ers"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_gen_augmenters
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_gen_augmenters..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_gen_augmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_gen_augmenters_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_gen_augmenters_%J.err" \
    "mamba run -n prove pyth on unified_training.py --dataset MapillaryVistas --mo del segformer_mit-b5 --strategy gen_augmenters --rati o 1.0 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS --data-root /scratch/aaa_exchange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_gen_augmenters"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          ugmenters
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          ugmenters..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          ugmenters" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          ugmenters_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          ugmenters_%J.err" \
    "mamba run -n prove python unified_training.py --dataset MapillaryV istas --model segformer_mit-b5 --strategy gen_augment ers --domain-filter clear_day --ratio 1.0 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS --data-root /scra tch/aaa_exchange/AWARE/FINAL_SPLITS --no-early-stop"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          ugmenters"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_gen_Qwen_Image_Edit
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_gen_Qwen_Image_Edit..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_gen_Qwen_Image_Edit" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_gen_Qwen_Image_Edit_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_gen_Qwen_Image_Edit_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strate gy gen_Qwen_Image_Edit --datasets IDD-AW --architectu res deeplabv3plus_r50 --work-dir /scratch/aaa_exchang e/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_gen_Qwen_Image_Edit"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_Qwen_Imag
                          e_Edit
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_Qwen_Imag
                          e_Edit..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_Qwen_Imag
                          e_Edit" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_Qwen_Imag
                          e_Edit_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_Qwen_Imag
                          e_Edit_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.p y --strategy gen_Qwen_Image_Edit --datasets IDD-AW -- architectures deeplabv3plus_r50_clear_day --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_Qwen_Imag
                          e_Edit"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_gen_Qwen_Image_Edit
echo "Submitting prove_IDD-AW_pspnet_r50_gen_Qwen_Image_Edit..."
if bsub -J "prove_IDD-AW_pspnet_r50_gen_Qwen_Image_Edit" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_gen_Qwen_Image_Edit_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_gen_Qwen_Image_Edit_%J.err" \
    "source ~/.bashrc && conda activat e prove && python unified_training.py --strategy gen_ Qwen_Image_Edit --datasets IDD-AW --architectures psp net_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHT S"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_gen_Qwen_Image_Edit"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_clear_day_gen_Qwen_Image_Edit
echo "Submitting prove_IDD-AW_pspnet_r50_clear_day_gen_Qwen_Image_Edit..."
if bsub -J "prove_IDD-AW_pspnet_r50_clear_day_gen_Qwen_Image_Edit" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_Qwen_Image_Edit_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_Qwen_Image_Edit_%J.err" \
    "source ~/.bashrc && con da activate prove && python unified_training.py --str ategy gen_Qwen_Image_Edit --datasets IDD-AW --archite ctures pspnet_r50_clear_day --work-dir /scratch/aaa_e xchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_clear_day_gen_Qwen_Image_Edit"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_gen_VisualCloze
echo "Submitting prove_BDD10k_deeplabv3plus_r50_gen_VisualCloze..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_gen_VisualCloze_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda acti vate prove && python unified_training.py --strategy g en_VisualCloze --datasets BDD10k --architectures deep labv3plus_r50 --work-dir /scratch/aaa_exchange/AWARE/ WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze
echo "Submitting prove_BDD10k_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py -- strategy gen_VisualCloze --datasets BDD10k --architec tures deeplabv3plus_r50_clear_day --work-dir /scratch /aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_gen_VisualCloze
echo "Submitting prove_BDD10k_pspnet_r50_gen_VisualCloze..."
if bsub -J "prove_BDD10k_pspnet_r50_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_gen_VisualCloze_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda activate pr ove && python unified_training.py --strategy gen_Visu alCloze --datasets BDD10k --architectures pspnet_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_clear_day_gen_VisualCloze
echo "Submitting prove_BDD10k_pspnet_r50_clear_day_gen_VisualCloze..."
if bsub -J "prove_BDD10k_pspnet_r50_clear_day_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_clear_day_gen_VisualCloze_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_clear_day_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda a ctivate prove && python unified_training.py --strateg y gen_VisualCloze --datasets BDD10k --architectures p spnet_r50_clear_day --work-dir /scratch/aaa_exchange/ AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_clear_day_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_gen_VisualCloze
echo "Submitting prove_BDD10k_segformer_mit-b5_gen_VisualCloze..."
if bsub -J "prove_BDD10k_segformer_mit-b5_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_gen_VisualCloze_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda activ ate prove && python unified_training.py --strategy ge n_VisualCloze --datasets BDD10k --architectures segfo rmer_mit-b5 --work-dir /scratch/aaa_exchange/AWARE/WE IGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_clear_day_gen_VisualCloz
                          e
echo "Submitting prove_BDD10k_segformer_mit-b5_clear_day_gen_VisualCloz
                          e..."
if bsub -J "prove_BDD10k_segformer_mit-b5_clear_day_gen_VisualCloz
                          e" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_VisualCloz
                          e_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_VisualCloz
                          e_%J.err" \
    "source ~/.bashrc && c onda activate prove && python unified_training.py --s trategy gen_VisualCloze --datasets BDD10k --architect ures segformer_mit-b5_clear_day --work-dir /scratch/a aa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_clear_day_gen_VisualCloz
                          e"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_gen_VisualCloze
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_gen_VisualCloze..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_gen_VisualCloze_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda acti vate prove && python unified_training.py --strategy g en_VisualCloze --datasets IDD-AW --architectures deep labv3plus_r50 --work-dir /scratch/aaa_exchange/AWARE/ WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py -- strategy gen_VisualCloze --datasets IDD-AW --architec tures deeplabv3plus_r50_clear_day --work-dir /scratch /aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_VisualClo
                          ze"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_gen_VisualCloze
echo "Submitting prove_IDD-AW_pspnet_r50_gen_VisualCloze..."
if bsub -J "prove_IDD-AW_pspnet_r50_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_gen_VisualCloze_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda activate pr ove && python unified_training.py --strategy gen_Visu alCloze --datasets IDD-AW --architectures pspnet_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_clear_day_gen_VisualCloze
echo "Submitting prove_IDD-AW_pspnet_r50_clear_day_gen_VisualCloze..."
if bsub -J "prove_IDD-AW_pspnet_r50_clear_day_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_VisualCloze_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda a ctivate prove && python unified_training.py --strateg y gen_VisualCloze --datasets IDD-AW --architectures p spnet_r50_clear_day --work-dir /scratch/aaa_exchange/ AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_clear_day_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_gen_VisualCloze
echo "Submitting prove_IDD-AW_segformer_mit-b5_gen_VisualCloze..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_gen_VisualCloze_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda activ ate prove && python unified_training.py --strategy ge n_VisualCloze --datasets IDD-AW --architectures segfo rmer_mit-b5 --work-dir /scratch/aaa_exchange/AWARE/WE IGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_clear_day_gen_VisualCloz
                          e
echo "Submitting prove_IDD-AW_segformer_mit-b5_clear_day_gen_VisualCloz
                          e..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_clear_day_gen_VisualCloz
                          e" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_VisualCloz
                          e_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_VisualCloz
                          e_%J.err" \
    "source ~/.bashrc && c onda activate prove && python unified_training.py --s trategy gen_VisualCloze --datasets IDD-AW --architect ures segformer_mit-b5_clear_day --work-dir /scratch/a aa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_clear_day_gen_VisualCloz
                          e"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_gen_VisualCloz
                          e
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_gen_VisualCloz
                          e..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_gen_VisualCloz
                          e" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_VisualCloz
                          e_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_VisualCloz
                          e_%J.err" \
    "source ~/.bashrc && c onda activate prove && python unified_training.py --s trategy gen_VisualCloze --datasets MapillaryVistas -- architectures deeplabv3plus_r50 --work-dir /scratch/a aa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_gen_VisualCloz
                          e"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          VisualCloze
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          VisualCloze..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          VisualCloze_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          VisualCloze_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_VisualCloze --datasets Mapillar yVistas --architectures deeplabv3plus_r50_clear_day - -work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          VisualCloze"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_gen_VisualCloze
echo "Submitting prove_MapillaryVistas_pspnet_r50_gen_VisualCloze..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_gen_VisualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_gen_VisualCloze_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_gen_VisualCloze_%J.err" \
    "source ~/.bashrc && conda ac tivate prove && python unified_training.py --strategy gen_VisualCloze --datasets MapillaryVistas --archite ctures pspnet_r50 --work-dir /scratch/aaa_exchange/AW ARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_gen_VisualCloze"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_clear_day_gen_VisualC
                          loze
echo "Submitting prove_MapillaryVistas_pspnet_r50_clear_day_gen_VisualC
                          loze..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_clear_day_gen_VisualC
                          loze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_VisualC
                          loze_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_VisualC
                          loze_%J.err" \
    "source ~/.bashrc & & conda activate prove && python unified_training.py --strategy gen_VisualCloze --datasets MapillaryVistas --architectures pspnet_r50_clear_day --work-dir /scr atch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_clear_day_gen_VisualC
                          loze"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_gen_VisualCloze
                          
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_gen_VisualCloze
                          ..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_gen_VisualCloze
                          " \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_gen_VisualCloze
                          _%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_gen_VisualCloze
                          _%J.err" \
    "source ~/.bashrc && co nda activate prove && python unified_training.py --st rategy gen_VisualCloze --datasets MapillaryVistas --a rchitectures segformer_mit-b5 --work-dir /scratch/aaa _exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_gen_VisualCloze
                          "
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_V
                          isualCloze
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_V
                          isualCloze..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_V
                          isualCloze" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_V
                          isualCloze_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_V
                          isualCloze_%J.err" \
    "source ~/.ba shrc && conda activate prove && python unified_traini ng.py --strategy gen_VisualCloze --datasets Mapillary Vistas --architectures segformer_mit-b5_clear_day --w ork-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_V
                          isualCloze"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_gen_albumentations_weat
                          her
echo "Submitting prove_BDD10k_deeplabv3plus_r50_gen_albumentations_weat
                          her..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_gen_albumentations_weat
                          her" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_gen_albumentations_weat
                          her_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_gen_albumentations_weat
                          her_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py - -strategy gen_albumentations_weather --datasets BDD10 k --architectures deeplabv3plus_r50 --work-dir /scrat ch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_gen_albumentations_weat
                          her"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather
echo "Submitting prove_BDD10k_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather_%J.err" \
    "source ~/ .bashrc && conda activate prove && python unified_tra ining.py --strategy gen_albumentations_weather --data sets BDD10k --architectures deeplabv3plus_r50_clear_d ay --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_gen_albumentations_weather
echo "Submitting prove_BDD10k_pspnet_r50_gen_albumentations_weather..."
if bsub -J "prove_BDD10k_pspnet_r50_gen_albumentations_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_gen_albumentations_weather_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_gen_albumentations_weather_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strate gy gen_albumentations_weather --datasets BDD10k --arc hitectures pspnet_r50 --work-dir /scratch/aaa_exchang e/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_gen_albumentations_weather"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_clear_day_gen_albumentations_w
                          eather
echo "Submitting prove_BDD10k_pspnet_r50_clear_day_gen_albumentations_w
                          eather..."
if bsub -J "prove_BDD10k_pspnet_r50_clear_day_gen_albumentations_w
                          eather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_clear_day_gen_albumentations_w
                          eather_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_clear_day_gen_albumentations_w
                          eather_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.p y --strategy gen_albumentations_weather --datasets BD D10k --architectures pspnet_r50_clear_day --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_clear_day_gen_albumentations_w
                          eather"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_gen_albumentations_weath
                          er
echo "Submitting prove_BDD10k_segformer_mit-b5_gen_albumentations_weath
                          er..."
if bsub -J "prove_BDD10k_segformer_mit-b5_gen_albumentations_weath
                          er" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_gen_albumentations_weath
                          er_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_gen_albumentations_weath
                          er_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py -- strategy gen_albumentations_weather --datasets BDD10k --architectures segformer_mit-b5 --work-dir /scratch /aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_gen_albumentations_weath
                          er"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather
echo "Submitting prove_BDD10k_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather..."
if bsub -J "prove_BDD10k_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_albumentations_weather --datase ts BDD10k --architectures segformer_mit-b5_clear_day --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_gen_cyclediffusion
echo "Submitting prove_BDD10k_deeplabv3plus_r50_gen_cyclediffusion..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_gen_cyclediffusion_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda a ctivate prove && python unified_training.py --strateg y gen_cyclediffusion --datasets BDD10k --architecture s deeplabv3plus_r50 --work-dir /scratch/aaa_exchange/ AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion
echo "Submitting prove_BDD10k_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strategy gen_cyclediffusion --datasets BDD10k --ar chitectures deeplabv3plus_r50_clear_day --work-dir /s cratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_gen_cyclediffusion
echo "Submitting prove_BDD10k_pspnet_r50_gen_cyclediffusion..."
if bsub -J "prove_BDD10k_pspnet_r50_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_gen_cyclediffusion_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strategy gen_c yclediffusion --datasets BDD10k --architectures pspne t_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_clear_day_gen_cyclediffusion
echo "Submitting prove_BDD10k_pspnet_r50_clear_day_gen_cyclediffusion..."
if bsub -J "prove_BDD10k_pspnet_r50_clear_day_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_clear_day_gen_cyclediffusion_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_clear_day_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strat egy gen_cyclediffusion --datasets BDD10k --architectu res pspnet_r50_clear_day --work-dir /scratch/aaa_exch ange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_clear_day_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_gen_cyclediffusion
echo "Submitting prove_BDD10k_segformer_mit-b5_gen_cyclediffusion..."
if bsub -J "prove_BDD10k_segformer_mit-b5_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_gen_cyclediffusion_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda ac tivate prove && python unified_training.py --strategy gen_cyclediffusion --datasets BDD10k --architectures segformer_mit-b5 --work-dir /scratch/aaa_exchange/AW ARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion
echo "Submitting prove_BDD10k_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion..."
if bsub -J "prove_BDD10k_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion_%J.err" \
    "source ~/.bashrc & & conda activate prove && python unified_training.py --strategy gen_cyclediffusion --datasets BDD10k --arc hitectures segformer_mit-b5_clear_day --work-dir /scr atch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_gen_step1x_v1p2
echo "Submitting prove_BDD10k_deeplabv3plus_r50_gen_step1x_v1p2..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda acti vate prove && python unified_training.py --strategy g en_step1x_v1p2 --datasets BDD10k --architectures deep labv3plus_r50 --work-dir /scratch/aaa_exchange/AWARE/ WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_BDD10k_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2
echo "Submitting prove_BDD10k_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2..."
if bsub -J "prove_BDD10k_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2_%J.log" \
    -e "logs/prove_BDD10k_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py -- strategy gen_step1x_v1p2 --datasets BDD10k --architec tures deeplabv3plus_r50_clear_day --work-dir /scratch /aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_gen_step1x_v1p2
echo "Submitting prove_BDD10k_pspnet_r50_gen_step1x_v1p2..."
if bsub -J "prove_BDD10k_pspnet_r50_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda activate pr ove && python unified_training.py --strategy gen_step 1x_v1p2 --datasets BDD10k --architectures pspnet_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_BDD10k_pspnet_r50_clear_day_gen_step1x_v1p2
echo "Submitting prove_BDD10k_pspnet_r50_clear_day_gen_step1x_v1p2..."
if bsub -J "prove_BDD10k_pspnet_r50_clear_day_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_pspnet_r50_clear_day_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_BDD10k_pspnet_r50_clear_day_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda a ctivate prove && python unified_training.py --strateg y gen_step1x_v1p2 --datasets BDD10k --architectures p spnet_r50_clear_day --work-dir /scratch/aaa_exchange/ AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_pspnet_r50_clear_day_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_gen_step1x_v1p2
echo "Submitting prove_BDD10k_segformer_mit-b5_gen_step1x_v1p2..."
if bsub -J "prove_BDD10k_segformer_mit-b5_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda activ ate prove && python unified_training.py --strategy ge n_step1x_v1p2 --datasets BDD10k --architectures segfo rmer_mit-b5 --work-dir /scratch/aaa_exchange/AWARE/WE IGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_BDD10k_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2
echo "Submitting prove_BDD10k_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2..."
if bsub -J "prove_BDD10k_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2_%J.log" \
    -e "logs/prove_BDD10k_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2_%J.err" \
    "source ~/.bashrc && c onda activate prove && python unified_training.py --s trategy gen_step1x_v1p2 --datasets BDD10k --architect ures segformer_mit-b5_clear_day --work-dir /scratch/a aa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_BDD10k_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_gen_step1x_v1p2
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_gen_step1x_v1p2..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda acti vate prove && python unified_training.py --strategy g en_step1x_v1p2 --datasets IDD-AW --architectures deep labv3plus_r50 --work-dir /scratch/aaa_exchange/AWARE/ WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py -- strategy gen_step1x_v1p2 --datasets IDD-AW --architec tures deeplabv3plus_r50_clear_day --work-dir /scratch /aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_step1x_v1
                          p2"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_gen_step1x_v1p2
echo "Submitting prove_IDD-AW_pspnet_r50_gen_step1x_v1p2..."
if bsub -J "prove_IDD-AW_pspnet_r50_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda activate pr ove && python unified_training.py --strategy gen_step 1x_v1p2 --datasets IDD-AW --architectures pspnet_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_clear_day_gen_step1x_v1p2
echo "Submitting prove_IDD-AW_pspnet_r50_clear_day_gen_step1x_v1p2..."
if bsub -J "prove_IDD-AW_pspnet_r50_clear_day_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda a ctivate prove && python unified_training.py --strateg y gen_step1x_v1p2 --datasets IDD-AW --architectures p spnet_r50_clear_day --work-dir /scratch/aaa_exchange/ AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_clear_day_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_gen_step1x_v1p2
echo "Submitting prove_IDD-AW_segformer_mit-b5_gen_step1x_v1p2..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda activ ate prove && python unified_training.py --strategy ge n_step1x_v1p2 --datasets IDD-AW --architectures segfo rmer_mit-b5 --work-dir /scratch/aaa_exchange/AWARE/WE IGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2
echo "Submitting prove_IDD-AW_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2_%J.err" \
    "source ~/.bashrc && c onda activate prove && python unified_training.py --s trategy gen_step1x_v1p2 --datasets IDD-AW --architect ures segformer_mit-b5_clear_day --work-dir /scratch/a aa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_clear_day_gen_step1x_v1p
                          2"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_gen_step1x_v1p
                          2
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_gen_step1x_v1p
                          2..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_gen_step1x_v1p
                          2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_step1x_v1p
                          2_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_step1x_v1p
                          2_%J.err" \
    "source ~/.bashrc && c onda activate prove && python unified_training.py --s trategy gen_step1x_v1p2 --datasets MapillaryVistas -- architectures deeplabv3plus_r50 --work-dir /scratch/a aa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_gen_step1x_v1p
                          2"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          step1x_v1p2
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          step1x_v1p2..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          step1x_v1p2_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          step1x_v1p2_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_step1x_v1p2 --datasets Mapillar yVistas --architectures deeplabv3plus_r50_clear_day - -work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_gen_step1x_v1p2
echo "Submitting prove_MapillaryVistas_pspnet_r50_gen_step1x_v1p2..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_gen_step1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_gen_step1x_v1p2_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_gen_step1x_v1p2_%J.err" \
    "source ~/.bashrc && conda ac tivate prove && python unified_training.py --strategy gen_step1x_v1p2 --datasets MapillaryVistas --archite ctures pspnet_r50 --work-dir /scratch/aaa_exchange/AW ARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_gen_step1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_clear_day_gen_step1x_
                          v1p2
echo "Submitting prove_MapillaryVistas_pspnet_r50_clear_day_gen_step1x_
                          v1p2..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_clear_day_gen_step1x_
                          v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_step1x_
                          v1p2_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_step1x_
                          v1p2_%J.err" \
    "source ~/.bashrc & & conda activate prove && python unified_training.py --strategy gen_step1x_v1p2 --datasets MapillaryVistas --architectures pspnet_r50_clear_day --work-dir /scr atch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_clear_day_gen_step1x_
                          v1p2"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_gen_step1x_v1p2
                          
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_gen_step1x_v1p2
                          ..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_gen_step1x_v1p2
                          " \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_gen_step1x_v1p2
                          _%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_gen_step1x_v1p2
                          _%J.err" \
    "source ~/.bashrc && co nda activate prove && python unified_training.py --st rategy gen_step1x_v1p2 --datasets MapillaryVistas --a rchitectures segformer_mit-b5 --work-dir /scratch/aaa _exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_gen_step1x_v1p2
                          "
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_s
                          tep1x_v1p2
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_s
                          tep1x_v1p2..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_s
                          tep1x_v1p2" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_s
                          tep1x_v1p2_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_s
                          tep1x_v1p2_%J.err" \
    "source ~/.ba shrc && conda activate prove && python unified_traini ng.py --strategy gen_step1x_v1p2 --datasets Mapillary Vistas --architectures segformer_mit-b5_clear_day --w ork-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_s
                          tep1x_v1p2"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_gen_albumentations_weat
                          her
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_gen_albumentations_weat
                          her..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_gen_albumentations_weat
                          her" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_gen_albumentations_weat
                          her_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_gen_albumentations_weat
                          her_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py - -strategy gen_albumentations_weather --datasets IDD-A W --architectures deeplabv3plus_r50 --work-dir /scrat ch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_gen_albumentations_weat
                          her"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather_%J.err" \
    "source ~/ .bashrc && conda activate prove && python unified_tra ining.py --strategy gen_albumentations_weather --data sets IDD-AW --architectures deeplabv3plus_r50_clear_d ay --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_albumenta
                          tions_weather"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_gen_albumentations_weather
echo "Submitting prove_IDD-AW_pspnet_r50_gen_albumentations_weather..."
if bsub -J "prove_IDD-AW_pspnet_r50_gen_albumentations_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_gen_albumentations_weather_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_gen_albumentations_weather_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strate gy gen_albumentations_weather --datasets IDD-AW --arc hitectures pspnet_r50 --work-dir /scratch/aaa_exchang e/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_gen_albumentations_weather"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_clear_day_gen_albumentations_w
                          eather
echo "Submitting prove_IDD-AW_pspnet_r50_clear_day_gen_albumentations_w
                          eather..."
if bsub -J "prove_IDD-AW_pspnet_r50_clear_day_gen_albumentations_w
                          eather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_albumentations_w
                          eather_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_albumentations_w
                          eather_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.p y --strategy gen_albumentations_weather --datasets ID D-AW --architectures pspnet_r50_clear_day --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_clear_day_gen_albumentations_w
                          eather"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_gen_albumentations_weath
                          er
echo "Submitting prove_IDD-AW_segformer_mit-b5_gen_albumentations_weath
                          er..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_gen_albumentations_weath
                          er" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_gen_albumentations_weath
                          er_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_gen_albumentations_weath
                          er_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py -- strategy gen_albumentations_weather --datasets IDD-AW --architectures segformer_mit-b5 --work-dir /scratch /aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_gen_albumentations_weath
                          er"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather
echo "Submitting prove_IDD-AW_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_albumentations_weather --datase ts IDD-AW --architectures segformer_mit-b5_clear_day --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_clear_day_gen_albumentat
                          ions_weather"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_gen_albumentat
                          ions_weather
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_gen_albumentat
                          ions_weather..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_gen_albumentat
                          ions_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_albumentat
                          ions_weather_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_albumentat
                          ions_weather_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_albumentations_weather --datase ts MapillaryVistas --architectures deeplabv3plus_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_gen_albumentat
                          ions_weather"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          albumentations_weather
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          albumentations_weather..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          albumentations_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          albumentations_weather_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          albumentations_weather_%J.err" \
    "source ~/.bashrc && conda activate prove && python un ified_training.py --strategy gen_albumentations_weath er --datasets MapillaryVistas --architectures deeplab v3plus_r50_clear_day --work-dir /scratch/aaa_exchange /AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          albumentations_weather"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_gen_albumentations_we
                          ather
echo "Submitting prove_MapillaryVistas_pspnet_r50_gen_albumentations_we
                          ather..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_gen_albumentations_we
                          ather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_gen_albumentations_we
                          ather_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_gen_albumentations_we
                          ather_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strategy gen_albumentations_weather --datasets Map illaryVistas --architectures pspnet_r50 --work-dir /s cratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_gen_albumentations_we
                          ather"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_clear_day_gen_albumen
                          tations_weather
echo "Submitting prove_MapillaryVistas_pspnet_r50_clear_day_gen_albumen
                          tations_weather..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_clear_day_gen_albumen
                          tations_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_albumen
                          tations_weather_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_albumen
                          tations_weather_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_t raining.py --strategy gen_albumentations_weather --da tasets MapillaryVistas --architectures pspnet_r50_cle ar_day --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_clear_day_gen_albumen
                          tations_weather"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_gen_albumentati
                          ons_weather
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_gen_albumentati
                          ons_weather..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_gen_albumentati
                          ons_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_gen_albumentati
                          ons_weather_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_gen_albumentati
                          ons_weather_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_albumentations_weather --datase ts MapillaryVistas --architectures segformer_mit-b5 - -work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_gen_albumentati
                          ons_weather"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          lbumentations_weather
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          lbumentations_weather..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          lbumentations_weather" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          lbumentations_weather_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          lbumentations_weather_%J.err" \
    "s ource ~/.bashrc && conda activate prove && python uni fied_training.py --strategy gen_albumentations_weathe r --datasets MapillaryVistas --architectures segforme r_mit-b5_clear_day --work-dir /scratch/aaa_exchange/A WARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_a
                          lbumentations_weather"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_gen_cyclediffusion
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_gen_cyclediffusion..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_gen_cyclediffusion_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda a ctivate prove && python unified_training.py --strateg y gen_cyclediffusion --datasets IDD-AW --architecture s deeplabv3plus_r50 --work-dir /scratch/aaa_exchange/ AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion
echo "Submitting prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion..."
if bsub -J "prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion_%J.log" \
    -e "logs/prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strategy gen_cyclediffusion --datasets IDD-AW --ar chitectures deeplabv3plus_r50_clear_day --work-dir /s cratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_deeplabv3plus_r50_clear_day_gen_cyclediff
                          usion"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_gen_cyclediffusion
echo "Submitting prove_IDD-AW_pspnet_r50_gen_cyclediffusion..."
if bsub -J "prove_IDD-AW_pspnet_r50_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_gen_cyclediffusion_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strategy gen_c yclediffusion --datasets IDD-AW --architectures pspne t_r50 --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_pspnet_r50_clear_day_gen_cyclediffusion
echo "Submitting prove_IDD-AW_pspnet_r50_clear_day_gen_cyclediffusion..."
if bsub -J "prove_IDD-AW_pspnet_r50_clear_day_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_cyclediffusion_%J.log" \
    -e "logs/prove_IDD-AW_pspnet_r50_clear_day_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strat egy gen_cyclediffusion --datasets IDD-AW --architectu res pspnet_r50_clear_day --work-dir /scratch/aaa_exch ange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_pspnet_r50_clear_day_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_gen_cyclediffusion
echo "Submitting prove_IDD-AW_segformer_mit-b5_gen_cyclediffusion..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_gen_cyclediffusion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_gen_cyclediffusion_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_gen_cyclediffusion_%J.err" \
    "source ~/.bashrc && conda ac tivate prove && python unified_training.py --strategy gen_cyclediffusion --datasets IDD-AW --architectures segformer_mit-b5 --work-dir /scratch/aaa_exchange/AW ARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_gen_cyclediffusion"
    ((failed++)) || true
fi


# Job: prove_IDD-AW_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion
echo "Submitting prove_IDD-AW_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion..."
if bsub -J "prove_IDD-AW_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion_%J.log" \
    -e "logs/prove_IDD-AW_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion_%J.err" \
    "source ~/.bashrc & & conda activate prove && python unified_training.py --strategy gen_cyclediffusion --datasets IDD-AW --arc hitectures segformer_mit-b5_clear_day --work-dir /scr atch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_IDD-AW_segformer_mit-b5_clear_day_gen_cyclediffu
                          sion"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_gen_flux_konte
                          xt
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_gen_flux_konte
                          xt..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_gen_flux_konte
                          xt" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_flux_konte
                          xt_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_gen_flux_konte
                          xt_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py -- strategy gen_flux_kontext --datasets MapillaryVistas --architectures deeplabv3plus_r50 --work-dir /scratch /aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_gen_flux_konte
                          xt"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          flux_kontext
echo "Submitting prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          flux_kontext..."
if bsub -J "prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          flux_kontext" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          flux_kontext_%J.log" \
    -e "logs/prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          flux_kontext_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_flux_kontext --datasets Mapilla ryVistas --architectures deeplabv3plus_r50_clear_day --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_deeplabv3plus_r50_clear_day_gen_
                          flux_kontext"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_gen_flux_kontext
echo "Submitting prove_MapillaryVistas_pspnet_r50_gen_flux_kontext..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_gen_flux_kontext" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_gen_flux_kontext_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_gen_flux_kontext_%J.err" \
    "source ~/.bashrc && conda a ctivate prove && python unified_training.py --strateg y gen_flux_kontext --datasets MapillaryVistas --archi tectures pspnet_r50 --work-dir /scratch/aaa_exchange/ AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_gen_flux_kontext"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_pspnet_r50_clear_day_gen_flux_ko
                          ntext
echo "Submitting prove_MapillaryVistas_pspnet_r50_clear_day_gen_flux_ko
                          ntext..."
if bsub -J "prove_MapillaryVistas_pspnet_r50_clear_day_gen_flux_ko
                          ntext" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_flux_ko
                          ntext_%J.log" \
    -e "logs/prove_MapillaryVistas_pspnet_r50_clear_day_gen_flux_ko
                          ntext_%J.err" \
    "source ~/.bashrc && conda activate prove && python unified_training.py --strategy gen_flux_kontext --datasets MapillaryVist as --architectures pspnet_r50_clear_day --work-dir /s cratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_pspnet_r50_clear_day_gen_flux_ko
                          ntext"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_gen_flux_kontex
                          t
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_gen_flux_kontex
                          t..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_gen_flux_kontex
                          t" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_gen_flux_kontex
                          t_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_gen_flux_kontex
                          t_%J.err" \
    "source ~/.bashrc && c onda activate prove && python unified_training.py --s trategy gen_flux_kontext --datasets MapillaryVistas - -architectures segformer_mit-b5 --work-dir /scratch/a aa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_gen_flux_kontex
                          t"
    ((failed++)) || true
fi


# Job: prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_f
                          lux_kontext
echo "Submitting prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_f
                          lux_kontext..."
if bsub -J "prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_f
                          lux_kontext" \
    -q $QUEUE \
    -n 4 \
    -gpu "num=1:mode=exclusive_process:gmem=${GPU_MEM}" \
    -W $MAX_TIME \
    -o "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_f
                          lux_kontext_%J.log" \
    -e "logs/prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_f
                          lux_kontext_%J.err" \
    "source ~/.b ashrc && conda activate prove && python unified_train ing.py --strategy gen_flux_kontext --datasets Mapilla ryVistas --architectures segformer_mit-b5_clear_day - -work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"; then
    ((submitted++)) || true
else
    echo "  FAILED to submit prove_MapillaryVistas_segformer_mit-b5_clear_day_gen_f
                          lux_kontext"
    ((failed++)) || true
fi


echo ""
echo "=========================================="
echo "Submission complete!"
echo "Successfully submitted: $submitted"
echo "Failed: $failed"
echo "=========================================="

================================================================================
MapillaryVistas Model Retraining Assessment
Generated: 2026-01-21 17:28:14
================================================================================

## Summary
- Total MapillaryVistas models: 162
- Models needing retraining: 44 (CityscapesDataset)
- Models OK: 118 (MapillaryDataset_v1)

## Models Needing Retraining

These models used CityscapesDataset type, which means:
- Training validation metrics (mIoU) were computed using 19 Cityscapes classes
- But the model has 66 MapillaryVistas output classes
- The training loss was correct, only validation metrics were wrong

### Stage 1

- **baseline**: deeplabv3plus_r50, pspnet_r50, segformer_mit-b5
- **gen_Attribute_Hallucination**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_CNetSeg**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_CUT**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_IP2P**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_Img2Img**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_LANIT**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_Qwen_Image_Edit**: deeplabv3plus_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_SUSTechGAN**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_UniControl**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_VisualCloze**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_Weather_Effect_Generator**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_albumentations_weather**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_augmenters**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50
- **gen_automold**: deeplabv3plus_r50_ratio0p50, pspnet_r50_ratio0p50, segformer_mit-b5_ratio0p50

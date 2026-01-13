# Testing Progress Tracker

**Last Updated:** 2026-01-13 16:41


This document tracks the progress of fine-grained testing for trained models.


## Overview


### Test Job Types

| Type | Description | Output Location |
|------|-------------|-----------------|
| **Initial Test** | First test after training completes | `{weights}/test_results_detailed/` |
| **Retest (Fixed)** | Retest after fine_grained_test.py bug fix | `{weights}/test_results_detailed_fixed/` |

---


## Current Retest Jobs


### Retest Job Status

| Status | Count | Description |
|--------|-------|-------------|
| 🔄 Running | 0 | Currently testing |
| ⏳ Pending | 0 | Queued, waiting to run |
| ✅ Complete | 0 | Test results available |

### Retest Jobs by Dataset

| Dataset | Running | Pending | Complete | Total |
|---------|---------|---------|----------|-------|
| BDD10k | 0 | 0 | 0 | 0 |
| IDD-AW | 0 | 0 | 0 | 0 |
| MapillaryVistas | 0 | 0 | 0 | 0 |
| OUTSIDE15k | 0 | 0 | 0 | 0 |

---


## mIoU Results (Clear Day Training)


*mIoU values shown are the best across all models (deeplabv3plus, pspnet, segformer). Values are percentages.*


### 🏆 Top 10 Strategies (by Average mIoU)

| Rank | Strategy | Avg mIoU | Best Dataset | Best mIoU | Datasets |
|------|----------|----------|--------------|-----------|----------|
| 🥇 | gen_flux_kontext | 52.0 | MapillaryVistas | 52.0 | 1/4 |
| 🥈 | gen_TSIT | 48.8 | MapillaryVistas | 52.3 | 4/4 |
| 🥉 | gen_albumentations_weather | 48.8 | MapillaryVistas | 52.1 | 4/4 |
| 4. | gen_cycleGAN | 48.5 | MapillaryVistas | 52.1 | 4/4 |
| 5. | gen_UniControl | 48.5 | MapillaryVistas | 52.0 | 4/4 |
| 6. | gen_Qwen_Image_Edit | 48.2 | MapillaryVistas | 52.1 | 3/4 |
| 7. | std_mixup | 48.1 | MapillaryVistas | 51.7 | 2/4 |
| 8. | gen_automold | 47.5 | MapillaryVistas | 51.7 | 4/4 |
| 9. | gen_VisualCloze | 47.2 | MapillaryVistas | 51.7 | 4/4 |
| 10. | gen_SUSTechGAN | 47.1 | MapillaryVistas | 52.1 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 45.3 | 43.2 | ⏳ | 48.7 | 45.7 |
| gen_augmenters | 50.5 | 43.2 | 45.2 | 48.6 | 46.9 |
| gen_automold | 51.2 | 38.5 | 51.7 | 48.5 | 47.5 |
| gen_CNetSeg | 50.0 | 43.3 | 46.3 | 48.4 | 47.0 |
| gen_CUT | 44.5 | 25.5 | 51.8 | 48.3 | 42.5 |
| gen_cyclediffusion | 50.9 | 39.2 | ⏳ | ⏳ | 45.0 |
| gen_cycleGAN | 50.0 | 43.3 | 52.1 | 48.7 | 48.5 |
| gen_flux_kontext | ⏳ | ⏳ | 52.0 | ⏳ | 52.0 |
| gen_Img2Img | 42.9 | 43.2 | 52.1 | 48.4 | 46.6 |
| gen_IP2P | 43.3 | 43.4 | 51.9 | 48.2 | 46.7 |
| gen_LANIT | 40.4 | 43.3 | 52.3 | 48.6 | 46.2 |
| gen_Qwen_Image_Edit | ⏳ | 43.3 | 52.1 | 49.1 | 48.2 |
| gen_stargan_v2 | 50.7 | 43.1 | 51.8 | 36.0 | 45.4 |
| gen_step1x_new | ⏳ | 43.3 | 52.6 | 28.7 | 41.5 |
| gen_step1x_v1p2 | 51.1 | 43.2 | 51.9 | 27.9 | 43.5 |
| gen_SUSTechGAN | 45.2 | 43.0 | 52.1 | 48.3 | 47.1 |
| gen_TSIT | 51.2 | 43.3 | 52.3 | 48.5 | 48.8 |
| gen_UniControl | 50.3 | 43.2 | 52.0 | 48.3 | 48.5 |
| gen_VisualCloze | 50.4 | 38.0 | 51.7 | 48.6 | 47.2 |
| gen_Weather_Effect_Generator | 44.6 | 38.1 | 52.1 | 48.5 | 45.8 |
| gen_albumentations_weather | 51.3 | 43.0 | 52.1 | 48.7 | 48.8 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 43.2 | 36.6 | 51.0 | 48.8 | 44.9 |
| photometric_distort | 45.2 | 38.4 | 51.7 | 29.9 | 41.3 |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_autoaugment | 45.5 | ⏳ | 52.1 | 29.0 | 42.2 |
| std_cutmix | 45.3 | 43.1 | 52.0 | ⏳ | 46.8 |
| std_mixup | 44.6 | ⏳ | 51.7 | ⏳ | 48.1 |
| std_randaugment | 45.2 | ⏳ | 46.5 | 33.9 | 41.9 |

---


## Test Result Status Matrix


### Legend
- ✅ Test results available (mIoU extracted)
- 🔄 Test in progress
- ⏳ Pending test/retest
- ❌ Test failed (path issue, awaiting retest)
- ➖ Not applicable (no trained model)


### Generative Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| gen_Attribute_Hallucination | ✅ | ✅ | ⏳ | ✅ |
| gen_augmenters | ✅ | ✅ | ✅ | ✅ |
| gen_automold | ✅ | ✅ | ✅ | ✅ |
| gen_CNetSeg | ✅ | ✅ | ✅ | ✅ |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ | ⏳ | ⏳ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ⏳ | ⏳ | ✅ | ⏳ |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ | ✅ | ✅ |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ⏳ | ✅ | ✅ | ✅ |
| gen_stargan_v2 | ✅ | ✅ | ✅ | ✅ |
| gen_step1x_new | ⏳ | ✅ | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ | ✅ | ✅ |
| gen_SUSTechGAN | ✅ | ✅ | ✅ | ✅ |
| gen_TSIT | ✅ | ✅ | ✅ | ✅ |
| gen_UniControl | ✅ | ✅ | ✅ | ✅ |
| gen_VisualCloze | ✅ | ✅ | ✅ | ✅ |
| gen_Weather_Effect_Generator | ✅ | ✅ | ✅ | ✅ |
| gen_albumentations_weather | ✅ | ✅ | ✅ | ✅ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ✅ | ✅ | ✅ |
| photometric_distort | ✅ | ✅ | ✅ | ✅ |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ | ⏳ | ✅ | ✅ |
| std_cutmix | ✅ | ✅ | ✅ | ⏳ |
| std_mixup | ✅ | ⏳ | ✅ | ⏳ |
| std_randaugment | ✅ | ⏳ | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 24 | 0 | 4 | 0 |
| IDD-AW | 23 | 0 | 5 | 0 |
| MapillaryVistas | 25 | 0 | 3 | 0 |
| OUTSIDE15k | 23 | 0 | 5 | 0 |

---


## Job Management


### Check Test Job Status
```bash
# List all retest jobs
bjobs -u mima2416 | grep retest

# Count by status
bjobs -u mima2416 -o "JOB_NAME STAT" | grep retest | awk '{print $2}' | sort | uniq -c
```

### Submit Retest Jobs
```bash
cd scripts/retest_jobs_lsf
bash submit_all_retests.sh
```


---


## Per-Model Performance Breakdown


This section shows average mIoU per model architecture to help select which models to focus on for ratio ablation.


### Model Summary (Average mIoU)

| Model | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Average |
|-------|------:|------:|------:|------:|--------:|
| SegFormer | 50.71 | 38.44 | 51.97 | 48.54 | **47.41** |
| PSPNet | 44.92 | 28.47 | 46.29 | 34.35 | **38.51** |
| DeepLabV3+ | 42.21 | 29.83 | 44.98 | 28.83 | **36.46** |

### Recommendation for Ratio Ablation

Based on average mIoU performance, recommended models for ratio ablation:
1. **SegFormer** (segformer_mit-b5) - avg: 47.41
2. **PSPNet** (pspnet_r50) - avg: 38.51

To generate ratio ablation jobs with only these models:
```bash
python scripts/generate_ratio_ablation_jobs.py --models segformer_mit-b5 pspnet_r50
```
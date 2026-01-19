# Testing Progress Tracker

**Last Updated:** 2026-01-17 02:02


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
| 🥈 | gen_IP2P | 51.4 | MapillaryVistas | 51.9 | 2/4 |
| 🥉 | gen_CUT | 51.4 | MapillaryVistas | 51.8 | 2/4 |
| 4. | std_mixup | 51.3 | MapillaryVistas | 51.7 | 2/4 |
| 5. | std_cutmix | 51.3 | MapillaryVistas | 52.0 | 2/4 |
| 6. | gen_Attribute_Hallucination | 51.3 | BDD10k | 51.3 | 1/4 |
| 7. | gen_LANIT | 49.0 | MapillaryVistas | 52.3 | 3/4 |
| 8. | gen_albumentations_weather | 48.8 | MapillaryVistas | 52.1 | 3/4 |
| 9. | gen_SUSTechGAN | 48.7 | MapillaryVistas | 52.1 | 3/4 |
| 10. | gen_automold | 48.6 | MapillaryVistas | 51.7 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 51.3 | ⏳ | ⏳ | ⏳ | 51.3 |
| gen_augmenters | 50.5 | 43.2 | 45.2 | 48.6 | 46.9 |
| gen_automold | 51.2 | 43.1 | 51.7 | 48.5 | 48.6 |
| gen_CNetSeg | 50.0 | ⏳ | 46.3 | ⏳ | 48.2 |
| gen_CUT | 50.9 | ⏳ | 51.8 | ⏳ | 51.4 |
| gen_cyclediffusion | 50.9 | 43.2 | ⏳ | ⏳ | 47.0 |
| gen_cycleGAN | 50.0 | 43.3 | 52.1 | 48.7 | 48.5 |
| gen_flux_kontext | ⏳ | ⏳ | 52.0 | ⏳ | 52.0 |
| gen_Img2Img | 50.5 | 43.2 | 52.1 | ⏳ | 48.6 |
| gen_IP2P | 50.8 | ⏳ | 51.9 | ⏳ | 51.4 |
| gen_LANIT | 51.4 | 43.3 | 52.3 | ⏳ | 49.0 |
| gen_Qwen_Image_Edit | ⏳ | 43.3 | 52.1 | ⏳ | 47.7 |
| gen_stargan_v2 | 50.7 | 43.1 | 51.8 | 36.0 | 45.4 |
| gen_step1x_new | ⏳ | 43.3 | 52.6 | 28.7 | 41.5 |
| gen_step1x_v1p2 | 51.1 | 43.2 | 51.9 | 27.9 | 43.5 |
| gen_SUSTechGAN | 51.0 | 43.0 | 52.1 | ⏳ | 48.7 |
| gen_TSIT | 51.2 | 43.3 | ⏳ | ⏳ | 47.2 |
| gen_UniControl | 50.3 | 43.2 | 52.0 | ⏳ | 48.5 |
| gen_VisualCloze | 50.4 | 43.3 | 51.7 | ⏳ | 48.4 |
| gen_Weather_Effect_Generator | 44.6 | 43.1 | 52.1 | 34.9 | 43.7 |
| gen_albumentations_weather | 51.3 | 43.0 | 52.1 | ⏳ | 48.8 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 49.2 | ⏳ | 51.0 | 26.5 | 42.2 |
| photometric_distort | 51.0 | ⏳ | 51.7 | 29.9 | 44.2 |
| std_autoaugment | 50.9 | ⏳ | 52.1 | 29.0 | 44.0 |
| std_cutmix | 50.6 | ⏳ | 52.0 | ⏳ | 51.3 |
| std_mixup | 50.9 | ⏳ | 51.7 | ⏳ | 51.3 |
| std_randaugment | 50.8 | ⏳ | 46.5 | 33.9 | 43.8 |

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
| gen_Attribute_Hallucination | ✅ | ❌ | ⏳ | ❌ |
| gen_augmenters | ✅ | ✅ | ✅ | ✅ |
| gen_automold | ✅ | ✅ | ✅ | ✅ |
| gen_CNetSeg | ✅ | ❌ | ✅ | ❌ |
| gen_CUT | ✅ | ⏳ | ✅ | ❌ |
| gen_cyclediffusion | ✅ | ✅ | ⏳ | ⏳ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ⏳ | ⏳ | ✅ | ⏳ |
| gen_Img2Img | ✅ | ✅ | ✅ | ❌ |
| gen_IP2P | ✅ | ❌ | ✅ | ❌ |
| gen_LANIT | ✅ | ✅ | ✅ | ❌ |
| gen_Qwen_Image_Edit | ⏳ | ✅ | ✅ | ❌ |
| gen_stargan_v2 | ✅ | ✅ | ✅ | ✅ |
| gen_step1x_new | ⏳ | ✅ | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ | ✅ | ✅ |
| gen_SUSTechGAN | ✅ | ✅ | ✅ | ❌ |
| gen_TSIT | ✅ | ✅ | ⏳ | ⏳ |
| gen_UniControl | ✅ | ✅ | ✅ | ❌ |
| gen_VisualCloze | ✅ | ✅ | ✅ | ❌ |
| gen_Weather_Effect_Generator | ✅ | ✅ | ✅ | ✅ |
| gen_albumentations_weather | ✅ | ✅ | ✅ | ❌ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ❌ | ✅ | ✅ |
| photometric_distort | ✅ | ❌ | ✅ | ✅ |
| std_autoaugment | ✅ | ❌ | ✅ | ✅ |
| std_cutmix | ✅ | ❌ | ✅ | ⏳ |
| std_mixup | ✅ | ❌ | ✅ | ⏳ |
| std_randaugment | ✅ | ❌ | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 24 | 0 | 3 | 0 |
| IDD-AW | 16 | 0 | 2 | 0 |
| MapillaryVistas | 24 | 0 | 3 | 0 |
| OUTSIDE15k | 11 | 0 | 5 | 0 |

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
| SegFormer | 52.01 | - | - | - | **52.01** |
| PSPNet | 43.67 | - | - | - | **43.67** |
| DeepLabV3+ | 42.03 | - | - | - | **42.03** |

### Recommendation for Ratio Ablation

Based on average mIoU performance, recommended models for ratio ablation:
1. **SegFormer** (segformer_mit-b5) - avg: 52.01
2. **PSPNet** (pspnet_r50) - avg: 43.67

To generate ratio ablation jobs with only these models:
```bash
python scripts/generate_ratio_ablation_jobs.py --models segformer_mit-b5 pspnet_r50
```
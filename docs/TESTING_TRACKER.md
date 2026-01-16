# Testing Progress Tracker

**Last Updated:** 2026-01-16 10:29


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
| 🥇 | gen_cyclediffusion | 55.8 | MapillaryVistas | 65.3 | 4/4 |
| 🥈 | std_randaugment | 55.6 | OUTSIDE15k | 63.8 | 3/4 |
| 🥉 | photometric_distort | 55.6 | OUTSIDE15k | 64.0 | 3/4 |
| 4. | std_minimal | 52.9 | MapillaryVistas | 65.2 | 4/4 |
| 5. | gen_step1x_new | 52.8 | OUTSIDE15k | 64.1 | 4/4 |
| 6. | std_autoaugment | 52.5 | OUTSIDE15k | 63.8 | 4/4 |
| 7. | gen_step1x_v1p2 | 52.5 | OUTSIDE15k | 64.0 | 4/4 |
| 8. | std_cutmix | 52.4 | OUTSIDE15k | 63.9 | 4/4 |
| 9. | std_mixup | 52.4 | OUTSIDE15k | 63.9 | 4/4 |
| 10. | gen_flux_kontext | 52.4 | OUTSIDE15k | 63.8 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 51.3 | 43.1 | 52.5 | 48.7 | 48.9 |
| gen_augmenters | 50.5 | 43.2 | 51.8 | 48.6 | 48.5 |
| gen_automold | 51.2 | 43.1 | 51.7 | 48.5 | 48.6 |
| gen_CNetSeg | 50.0 | ⏳ | 52.6 | 48.4 | 50.4 |
| gen_CUT | 50.9 | 43.2 | 51.8 | 48.3 | 48.5 |
| gen_cyclediffusion | 50.9 | 43.2 | 65.3 | 63.8 | 55.8 |
| gen_cycleGAN | 50.0 | 43.3 | 52.1 | 48.7 | 48.5 |
| gen_flux_kontext | 50.8 | 43.1 | 52.0 | 63.8 | 52.4 |
| gen_Img2Img | 50.5 | 43.2 | 52.1 | 48.4 | 48.5 |
| gen_IP2P | 50.8 | ⏳ | 51.9 | 48.2 | 50.3 |
| gen_LANIT | 51.4 | 43.3 | 52.3 | 48.6 | 48.9 |
| gen_Qwen_Image_Edit | 50.7 | 43.3 | 52.1 | 49.1 | 48.8 |
| gen_stargan_v2 | 50.7 | 43.1 | 51.8 | 63.9 | 52.4 |
| gen_step1x_new | 51.1 | 43.3 | 52.6 | 64.1 | 52.8 |
| gen_step1x_v1p2 | 51.1 | 43.2 | 51.9 | 64.0 | 52.5 |
| gen_SUSTechGAN | 51.0 | 43.0 | 52.1 | 48.3 | 48.6 |
| gen_TSIT | 51.2 | 43.3 | 52.3 | 48.5 | 48.8 |
| gen_UniControl | 50.3 | 43.2 | 52.0 | 48.3 | 48.5 |
| gen_VisualCloze | 50.4 | 43.3 | 51.7 | 48.6 | 48.5 |
| gen_Weather_Effect_Generator | 44.6 | 43.1 | 52.1 | 48.5 | 47.1 |
| gen_albumentations_weather | 51.3 | 43.0 | 52.1 | 48.7 | 48.8 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 49.2 | 42.0 | 51.0 | 48.8 | 47.7 |
| photometric_distort | 51.0 | ⏳ | 51.7 | 64.0 | 55.6 |
| std_minimal | 44.0 | 38.3 | 65.2 | 64.1 | 52.9 |
| std_autoaugment | 50.9 | 43.3 | 52.1 | 63.8 | 52.5 |
| std_cutmix | 50.6 | 43.3 | 52.0 | 63.9 | 52.4 |
| std_mixup | 50.9 | 43.2 | 51.7 | 63.9 | 52.4 |
| std_randaugment | 50.8 | ⏳ | 52.3 | 63.8 | 55.6 |

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
| gen_Attribute_Hallucination | ✅ | ✅ | ✅ | ✅ |
| gen_augmenters | ✅ | ✅ | ✅ | ✅ |
| gen_automold | ✅ | ✅ | ✅ | ✅ |
| gen_CNetSeg | ✅ | ❌ | ✅ | ✅ |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ | ✅ | ✅ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ✅ | ✅ | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |
| gen_IP2P | ✅ | ❌ | ✅ | ✅ |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ✅ | ✅ | ✅ | ✅ |
| gen_stargan_v2 | ✅ | ✅ | ✅ | ✅ |
| gen_step1x_new | ✅ | ✅ | ✅ | ✅ |
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
| photometric_distort | ✅ | ❌ | ✅ | ✅ |
| std_minimal | ✅ | ✅ | ✅ | ✅ |
| std_autoaugment | ✅ | ✅ | ✅ | ✅ |
| std_cutmix | ✅ | ✅ | ✅ | ✅ |
| std_mixup | ✅ | ✅ | ✅ | ✅ |
| std_randaugment | ✅ | ❌ | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 28 | 0 | 0 | 0 |
| IDD-AW | 24 | 0 | 0 | 0 |
| MapillaryVistas | 28 | 0 | 0 | 0 |
| OUTSIDE15k | 28 | 0 | 0 | 0 |

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
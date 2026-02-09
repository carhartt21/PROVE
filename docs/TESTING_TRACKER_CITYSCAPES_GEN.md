# Testing Progress Tracker

**Last Updated:** 2026-02-09 21:48


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
| Cityscapes | 0 | 0 | 0 | 0 |
| ACDC (cross-domain) | 0 | 0 | 0 | 0 |

---


## mIoU Results (Clear Day Training)


*mIoU values shown are the best across all models (deeplabv3plus, pspnet, segformer). Values are percentages.*


### 🏆 Top 10 Strategies (by Average mIoU)

| Rank | Strategy | Avg mIoU | Best Dataset | Best mIoU | Datasets |
|------|----------|----------|--------------|-----------|----------|
| 🥇 | gen_VisualCloze | 60.7 | Cityscapes | 69.8 | 2/4 |
| 🥈 | std_autoaugment | 60.6 | Cityscapes | 68.7 | 2/4 |
| 🥉 | std_randaugment | 60.3 | Cityscapes | 68.4 | 2/4 |
| 4. | gen_step1x_v1p2 | 60.3 | Cityscapes | 69.7 | 2/4 |
| 5. | gen_flux_kontext | 60.3 | Cityscapes | 68.7 | 2/4 |
| 6. | std_cutmix | 60.2 | Cityscapes | 69.1 | 2/4 |
| 7. | baseline | 60.0 | Cityscapes | 69.0 | 2/4 |
| 8. | std_mixup | 59.9 | Cityscapes | 69.0 | 2/4 |
| 9. | gen_albumentations_weather | 59.0 | Cityscapes | 67.8 | 2/4 |
| 10. | gen_step1x_new | 58.8 | Cityscapes | 66.6 | 2/4 |


### Generative Image Augmentation Strategies

| Strategy | Cityscapes | ACDC (cross-domain) | Avg |
|----------|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 63.0 | 45.3 | 54.1 |
| gen_augmenters | 64.0 | 47.1 | 55.5 |
| gen_automold | 66.0 | 50.3 | 58.2 |
| gen_CNetSeg | 62.4 | 43.7 | 53.1 |
| gen_CUT | 62.1 | 45.1 | 53.6 |
| gen_cyclediffusion | 63.3 | 52.3 | 57.8 |
| gen_cycleGAN | ⏳ | ⏳ | - |
| gen_flux_kontext | 68.7 | 51.8 | 60.3 |
| gen_Img2Img | 63.7 | 46.1 | 54.9 |
| gen_IP2P | 63.5 | 44.4 | 53.9 |
| gen_LANIT | ⏳ | ⏳ | - |
| gen_Qwen_Image_Edit | 63.9 | 45.9 | 54.9 |
| gen_stargan_v2 | 63.1 | 45.5 | 54.3 |
| gen_step1x_new | 66.6 | 51.0 | 58.8 |
| gen_step1x_v1p2 | 69.7 | 51.0 | 60.3 |
| gen_SUSTechGAN | 63.8 | 50.9 | 57.3 |
| gen_TSIT | 63.5 | 43.9 | 53.7 |
| gen_UniControl | 62.6 | 46.1 | 54.3 |
| gen_VisualCloze | 69.8 | 51.6 | 60.7 |
| gen_Weather_Effect_Generator | 63.5 | 46.5 | 55.0 |
| gen_albumentations_weather | 67.8 | 50.3 | 59.0 |

### Standard Augmentation Strategies

| Strategy | Cityscapes | ACDC (cross-domain) | Avg |
|----------|-------:|-------:|-------:|
| baseline | 69.0 | 51.1 | 60.0 |
| std_minimal | ⏳ | ⏳ | - |
| std_photometric_distort | ⏳ | ⏳ | - |
| std_autoaugment | 68.7 | 52.5 | 60.6 |
| std_cutmix | 69.1 | 51.3 | 60.2 |
| std_mixup | 69.0 | 50.9 | 59.9 |
| std_randaugment | 68.4 | 52.2 | 60.3 |

---


## Test Result Status Matrix


### Legend
- ✅ Test results available (mIoU extracted)
- 🔄 Test in progress
- ⏳ Pending test/retest
- ❌ Test failed (path issue, awaiting retest)
- ➖ Not applicable (no trained model)


### Generative Strategies Status

| Strategy | Cityscapes | ACDC (cross-domain) |
|----------|--------|--------|
| gen_Attribute_Hallucination | ✅ | ✅ |
| gen_augmenters | ✅ | ✅ |
| gen_automold | ✅ | ✅ |
| gen_CNetSeg | ✅ | ✅ |
| gen_CUT | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ |
| gen_cycleGAN | ⏳ | ⏳ |
| gen_flux_kontext | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ |
| gen_LANIT | ⏳ | ⏳ |
| gen_Qwen_Image_Edit | ✅ | ✅ |
| gen_stargan_v2 | ✅ | ✅ |
| gen_step1x_new | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ |
| gen_SUSTechGAN | ✅ | ✅ |
| gen_TSIT | ✅ | ✅ |
| gen_UniControl | ✅ | ✅ |
| gen_VisualCloze | ✅ | ✅ |
| gen_Weather_Effect_Generator | ✅ | ✅ |
| gen_albumentations_weather | ✅ | ✅ |

### Standard Strategies Status

| Strategy | Cityscapes | ACDC (cross-domain) |
|----------|--------|--------|
| baseline | ✅ | ✅ |
| std_minimal | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ |
| std_autoaugment | ✅ | ✅ |
| std_cutmix | ✅ | ✅ |
| std_mixup | ✅ | ✅ |
| std_randaugment | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| Cityscapes | 24 | 0 | 4 | 0 |
| ACDC (cross-domain) | 24 | 0 | 4 | 0 |

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

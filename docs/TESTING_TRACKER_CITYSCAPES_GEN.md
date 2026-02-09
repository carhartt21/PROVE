# Testing Progress Tracker

**Last Updated:** 2026-02-09 21:41


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

---


## mIoU Results (Clear Day Training)


*mIoU values shown are the best across all models (deeplabv3plus, pspnet, segformer). Values are percentages.*


### 🏆 Top 10 Strategies (by Average mIoU)

| Rank | Strategy | Avg mIoU | Best Dataset | Best mIoU | Datasets |
|------|----------|----------|--------------|-----------|----------|
| 🥇 | gen_VisualCloze | 69.8 | Cityscapes | 69.8 | 1/4 |
| 🥈 | gen_step1x_v1p2 | 69.7 | Cityscapes | 69.7 | 1/4 |
| 🥉 | std_cutmix | 69.1 | Cityscapes | 69.1 | 1/4 |
| 4. | baseline | 69.0 | Cityscapes | 69.0 | 1/4 |
| 5. | std_mixup | 69.0 | Cityscapes | 69.0 | 1/4 |
| 6. | gen_flux_kontext | 68.7 | Cityscapes | 68.7 | 1/4 |
| 7. | std_autoaugment | 68.7 | Cityscapes | 68.7 | 1/4 |
| 8. | std_randaugment | 68.4 | Cityscapes | 68.4 | 1/4 |
| 9. | gen_albumentations_weather | 67.8 | Cityscapes | 67.8 | 1/4 |
| 10. | gen_step1x_new | 66.6 | Cityscapes | 66.6 | 1/4 |


### Generative Image Augmentation Strategies

| Strategy | Cityscapes | Avg |
|----------|-------:|-------:|
| gen_Attribute_Hallucination | 63.0 | 63.0 |
| gen_augmenters | 64.0 | 64.0 |
| gen_automold | 66.0 | 66.0 |
| gen_CNetSeg | 62.4 | 62.4 |
| gen_CUT | 62.1 | 62.1 |
| gen_cyclediffusion | 63.3 | 63.3 |
| gen_cycleGAN | ⏳ | - |
| gen_flux_kontext | 68.7 | 68.7 |
| gen_Img2Img | 63.7 | 63.7 |
| gen_IP2P | 63.5 | 63.5 |
| gen_LANIT | ⏳ | - |
| gen_Qwen_Image_Edit | 63.9 | 63.9 |
| gen_stargan_v2 | 63.1 | 63.1 |
| gen_step1x_new | 66.6 | 66.6 |
| gen_step1x_v1p2 | 69.7 | 69.7 |
| gen_SUSTechGAN | 63.8 | 63.8 |
| gen_TSIT | 63.5 | 63.5 |
| gen_UniControl | 62.6 | 62.6 |
| gen_VisualCloze | 69.8 | 69.8 |
| gen_Weather_Effect_Generator | 63.5 | 63.5 |
| gen_albumentations_weather | 67.8 | 67.8 |

### Standard Augmentation Strategies

| Strategy | Cityscapes | Avg |
|----------|-------:|-------:|
| baseline | 69.0 | 69.0 |
| std_minimal | ⏳ | - |
| std_photometric_distort | ⏳ | - |
| std_autoaugment | 68.7 | 68.7 |
| std_cutmix | 69.1 | 69.1 |
| std_mixup | 69.0 | 69.0 |
| std_randaugment | 68.4 | 68.4 |

---


## Test Result Status Matrix


### Legend
- ✅ Test results available (mIoU extracted)
- 🔄 Test in progress
- ⏳ Pending test/retest
- ❌ Test failed (path issue, awaiting retest)
- ➖ Not applicable (no trained model)


### Generative Strategies Status

| Strategy | Cityscapes |
|----------|--------|
| gen_Attribute_Hallucination | ✅ |
| gen_augmenters | ✅ |
| gen_automold | ✅ |
| gen_CNetSeg | ✅ |
| gen_CUT | ✅ |
| gen_cyclediffusion | ✅ |
| gen_cycleGAN | ⏳ |
| gen_flux_kontext | ✅ |
| gen_Img2Img | ✅ |
| gen_IP2P | ✅ |
| gen_LANIT | ⏳ |
| gen_Qwen_Image_Edit | ✅ |
| gen_stargan_v2 | ✅ |
| gen_step1x_new | ✅ |
| gen_step1x_v1p2 | ✅ |
| gen_SUSTechGAN | ✅ |
| gen_TSIT | ✅ |
| gen_UniControl | ✅ |
| gen_VisualCloze | ✅ |
| gen_Weather_Effect_Generator | ✅ |
| gen_albumentations_weather | ✅ |

### Standard Strategies Status

| Strategy | Cityscapes |
|----------|--------|
| baseline | ✅ |
| std_minimal | ⏳ |
| std_photometric_distort | ⏳ |
| std_autoaugment | ✅ |
| std_cutmix | ✅ |
| std_mixup | ✅ |
| std_randaugment | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| Cityscapes | 24 | 0 | 4 | 0 |

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

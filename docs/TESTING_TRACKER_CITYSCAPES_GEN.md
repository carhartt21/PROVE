# Testing Progress Tracker

**Last Updated:** 2026-02-08 08:38


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


### Generative Image Augmentation Strategies

| Strategy | Cityscapes | Avg |
|----------|-------:|-------:|
| gen_Attribute_Hallucination | ⏳ | - |
| gen_augmenters | ⏳ | - |
| gen_automold | ⏳ | - |
| gen_CNetSeg | ⏳ | - |
| gen_CUT | ⏳ | - |
| gen_cyclediffusion | ⏳ | - |
| gen_cycleGAN | ⏳ | - |
| gen_flux_kontext | ⏳ | - |
| gen_Img2Img | ⏳ | - |
| gen_IP2P | ⏳ | - |
| gen_LANIT | ⏳ | - |
| gen_Qwen_Image_Edit | ⏳ | - |
| gen_stargan_v2 | ⏳ | - |
| gen_step1x_new | ⏳ | - |
| gen_step1x_v1p2 | ⏳ | - |
| gen_SUSTechGAN | ⏳ | - |
| gen_TSIT | ⏳ | - |
| gen_UniControl | ⏳ | - |
| gen_VisualCloze | ⏳ | - |
| gen_Weather_Effect_Generator | ⏳ | - |
| gen_albumentations_weather | ⏳ | - |

### Standard Augmentation Strategies

| Strategy | Cityscapes | Avg |
|----------|-------:|-------:|
| baseline | ⏳ | - |
| std_minimal | ⏳ | - |
| std_photometric_distort | ⏳ | - |
| std_autoaugment | ⏳ | - |
| std_cutmix | ⏳ | - |
| std_mixup | ⏳ | - |
| std_randaugment | ⏳ | - |

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
| gen_Attribute_Hallucination | ⏳ |
| gen_augmenters | ⏳ |
| gen_automold | ❌ |
| gen_CNetSeg | ⏳ |
| gen_CUT | ⏳ |
| gen_cyclediffusion | ⏳ |
| gen_cycleGAN | ⏳ |
| gen_flux_kontext | ❌ |
| gen_Img2Img | ⏳ |
| gen_IP2P | ⏳ |
| gen_LANIT | ⏳ |
| gen_Qwen_Image_Edit | ⏳ |
| gen_stargan_v2 | ⏳ |
| gen_step1x_new | ❌ |
| gen_step1x_v1p2 | ⏳ |
| gen_SUSTechGAN | ⏳ |
| gen_TSIT | ⏳ |
| gen_UniControl | ⏳ |
| gen_VisualCloze | ⏳ |
| gen_Weather_Effect_Generator | ⏳ |
| gen_albumentations_weather | ❌ |

### Standard Strategies Status

| Strategy | Cityscapes |
|----------|--------|
| baseline | ❌ |
| std_minimal | ⏳ |
| std_photometric_distort | ⏳ |
| std_autoaugment | ❌ |
| std_cutmix | ❌ |
| std_mixup | ❌ |
| std_randaugment | ❌ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| Cityscapes | 0 | 0 | 19 | 0 |

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

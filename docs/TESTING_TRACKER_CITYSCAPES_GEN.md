# Testing Progress Tracker

**Last Updated:** 2026-02-16 13:40


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
| 🥇 | gen_CUT | 61.4 | Cityscapes | 70.0 | 2/4 |
| 🥈 | gen_augmenters | 61.1 | Cityscapes | 70.0 | 2/4 |
| 🥉 | gen_cyclediffusion | 61.0 | Cityscapes | 69.7 | 2/4 |
| 4. | gen_VisualCloze | 60.7 | Cityscapes | 69.8 | 2/4 |
| 5. | gen_Attribute_Hallucination | 60.7 | Cityscapes | 69.5 | 2/4 |
| 6. | gen_cycleGAN | 60.6 | Cityscapes | 69.2 | 2/4 |
| 7. | std_autoaugment | 60.6 | Cityscapes | 68.7 | 2/4 |
| 8. | gen_TSIT | 60.6 | Cityscapes | 69.0 | 2/4 |
| 9. | std_randaugment | 60.3 | Cityscapes | 68.4 | 2/4 |
| 10. | gen_step1x_v1p2 | 60.3 | Cityscapes | 69.7 | 2/4 |


### Generative Image Augmentation Strategies

| Strategy | Cityscapes | ACDC (cross-domain) | Avg |
|----------|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 69.5 | 51.9 | 60.7 |
| gen_augmenters | 70.0 | 52.1 | 61.1 |
| gen_automold | 66.0 | 50.3 | 58.2 |
| gen_CNetSeg | 66.7 | 51.7 | 59.2 |
| gen_CUT | 70.0 | 52.7 | 61.4 |
| gen_cyclediffusion | 69.7 | 52.3 | 61.0 |
| gen_cycleGAN | 69.2 | 51.9 | 60.6 |
| gen_flux_kontext | 68.7 | 51.8 | 60.3 |
| gen_Img2Img | 69.0 | 51.5 | 60.3 |
| gen_IP2P | 68.7 | 51.5 | 60.1 |
| gen_Qwen_Image_Edit | 67.9 | 50.9 | 59.4 |
| gen_stargan_v2 | 67.4 | 51.0 | 59.2 |
| gen_step1x_new | 66.6 | 51.0 | 58.8 |
| gen_step1x_v1p2 | 69.7 | 51.0 | 60.3 |
| gen_SUSTechGAN | 66.8 | 50.9 | 58.8 |
| gen_TSIT | 69.0 | 52.1 | 60.6 |
| gen_UniControl | 67.9 | 51.1 | 59.5 |
| gen_VisualCloze | 69.8 | 51.6 | 60.7 |
| gen_Weather_Effect_Generator | 65.8 | 50.2 | 58.0 |
| gen_albumentations_weather | 67.8 | 50.3 | 59.0 |

### Standard Augmentation Strategies

| Strategy | Cityscapes | ACDC (cross-domain) | Avg |
|----------|-------:|-------:|-------:|
| baseline | 69.0 | 51.1 | 60.0 |
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
| gen_cycleGAN | ✅ | ✅ |
| gen_flux_kontext | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ |
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
| std_autoaugment | ✅ | ✅ |
| std_cutmix | ✅ | ✅ |
| std_mixup | ✅ | ✅ |
| std_randaugment | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| Cityscapes | 25 | 0 | 0 | 0 |
| ACDC (cross-domain) | 25 | 0 | 0 | 0 |

---


## Job Management


### Check Test Job Status
```bash
# List all retest jobs
bjobs -u ${USER} | grep retest

# Count by status
bjobs -u ${USER} -o "JOB_NAME STAT" | grep retest | awk '{print $2}' | sort | uniq -c
```

### Submit Retest Jobs
```bash
cd scripts/retest_jobs_lsf
bash submit_all_retests.sh
```

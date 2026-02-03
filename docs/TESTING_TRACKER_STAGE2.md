# Testing Progress Tracker

**Last Updated:** 2026-02-03 06:57


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
| 🥇 | baseline | 46.0 | BDD10k | 49.8 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_automold | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_CUT | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_cyclediffusion | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_cycleGAN | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_flux_kontext | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_Img2Img | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_IP2P | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_LANIT | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_Qwen_Image_Edit | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_step1x_new | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_step1x_v1p2 | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_SUSTechGAN | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_UniControl | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_VisualCloze | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_albumentations_weather | ⏳ | ⏳ | ⏳ | ⏳ | - |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 49.8 | 44.4 | 41.9 | 47.6 | 46.0 |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_autoaugment | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_cutmix | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_mixup | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_randaugment | ⏳ | ⏳ | ⏳ | ⏳ | - |

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
| gen_Attribute_Hallucination | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_automold | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CUT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_cyclediffusion | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_cycleGAN | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_flux_kontext | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_Img2Img | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_IP2P | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_LANIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_Qwen_Image_Edit | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_new | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_v1p2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_SUSTechGAN | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_UniControl | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_VisualCloze | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_albumentations_weather | ⏳ | ⏳ | ⏳ | ⏳ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ✅ | ✅ | ✅ |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ⏳ | ⏳ | ⏳ | ⏳ |
| std_cutmix | ⏳ | ⏳ | ⏳ | ⏳ |
| std_mixup | ⏳ | ⏳ | ⏳ | ⏳ |
| std_randaugment | ⏳ | ⏳ | ⏳ | ⏳ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 1 | 0 | 27 | 0 |
| IDD-AW | 1 | 0 | 27 | 0 |
| MapillaryVistas | 1 | 0 | 27 | 0 |
| OUTSIDE15k | 1 | 0 | 27 | 0 |

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

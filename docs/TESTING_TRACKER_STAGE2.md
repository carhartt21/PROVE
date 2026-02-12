# Testing Progress Tracker

**Last Updated:** 2026-02-12 21:15


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
| 🥇 | baseline | 45.8 | BDD10k | 50.5 | 4/4 |
| 🥈 | gen_step1x_new | 44.5 | BDD10k | 48.7 | 4/4 |
| 🥉 | gen_Img2Img | 44.4 | BDD10k | 49.0 | 4/4 |
| 4. | gen_Qwen_Image_Edit | 44.4 | BDD10k | 48.3 | 4/4 |
| 5. | gen_UniControl | 44.4 | BDD10k | 48.9 | 4/4 |
| 6. | gen_CUT | 44.3 | BDD10k | 48.2 | 4/4 |
| 7. | gen_augmenters | 44.2 | BDD10k | 48.0 | 4/4 |
| 8. | std_cutmix | 43.3 | BDD10k | 51.6 | 4/4 |
| 9. | std_autoaugment | 43.0 | BDD10k | 48.4 | 4/4 |
| 10. | gen_flux_kontext | 42.5 | BDD10k | 48.2 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 46.7 | 40.6 | 34.8 | 45.0 | 41.8 |
| gen_augmenters | 48.0 | 41.0 | 41.2 | 46.5 | 44.2 |
| gen_automold | 47.1 | 40.6 | 35.0 | 44.4 | 41.8 |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_CUT | 48.2 | 42.8 | 41.1 | 45.2 | 44.3 |
| gen_cyclediffusion | 46.9 | 40.7 | 35.0 | 44.8 | 41.8 |
| gen_cycleGAN | 47.9 | 40.9 | 35.1 | 44.9 | 42.2 |
| gen_flux_kontext | 48.2 | 40.9 | 35.7 | 45.1 | 42.5 |
| gen_Img2Img | 49.0 | 42.8 | 41.0 | 44.8 | 44.4 |
| gen_IP2P | 47.7 | 40.5 | 35.0 | 44.7 | 42.0 |
| gen_LANIT | 46.6 | 40.6 | 34.7 | 45.1 | 41.8 |
| gen_Qwen_Image_Edit | 48.3 | 42.7 | 41.5 | 45.1 | 44.4 |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_step1x_new | 48.7 | 42.9 | 41.0 | 45.5 | 44.5 |
| gen_step1x_v1p2 | 46.4 | 40.5 | 35.0 | 44.9 | 41.7 |
| gen_SUSTechGAN | 46.4 | 40.7 | 34.9 | 44.8 | 41.7 |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_UniControl | 48.9 | 42.9 | 40.7 | 45.0 | 44.4 |
| gen_VisualCloze | 47.2 | 40.7 | 35.0 | 44.7 | 41.9 |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_albumentations_weather | 47.1 | 40.6 | 35.1 | 44.7 | 41.9 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 50.5 | 43.0 | 41.9 | 47.6 | 45.8 |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_autoaugment | 48.4 | 42.5 | 35.6 | 45.5 | 43.0 |
| std_cutmix | 51.6 | 40.9 | 35.8 | 45.0 | 43.3 |
| std_mixup | 48.1 | 40.9 | 35.7 | 44.5 | 42.3 |
| std_randaugment | 47.7 | 40.5 | 35.1 | 44.7 | 42.0 |

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
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ | ✅ | ✅ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ✅ | ✅ | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ | ✅ | ✅ |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ✅ | ✅ | ✅ | ✅ |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_new | ✅ | ✅ | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ | ✅ | ✅ |
| gen_SUSTechGAN | ✅ | ✅ | ✅ | ✅ |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_UniControl | ✅ | ✅ | ✅ | ✅ |
| gen_VisualCloze | ✅ | ✅ | ✅ | ✅ |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_albumentations_weather | ✅ | ✅ | ✅ | ✅ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ✅ | ✅ | ✅ |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ | ✅ | ✅ | ✅ |
| std_cutmix | ✅ | ✅ | ✅ | ✅ |
| std_mixup | ✅ | ✅ | ✅ | ✅ |
| std_randaugment | ✅ | ✅ | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 22 | 0 | 6 | 0 |
| IDD-AW | 22 | 0 | 6 | 0 |
| MapillaryVistas | 22 | 0 | 6 | 0 |
| OUTSIDE15k | 22 | 0 | 6 | 0 |

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

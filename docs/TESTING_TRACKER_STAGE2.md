# Testing Progress Tracker

**Last Updated:** 2026-02-12 09:52


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
| 🥈 | gen_Qwen_Image_Edit | 44.6 | BDD10k | 48.3 | 2/4 |
| 🥉 | gen_step1x_new | 44.5 | BDD10k | 48.7 | 4/4 |
| 4. | gen_Img2Img | 44.4 | BDD10k | 49.0 | 4/4 |
| 5. | std_autoaugment | 42.7 | BDD10k | 48.4 | 4/4 |
| 6. | std_cutmix | 42.6 | BDD10k | 48.6 | 4/4 |
| 7. | std_mixup | 42.3 | BDD10k | 48.1 | 4/4 |
| 8. | std_randaugment | 42.0 | BDD10k | 47.7 | 4/4 |
| 9. | gen_IP2P | 42.0 | BDD10k | 47.7 | 4/4 |
| 10. | gen_VisualCloze | 41.9 | BDD10k | 47.2 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 46.7 | 40.6 | 34.8 | 45.0 | 41.8 |
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_automold | 47.1 | 40.6 | 35.0 | 44.4 | 41.8 |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_CUT | 46.8 | 40.5 | 35.0 | 44.7 | 41.7 |
| gen_cyclediffusion | 46.9 | 40.7 | 35.0 | 44.8 | 41.8 |
| gen_cycleGAN | 46.9 | 40.5 | 34.9 | 44.7 | 41.7 |
| gen_flux_kontext | 47.0 | 40.4 | 35.0 | 44.9 | 41.8 |
| gen_Img2Img | 49.0 | 42.8 | 41.0 | 44.8 | 44.4 |
| gen_IP2P | 47.7 | 40.5 | 35.0 | 44.7 | 42.0 |
| gen_LANIT | 46.6 | 40.6 | 34.7 | 45.1 | 41.8 |
| gen_Qwen_Image_Edit | 48.3 | 40.8 | ⏳ | ⏳ | 44.6 |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_step1x_new | 48.7 | 42.9 | 41.0 | 45.5 | 44.5 |
| gen_step1x_v1p2 | 46.4 | 40.5 | 35.0 | 44.9 | 41.7 |
| gen_SUSTechGAN | 46.4 | 40.7 | 34.9 | 44.8 | 41.7 |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_UniControl | 46.4 | 40.7 | 35.0 | 45.0 | 41.8 |
| gen_VisualCloze | 47.2 | 40.7 | 35.0 | 44.7 | 41.9 |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ | - |
| gen_albumentations_weather | 47.1 | 40.6 | 35.1 | 44.7 | 41.9 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 50.5 | 43.0 | 41.9 | 47.6 | 45.8 |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_autoaugment | 48.4 | 41.2 | 35.6 | 45.5 | 42.7 |
| std_cutmix | 48.6 | 40.9 | 35.8 | 45.0 | 42.6 |
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
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_automold | ✅ | ✅ | ✅ | ✅ |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ | ✅ | ✅ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ✅ | ✅ | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ | ✅ | ✅ |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ✅ | ✅ | ⏳ | ⏳ |
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
| BDD10k | 21 | 0 | 7 | 0 |
| IDD-AW | 21 | 0 | 7 | 0 |
| MapillaryVistas | 20 | 0 | 8 | 0 |
| OUTSIDE15k | 20 | 0 | 8 | 0 |

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

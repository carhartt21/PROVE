# Testing Progress Tracker

**Last Updated:** 2026-02-03 07:00


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
| 🥇 | gen_VisualCloze | 51.1 | BDD10k | 51.1 | 1/4 |
| 🥈 | gen_LANIT | 49.7 | BDD10k | 49.7 | 1/4 |
| 🥉 | gen_automold | 49.6 | BDD10k | 49.6 | 1/4 |
| 4. | gen_flux_kontext | 49.4 | BDD10k | 49.4 | 1/4 |
| 5. | gen_cycleGAN | 49.2 | BDD10k | 49.2 | 1/4 |
| 6. | gen_SUSTechGAN | 49.1 | BDD10k | 49.1 | 1/4 |
| 7. | std_randaugment | 48.1 | BDD10k | 48.1 | 1/4 |
| 8. | gen_albumentations_weather | 46.4 | BDD10k | 46.4 | 1/4 |
| 9. | std_photometric_distort | 45.5 | BDD10k | 45.5 | 1/4 |
| 10. | std_minimal | 44.9 | BDD10k | 44.9 | 1/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | ⏳ | ⏳ | 34.8 | 36.2 | 35.5 |
| gen_augmenters | ⏳ | ⏳ | 34.9 | ⏳ | 34.9 |
| gen_automold | 49.6 | ⏳ | ⏳ | ⏳ | 49.6 |
| gen_CNetSeg | ⏳ | ⏳ | 34.5 | ⏳ | 34.5 |
| gen_CUT | ⏳ | ⏳ | 34.2 | ⏳ | 34.2 |
| gen_cyclediffusion | 48.9 | ⏳ | 34.4 | ⏳ | 41.6 |
| gen_cycleGAN | 49.2 | ⏳ | ⏳ | ⏳ | 49.2 |
| gen_flux_kontext | 49.4 | ⏳ | ⏳ | ⏳ | 49.4 |
| gen_Img2Img | ⏳ | ⏳ | 34.7 | ⏳ | 34.7 |
| gen_IP2P | 48.7 | ⏳ | 34.7 | ⏳ | 41.7 |
| gen_LANIT | 49.7 | ⏳ | ⏳ | ⏳ | 49.7 |
| gen_Qwen_Image_Edit | ⏳ | ⏳ | 34.7 | ⏳ | 34.7 |
| gen_stargan_v2 | ⏳ | ⏳ | 34.6 | ⏳ | 34.6 |
| gen_step1x_new | 49.7 | ⏳ | 29.0 | 42.5 | 40.4 |
| gen_step1x_v1p2 | 46.5 | ⏳ | ⏳ | 42.7 | 44.6 |
| gen_SUSTechGAN | 49.1 | ⏳ | ⏳ | ⏳ | 49.1 |
| gen_TSIT | ⏳ | ⏳ | 34.5 | ⏳ | 34.5 |
| gen_UniControl | ⏳ | ⏳ | 35.0 | ⏳ | 35.0 |
| gen_VisualCloze | 51.1 | ⏳ | ⏳ | ⏳ | 51.1 |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | 35.2 | ⏳ | 35.2 |
| gen_albumentations_weather | 46.4 | ⏳ | ⏳ | ⏳ | 46.4 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 41.3 | ⏳ | 34.6 | 38.7 | 38.2 |
| std_minimal | 44.9 | ⏳ | ⏳ | ⏳ | 44.9 |
| std_photometric_distort | 45.5 | ⏳ | ⏳ | ⏳ | 45.5 |
| std_autoaugment | 45.5 | ⏳ | 34.9 | ⏳ | 40.2 |
| std_cutmix | 48.9 | ⏳ | 34.9 | 43.0 | 42.2 |
| std_mixup | 49.1 | ⏳ | 34.9 | 42.8 | 42.3 |
| std_randaugment | 48.1 | ⏳ | ⏳ | ⏳ | 48.1 |

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
| gen_Attribute_Hallucination | ⏳ | ⏳ | ✅ | ✅ |
| gen_augmenters | ⏳ | ⏳ | ✅ | ⏳ |
| gen_automold | ✅ | ⏳ | ⏳ | ⏳ |
| gen_CNetSeg | ⏳ | ⏳ | ✅ | ⏳ |
| gen_CUT | ⏳ | ⏳ | ✅ | ⏳ |
| gen_cyclediffusion | ✅ | ⏳ | ✅ | ⏳ |
| gen_cycleGAN | ✅ | ⏳ | ⏳ | ⏳ |
| gen_flux_kontext | ✅ | ⏳ | ⏳ | ⏳ |
| gen_Img2Img | ⏳ | ⏳ | ✅ | ⏳ |
| gen_IP2P | ✅ | ⏳ | ✅ | ⏳ |
| gen_LANIT | ✅ | ⏳ | ⏳ | ⏳ |
| gen_Qwen_Image_Edit | ⏳ | ⏳ | ✅ | ⏳ |
| gen_stargan_v2 | ⏳ | ⏳ | ✅ | ⏳ |
| gen_step1x_new | ✅ | ⏳ | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ⏳ | ⏳ | ✅ |
| gen_SUSTechGAN | ✅ | ⏳ | ⏳ | ⏳ |
| gen_TSIT | ⏳ | ⏳ | ✅ | ⏳ |
| gen_UniControl | ⏳ | ⏳ | ✅ | ⏳ |
| gen_VisualCloze | ✅ | ⏳ | ⏳ | ⏳ |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ✅ | ⏳ |
| gen_albumentations_weather | ✅ | ⏳ | ⏳ | ⏳ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ⏳ | ✅ | ✅ |
| std_minimal | ✅ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ✅ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ | ⏳ | ✅ | ⏳ |
| std_cutmix | ✅ | ⏳ | ✅ | ✅ |
| std_mixup | ✅ | ⏳ | ✅ | ✅ |
| std_randaugment | ✅ | ⏳ | ⏳ | ⏳ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 18 | 0 | 10 | 0 |
| IDD-AW | 0 | 0 | 28 | 0 |
| MapillaryVistas | 17 | 0 | 11 | 0 |
| OUTSIDE15k | 6 | 0 | 22 | 0 |

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

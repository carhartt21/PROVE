# Testing Progress Tracker

**Last Updated:** 2026-01-24 13:01


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
| 🥇 | gen_LANIT | 45.9 | BDD10k | 51.4 | 4/4 |
| 🥈 | gen_TSIT | 45.8 | BDD10k | 51.2 | 4/4 |
| 🥉 | gen_Qwen_Image_Edit | 45.8 | BDD10k | 50.7 | 4/4 |
| 4. | gen_step1x_new | 45.8 | BDD10k | 51.1 | 4/4 |
| 5. | gen_flux_kontext | 45.8 | BDD10k | 50.8 | 4/4 |
| 6. | gen_augmenters | 45.8 | BDD10k | 50.5 | 4/4 |
| 7. | gen_Attribute_Hallucination | 45.8 | BDD10k | 51.3 | 4/4 |
| 8. | gen_albumentations_weather | 45.8 | BDD10k | 51.3 | 4/4 |
| 9. | std_cutmix | 45.8 | BDD10k | 50.6 | 4/4 |
| 10. | std_randaugment | 45.8 | BDD10k | 51.0 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 51.3 | 43.1 | 40.0 | 48.7 | 45.8 |
| gen_augmenters | 50.5 | 43.2 | 40.8 | 48.6 | 45.8 |
| gen_automold | 51.2 | 43.1 | 40.0 | 48.5 | 45.7 |
| gen_CNetSeg | 50.0 | 43.2 | 40.9 | 48.4 | 45.6 |
| gen_CUT | 50.9 | 43.2 | 40.3 | 48.3 | 45.6 |
| gen_cyclediffusion | 50.9 | 43.2 | 40.2 | 48.5 | 45.7 |
| gen_cycleGAN | 50.0 | 43.3 | 40.3 | 48.7 | 45.6 |
| gen_flux_kontext | 50.8 | 43.1 | 40.3 | 49.0 | 45.8 |
| gen_Img2Img | 50.5 | 43.2 | 40.1 | 48.4 | 45.5 |
| gen_IP2P | 50.8 | 43.0 | 40.6 | 48.2 | 45.7 |
| gen_LANIT | 51.4 | 43.3 | 40.4 | 48.6 | 45.9 |
| gen_Qwen_Image_Edit | 50.7 | 43.3 | 40.2 | 49.1 | 45.8 |
| gen_stargan_v2 | 50.7 | 43.1 | 40.2 | 48.1 | 45.5 |
| gen_step1x_new | 51.1 | 43.3 | 40.2 | 48.6 | 45.8 |
| gen_step1x_v1p2 | 51.1 | 43.2 | 40.2 | 48.4 | 45.7 |
| gen_SUSTechGAN | 51.0 | 43.0 | 40.2 | 48.3 | 45.6 |
| gen_TSIT | 51.2 | 43.3 | 40.3 | 48.5 | 45.8 |
| gen_UniControl | 50.3 | 43.2 | 40.1 | 48.3 | 45.5 |
| gen_VisualCloze | 50.4 | 43.3 | 39.9 | 48.6 | 45.5 |
| gen_Weather_Effect_Generator | 50.9 | 43.1 | 40.2 | 48.5 | 45.7 |
| gen_albumentations_weather | 51.3 | 43.0 | 40.1 | 48.7 | 45.8 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 49.2 | 42.0 | 39.5 | 48.8 | 44.9 |
| std_std_photometric_distort | 51.0 | 43.2 | 40.3 | 48.4 | 45.7 |
| std_autoaugment | 50.6 | 43.3 | 40.5 | 48.3 | 45.7 |
| std_cutmix | 50.6 | 43.3 | 40.2 | 49.0 | 45.8 |
| std_mixup | 51.3 | 43.0 | 39.9 | 48.3 | 45.6 |
| std_randaugment | 51.0 | 43.4 | 40.1 | 48.6 | 45.8 |

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
| gen_CNetSeg | ✅ | ✅ | ✅ | ✅ |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ | ✅ | ✅ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ✅ | ✅ | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ | ✅ | ✅ |
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
| std_std_photometric_distort | ✅ | ✅ | ✅ | ✅ |
| std_autoaugment | ✅ | ✅ | ✅ | ✅ |
| std_cutmix | ✅ | ✅ | ✅ | ✅ |
| std_mixup | ✅ | ✅ | ✅ | ✅ |
| std_randaugment | ✅ | ✅ | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 27 | 0 | 0 | 0 |
| IDD-AW | 27 | 0 | 0 | 0 |
| MapillaryVistas | 27 | 0 | 0 | 0 |
| OUTSIDE15k | 27 | 0 | 0 | 0 |

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

# Testing Progress Tracker

**Last Updated:** 2026-02-08 08:21


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
| 🥇 | gen_albumentations_weather | 46.4 | BDD10k | 46.4 | 1/4 |
| 🥈 | std_photometric_distort | 45.5 | BDD10k | 45.5 | 1/4 |
| 🥉 | std_minimal | 44.9 | BDD10k | 44.9 | 1/4 |
| 4. | gen_VisualCloze | 43.1 | BDD10k | 51.6 | 3/4 |
| 5. | gen_stargan_v2 | 43.1 | BDD10k | 50.9 | 3/4 |
| 6. | gen_Attribute_Hallucination | 42.9 | BDD10k | 50.5 | 3/4 |
| 7. | gen_LANIT | 42.8 | BDD10k | 50.1 | 3/4 |
| 8. | gen_UniControl | 42.8 | BDD10k | 50.3 | 3/4 |
| 9. | gen_CNetSeg | 42.7 | BDD10k | 49.8 | 3/4 |
| 10. | gen_TSIT | 42.7 | BDD10k | 50.7 | 3/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 50.5 | ⏳ | 34.9 | 43.4 | 42.9 |
| gen_augmenters | 49.1 | ⏳ | 34.9 | 43.5 | 42.5 |
| gen_automold | 49.5 | ⏳ | 35.0 | 43.1 | 42.5 |
| gen_CNetSeg | 49.8 | ⏳ | 35.0 | 43.3 | 42.7 |
| gen_CUT | 48.3 | ⏳ | 34.9 | 43.1 | 42.1 |
| gen_cyclediffusion | 48.9 | ⏳ | 34.4 | ⏳ | 41.6 |
| gen_cycleGAN | 49.6 | ⏳ | 35.0 | 43.1 | 42.5 |
| gen_flux_kontext | 48.7 | ⏳ | 35.2 | 43.2 | 42.4 |
| gen_Img2Img | 46.7 | ⏳ | 34.8 | 43.0 | 41.5 |
| gen_IP2P | 49.0 | ⏳ | 34.8 | 43.5 | 42.4 |
| gen_LANIT | 50.1 | ⏳ | 35.1 | 43.3 | 42.8 |
| gen_Qwen_Image_Edit | 49.5 | ⏳ | 35.1 | 43.5 | 42.7 |
| gen_stargan_v2 | 50.9 | ⏳ | 34.6 | 43.7 | 43.1 |
| gen_step1x_new | 49.9 | ⏳ | 35.0 | 42.9 | 42.6 |
| gen_step1x_v1p2 | 47.0 | ⏳ | 34.7 | 42.9 | 41.5 |
| gen_SUSTechGAN | 49.3 | ⏳ | 34.7 | 42.9 | 42.3 |
| gen_TSIT | 50.7 | ⏳ | 34.5 | 43.0 | 42.7 |
| gen_UniControl | 50.3 | ⏳ | 35.0 | 43.1 | 42.8 |
| gen_VisualCloze | 51.6 | ⏳ | 34.9 | 42.7 | 43.1 |
| gen_Weather_Effect_Generator | 48.8 | ⏳ | 35.2 | 42.9 | 42.3 |
| gen_albumentations_weather | 46.4 | ⏳ | ⏳ | ⏳ | 46.4 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 51.6 | ⏳ | 34.6 | 38.7 | 41.6 |
| std_minimal | 44.9 | ⏳ | ⏳ | ⏳ | 44.9 |
| std_photometric_distort | 45.5 | ⏳ | ⏳ | ⏳ | 45.5 |
| std_autoaugment | 49.4 | ⏳ | 34.9 | 43.2 | 42.5 |
| std_cutmix | 49.4 | ⏳ | 34.9 | 43.0 | 42.4 |
| std_mixup | 49.4 | ⏳ | 34.9 | 42.8 | 42.4 |
| std_randaugment | 48.6 | ⏳ | 34.8 | 43.4 | 42.3 |

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
| gen_Attribute_Hallucination | ✅ | ⏳ | ✅ | ✅ |
| gen_augmenters | ✅ | ⏳ | ✅ | ✅ |
| gen_automold | ✅ | ⏳ | ✅ | ✅ |
| gen_CNetSeg | ✅ | ⏳ | ✅ | ✅ |
| gen_CUT | ✅ | ⏳ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ⏳ | ✅ | ⏳ |
| gen_cycleGAN | ✅ | ⏳ | ✅ | ✅ |
| gen_flux_kontext | ✅ | ⏳ | ✅ | ✅ |
| gen_Img2Img | ✅ | ⏳ | ✅ | ✅ |
| gen_IP2P | ✅ | ⏳ | ✅ | ✅ |
| gen_LANIT | ✅ | ⏳ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ✅ | ⏳ | ✅ | ✅ |
| gen_stargan_v2 | ✅ | ⏳ | ✅ | ✅ |
| gen_step1x_new | ✅ | ⏳ | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ⏳ | ✅ | ✅ |
| gen_SUSTechGAN | ✅ | ⏳ | ✅ | ✅ |
| gen_TSIT | ✅ | ⏳ | ✅ | ✅ |
| gen_UniControl | ✅ | ⏳ | ✅ | ✅ |
| gen_VisualCloze | ✅ | ⏳ | ✅ | ✅ |
| gen_Weather_Effect_Generator | ✅ | ⏳ | ✅ | ✅ |
| gen_albumentations_weather | ✅ | ⏳ | ⏳ | ⏳ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ⏳ | ✅ | ✅ |
| std_minimal | ✅ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ✅ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ | ⏳ | ✅ | ✅ |
| std_cutmix | ✅ | ⏳ | ✅ | ✅ |
| std_mixup | ✅ | ⏳ | ✅ | ✅ |
| std_randaugment | ✅ | ⏳ | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 28 | 0 | 0 | 0 |
| IDD-AW | 0 | 0 | 28 | 0 |
| MapillaryVistas | 25 | 0 | 3 | 0 |
| OUTSIDE15k | 24 | 0 | 4 | 0 |

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

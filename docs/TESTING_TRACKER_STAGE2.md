# Testing Progress Tracker

**Last Updated:** 2026-02-13 23:12


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
| 🥉 | std_cutmix | 44.5 | BDD10k | 51.6 | 4/4 |
| 4. | gen_flux_kontext | 44.5 | BDD10k | 48.2 | 4/4 |
| 5. | std_autoaugment | 44.4 | BDD10k | 48.4 | 4/4 |
| 6. | gen_Img2Img | 44.4 | BDD10k | 49.0 | 4/4 |
| 7. | gen_Qwen_Image_Edit | 44.4 | BDD10k | 48.3 | 4/4 |
| 8. | gen_UniControl | 44.4 | BDD10k | 48.9 | 4/4 |
| 9. | gen_CUT | 44.3 | BDD10k | 48.2 | 4/4 |
| 10. | gen_cycleGAN | 44.3 | BDD10k | 50.4 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 46.7 | 40.6 | 41.1 | 45.0 | 43.4 |
| gen_augmenters | 48.0 | 41.0 | 41.2 | 46.5 | 44.2 |
| gen_automold | 47.1 | 40.6 | 40.8 | 45.0 | 43.4 |
| gen_CNetSeg | ⏳ | ⏳ | 41.0 | 46.6 | 43.8 |
| gen_CUT | 48.2 | 42.8 | 41.1 | 45.2 | 44.3 |
| gen_cyclediffusion | 46.9 | 40.7 | 40.7 | 45.2 | 43.4 |
| gen_cycleGAN | 50.4 | 40.9 | 41.2 | 44.9 | 44.3 |
| gen_flux_kontext | 48.2 | 42.4 | 40.9 | 46.4 | 44.5 |
| gen_Img2Img | 49.0 | 42.8 | 41.0 | 44.8 | 44.4 |
| gen_IP2P | 47.7 | 40.5 | 40.9 | 45.9 | 43.8 |
| gen_LANIT | 46.6 | 40.6 | 34.7 | 45.1 | 41.8 |
| gen_Qwen_Image_Edit | 48.3 | 42.7 | 41.5 | 45.1 | 44.4 |
| gen_stargan_v2 | ⏳ | ⏳ | 40.8 | 46.7 | 43.7 |
| gen_step1x_new | 48.7 | 42.9 | 41.0 | 45.5 | 44.5 |
| gen_step1x_v1p2 | 46.4 | 40.5 | 40.9 | 44.9 | 43.2 |
| gen_SUSTechGAN | 46.4 | 40.7 | 40.8 | 46.5 | 43.6 |
| gen_TSIT | ⏳ | ⏳ | 41.0 | 44.8 | 42.9 |
| gen_UniControl | 48.9 | 42.9 | 40.7 | 45.0 | 44.4 |
| gen_VisualCloze | 47.2 | 40.7 | 41.1 | 45.0 | 43.5 |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | 41.0 | 46.2 | 43.6 |
| gen_albumentations_weather | 48.5 | 40.6 | 41.2 | 46.6 | 44.2 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 50.5 | 43.0 | 41.9 | 47.6 | 45.8 |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_autoaugment | 48.4 | 42.5 | 41.3 | 45.5 | 44.4 |
| std_cutmix | 51.6 | 40.9 | 40.5 | 45.0 | 44.5 |
| std_mixup | 48.1 | 40.9 | 41.1 | 44.5 | 43.7 |
| std_randaugment | 48.3 | 40.9 | 40.8 | 44.8 | 43.7 |

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
| gen_CNetSeg | ⏳ | ⏳ | ✅ | ✅ |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ | ✅ | ✅ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ✅ | ✅ | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ | ✅ | ✅ |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ✅ | ✅ | ✅ | ✅ |
| gen_stargan_v2 | ⏳ | ⏳ | ✅ | ✅ |
| gen_step1x_new | ✅ | ✅ | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ | ✅ | ✅ |
| gen_SUSTechGAN | ✅ | ✅ | ✅ | ✅ |
| gen_TSIT | ⏳ | ⏳ | ✅ | ✅ |
| gen_UniControl | ✅ | ✅ | ✅ | ✅ |
| gen_VisualCloze | ✅ | ✅ | ✅ | ✅ |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ✅ | ✅ |
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
| MapillaryVistas | 26 | 0 | 2 | 0 |
| OUTSIDE15k | 26 | 0 | 2 | 0 |

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

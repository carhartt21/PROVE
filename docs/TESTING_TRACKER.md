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
| 🥇 | baseline | 44.7 | BDD10k | 51.6 | 4/4 |
| 🥈 | gen_UniControl | 44.5 | BDD10k | 52.5 | 4/4 |
| 🥉 | gen_SUSTechGAN | 44.2 | BDD10k | 49.3 | 4/4 |
| 4. | gen_cyclediffusion | 44.0 | BDD10k | 48.9 | 4/4 |
| 5. | std_autoaugment | 43.9 | BDD10k | 49.4 | 4/4 |
| 6. | gen_flux_kontext | 43.8 | BDD10k | 48.7 | 4/4 |
| 7. | gen_augmenters | 43.8 | BDD10k | 49.1 | 4/4 |
| 8. | gen_TSIT | 43.8 | BDD10k | 50.7 | 4/4 |
| 9. | gen_automold | 43.8 | BDD10k | 49.5 | 4/4 |
| 10. | gen_stargan_v2 | 43.7 | BDD10k | 50.9 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 50.5 | 40.0 | 35.3 | 43.9 | 42.4 |
| gen_augmenters | 49.1 | 40.2 | 40.3 | 45.7 | 43.8 |
| gen_automold | 49.5 | 40.6 | 40.7 | 44.2 | 43.8 |
| gen_CNetSeg | 49.8 | 40.2 | 35.5 | 45.5 | 42.8 |
| gen_CUT | 48.3 | 40.2 | 35.3 | 44.3 | 42.0 |
| gen_cyclediffusion | 48.9 | 41.2 | 40.3 | 45.3 | 44.0 |
| gen_cycleGAN | 49.6 | 40.4 | 40.4 | 43.5 | 43.5 |
| gen_flux_kontext | 48.7 | 40.4 | 40.7 | 45.5 | 43.8 |
| gen_Img2Img | 51.7 | 41.5 | 35.6 | 45.2 | 43.5 |
| gen_IP2P | 49.0 | 40.4 | 35.5 | 45.2 | 42.5 |
| gen_LANIT | 50.1 | 40.2 | 35.1 | 43.3 | 42.2 |
| gen_Qwen_Image_Edit | 49.5 | 40.4 | 40.5 | 43.5 | 43.5 |
| gen_stargan_v2 | 50.9 | 39.8 | 40.5 | 43.7 | 43.7 |
| gen_step1x_new | 49.9 | 40.6 | 35.2 | 45.4 | 42.8 |
| gen_step1x_v1p2 | 47.0 | 40.1 | 35.5 | 45.8 | 42.1 |
| gen_SUSTechGAN | 49.3 | 40.6 | 40.4 | 46.4 | 44.2 |
| gen_TSIT | 50.7 | 40.0 | 40.1 | 44.4 | 43.8 |
| gen_UniControl | 52.5 | 41.3 | 40.4 | 43.9 | 44.5 |
| gen_VisualCloze | 51.6 | 40.7 | 35.2 | 45.6 | 43.3 |
| gen_Weather_Effect_Generator | 48.8 | 39.8 | 35.2 | 45.2 | 42.3 |
| gen_albumentations_weather | 46.5 | 40.0 | 40.6 | 45.4 | 43.1 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 51.6 | 41.6 | 40.8 | 45.0 | 44.7 |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ | - |
| std_autoaugment | 49.4 | 40.5 | 40.5 | 45.3 | 43.9 |
| std_cutmix | 49.4 | 40.4 | 40.0 | 45.1 | 43.7 |
| std_mixup | 49.4 | 40.0 | 35.4 | 45.4 | 42.6 |
| std_randaugment | 48.6 | 40.2 | 40.4 | 43.5 | 43.2 |

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
| BDD10k | 26 | 0 | 2 | 0 |
| IDD-AW | 26 | 0 | 2 | 0 |
| MapillaryVistas | 26 | 0 | 2 | 0 |
| OUTSIDE15k | 26 | 0 | 2 | 0 |

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

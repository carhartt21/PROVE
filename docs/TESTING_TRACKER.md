# Testing Progress Tracker

**Last Updated:** 2026-01-12 22:19


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
| 🔄 Running | 12 | Currently testing |
| ⏳ Pending | 27 | Queued, waiting to run |
| ✅ Complete | 141 | Test results available |

### Retest Jobs by Dataset

| Dataset | Running | Pending | Complete | Total |
|---------|---------|---------|----------|-------|
| BDD10k | 0 | 0 | 14 | 14 |
| IDD-AW | 0 | 0 | 62 | 62 |
| MapillaryVistas | 12 | 19 | 34 | 65 |
| OUTSIDE15k | 0 | 8 | 31 | 39 |

---


## mIoU Results (Clear Day Training)


*mIoU values shown are the best across all models (deeplabv3plus, pspnet, segformer). Values are percentages.*


### 🏆 Top 10 Strategies (by Average mIoU)

| Rank | Strategy | Avg mIoU | Best Dataset | Best mIoU | Datasets |
|------|----------|----------|--------------|-----------|----------|
| 🥇 | gen_TSIT | 48.8 | MapillaryVistas | 52.3 | 4/4 |
| 🥈 | gen_albumentations_weather | 48.8 | MapillaryVistas | 52.1 | 4/4 |
| 🥉 | gen_UniControl | 48.5 | MapillaryVistas | 52.0 | 4/4 |
| 4. | gen_Qwen_Image_Edit | 48.2 | MapillaryVistas | 52.1 | 3/4 |
| 5. | gen_CNetSeg | 47.3 | BDD10k | 50.0 | 3/4 |
| 6. | gen_VisualCloze | 47.2 | MapillaryVistas | 51.7 | 4/4 |
| 7. | gen_SUSTechGAN | 47.1 | MapillaryVistas | 52.1 | 4/4 |
| 8. | gen_IP2P | 46.7 | MapillaryVistas | 51.9 | 4/4 |
| 9. | gen_Img2Img | 46.6 | MapillaryVistas | 52.1 | 4/4 |
| 10. | gen_LANIT | 46.2 | MapillaryVistas | 52.3 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 45.3 | 43.2 | ⏳ | 48.7 | 45.7 |
| gen_augmenters | 50.5 | 43.2 | 45.2 | 34.2 | 43.3 |
| gen_automold | 51.2 | 38.5 | 51.7 | 34.5 | 44.0 |
| gen_CNetSeg | 50.0 | 43.3 | ⏳ | 48.4 | 47.3 |
| gen_CUT | 44.5 | 25.5 | 51.8 | 48.3 | 42.5 |
| gen_cyclediffusion | 50.9 | 39.2 | ⏳ | ⏳ | 45.0 |
| gen_cycleGAN | 50.0 | 43.3 | 52.1 | 35.1 | 45.1 |
| gen_flux_kontext | ⏳ | ⏳ | 52.0 | 4.4 | 28.2 |
| gen_Img2Img | 42.9 | 43.2 | 52.1 | 48.4 | 46.6 |
| gen_IP2P | 43.3 | 43.4 | 51.9 | 48.2 | 46.7 |
| gen_LANIT | 40.4 | 43.3 | 52.3 | 48.6 | 46.2 |
| gen_Qwen_Image_Edit | ⏳ | 43.3 | 52.1 | 49.1 | 48.2 |
| gen_stargan_v2 | 50.7 | 43.1 | 45.6 | 29.1 | 42.1 |
| gen_step1x_new | ⏳ | 43.3 | ⏳ | 4.5 | 23.9 |
| gen_step1x_v1p2 | 51.1 | 43.2 | ⏳ | 4.6 | 33.0 |
| gen_SUSTechGAN | 45.2 | 43.0 | 52.1 | 48.3 | 47.1 |
| gen_TSIT | 51.2 | 43.3 | 52.3 | 48.5 | 48.8 |
| gen_UniControl | 50.3 | 43.2 | 52.0 | 48.3 | 48.5 |
| gen_VisualCloze | 50.4 | 38.0 | 51.7 | 48.6 | 47.2 |
| gen_Weather_Effect_Generator | 44.6 | 38.1 | ⏳ | 48.5 | 43.7 |
| gen_albumentations_weather | 51.3 | 43.0 | 52.1 | 48.7 | 48.8 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 43.2 | 36.6 | 51.0 | 48.8 | 44.9 |
| photometric_distort | 45.2 | 38.4 | ⏳ | 29.9 | 37.8 |
| std_autoaugment | 45.5 | ⏳ | ⏳ | 29.0 | 37.2 |
| std_cutmix | 45.3 | 43.1 | ⏳ | 3.8 | 30.8 |
| std_mixup | 44.6 | ⏳ | ⏳ | 3.9 | 24.2 |
| std_randaugment | 45.2 | ⏳ | ⏳ | 33.9 | 39.5 |

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
| gen_Attribute_Hallucination | ✅ | ✅ | ❌ | ✅ |
| gen_augmenters | ✅ | ✅ | ✅ | ✅ |
| gen_automold | ✅ | ✅ | ✅ | ✅ |
| gen_CNetSeg | ✅ | ✅ | 🔄 | ✅ |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |
| gen_cyclediffusion | ✅ | ✅ | ⏳ | ⏳ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ⏳ | ⏳ | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |
| gen_IP2P | ✅ | ✅ | ✅ | ✅ |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ⏳ | ✅ | ✅ | ✅ |
| gen_stargan_v2 | ✅ | ✅ | ✅ | ✅ |
| gen_step1x_new | ⏳ | ✅ | 🔄 | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ | ⏳ | ✅ |
| gen_SUSTechGAN | ✅ | ✅ | ✅ | ✅ |
| gen_TSIT | ✅ | ✅ | ✅ | ✅ |
| gen_UniControl | ✅ | ✅ | ✅ | ✅ |
| gen_VisualCloze | ✅ | ✅ | ✅ | ✅ |
| gen_Weather_Effect_Generator | ✅ | ✅ | ❌ | ✅ |
| gen_albumentations_weather | ✅ | ✅ | ✅ | ✅ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ✅ | ✅ | ✅ |
| photometric_distort | ✅ | ✅ | ⏳ | ✅ |
| std_autoaugment | ✅ | ⏳ | ⏳ | ✅ |
| std_cutmix | ✅ | ✅ | ⏳ | ✅ |
| std_mixup | ✅ | ⏳ | ⏳ | ✅ |
| std_randaugment | ✅ | ⏳ | ⏳ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 24 | 0 | 3 | 0 |
| IDD-AW | 23 | 0 | 4 | 0 |
| MapillaryVistas | 16 | 2 | 7 | 0 |
| OUTSIDE15k | 26 | 0 | 1 | 0 |

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

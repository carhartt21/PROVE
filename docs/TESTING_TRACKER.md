# Testing Progress Tracker

**Last Updated:** 2026-01-29 08:01


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
| 🥇 | gen_TSIT | 45.1 | BDD10k | 47.7 | 2/4 |
| 🥈 | gen_Attribute_Hallucination | 45.0 | BDD10k | 47.8 | 2/4 |
| 🥉 | gen_augmenters | 45.0 | BDD10k | 47.8 | 2/4 |
| 4. | gen_SUSTechGAN | 44.9 | BDD10k | 47.9 | 2/4 |
| 5. | gen_CUT | 44.9 | BDD10k | 47.9 | 2/4 |
| 6. | gen_Weather_Effect_Generator | 44.7 | BDD10k | 47.4 | 2/4 |
| 7. | gen_IP2P | 44.7 | BDD10k | 47.3 | 2/4 |
| 8. | gen_UniControl | 44.7 | BDD10k | 47.5 | 2/4 |
| 9. | gen_VisualCloze | 44.7 | BDD10k | 47.4 | 2/4 |
| 10. | gen_Img2Img | 44.6 | BDD10k | 47.2 | 2/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 47.8 | 42.2 | ⏳ | ⏳ | 45.0 |
| gen_augmenters | 47.8 | 42.1 | ⏳ | ⏳ | 45.0 |
| gen_automold | 47.6 | 42.1 | 37.9 | 46.1 | 43.4 |
| gen_CNetSeg | 47.1 | 42.0 | ⏳ | ⏳ | 44.6 |
| gen_CUT | 47.9 | 41.9 | ⏳ | ⏳ | 44.9 |
| gen_cyclediffusion | 46.9 | 42.1 | ⏳ | ⏳ | 44.5 |
| gen_cycleGAN | 47.5 | 42.1 | 38.1 | 46.2 | 43.5 |
| gen_flux_kontext | 47.7 | 42.1 | 37.8 | 46.5 | 43.5 |
| gen_Img2Img | 47.2 | 42.1 | ⏳ | ⏳ | 44.6 |
| gen_IP2P | 47.3 | 42.2 | ⏳ | ⏳ | 44.7 |
| gen_LANIT | 46.9 | 42.1 | 34.3 | 40.8 | 41.0 |
| gen_Qwen_Image_Edit | 47.0 | 42.0 | ⏳ | ⏳ | 44.5 |
| gen_stargan_v2 | 46.8 | 42.0 | ⏳ | ⏳ | 44.4 |
| gen_step1x_new | 47.2 | 42.1 | 37.9 | 46.3 | 43.4 |
| gen_step1x_v1p2 | 48.2 | 42.5 | 34.2 | 40.9 | 41.5 |
| gen_SUSTechGAN | 47.9 | 42.0 | ⏳ | ⏳ | 44.9 |
| gen_TSIT | 47.7 | 42.4 | ⏳ | ⏳ | 45.1 |
| gen_UniControl | 47.5 | 42.0 | ⏳ | ⏳ | 44.7 |
| gen_VisualCloze | 47.4 | 42.0 | ⏳ | ⏳ | 44.7 |
| gen_Weather_Effect_Generator | 47.4 | 42.1 | ⏳ | ⏳ | 44.7 |
| gen_albumentations_weather | 47.3 | 42.1 | 34.3 | 46.4 | 42.5 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 47.9 | 41.9 | 37.6 | 46.7 | 43.5 |
| std_minimal | 47.8 | 42.5 | 38.4 | 46.3 | 43.7 |
| std_photometric_distort | 48.1 | 42.3 | 37.7 | 46.3 | 43.6 |
| std_autoaugment | 48.0 | 42.0 | 37.7 | 46.3 | 43.5 |
| std_cutmix | 47.9 | 42.2 | 38.1 | 46.6 | 43.7 |
| std_mixup | 46.9 | 42.2 | 38.3 | 46.1 | 43.4 |
| std_randaugment | 47.1 | 41.7 | 37.6 | 46.1 | 43.1 |

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
| gen_Attribute_Hallucination | ✅ | ✅ | ⏳ | ⏳ |
| gen_augmenters | ✅ | ✅ | ⏳ | ⏳ |
| gen_automold | ✅ | ✅ | ✅ | ✅ |
| gen_CNetSeg | ✅ | ✅ | ⏳ | ⏳ |
| gen_CUT | ✅ | ✅ | ⏳ | ⏳ |
| gen_cyclediffusion | ✅ | ✅ | ⏳ | ⏳ |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |
| gen_flux_kontext | ✅ | ✅ | ✅ | ✅ |
| gen_Img2Img | ✅ | ✅ | ⏳ | ⏳ |
| gen_IP2P | ✅ | ✅ | ⏳ | ⏳ |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |
| gen_Qwen_Image_Edit | ✅ | ✅ | ⏳ | ⏳ |
| gen_stargan_v2 | ✅ | ✅ | ⏳ | ⏳ |
| gen_step1x_new | ✅ | ✅ | ✅ | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ | ✅ | ✅ |
| gen_SUSTechGAN | ✅ | ✅ | ⏳ | ⏳ |
| gen_TSIT | ✅ | ✅ | ⏳ | ⏳ |
| gen_UniControl | ✅ | ✅ | ⏳ | ⏳ |
| gen_VisualCloze | ✅ | ✅ | ⏳ | ⏳ |
| gen_Weather_Effect_Generator | ✅ | ✅ | ⏳ | ⏳ |
| gen_albumentations_weather | ✅ | ✅ | ✅ | ✅ |

### Standard Strategies Status

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|--------|--------|
| baseline | ✅ | ✅ | ✅ | ✅ |
| std_minimal | ✅ | ✅ | ✅ | ✅ |
| std_photometric_distort | ✅ | ✅ | ✅ | ✅ |
| std_autoaugment | ✅ | ✅ | ✅ | ✅ |
| std_cutmix | ✅ | ✅ | ✅ | ✅ |
| std_mixup | ✅ | ✅ | ✅ | ✅ |
| std_randaugment | ✅ | ✅ | ✅ | ✅ |

---


## Test Result Summary

| Dataset | Complete | Running | Pending | Skip |
|---------|----------|---------|---------|------|
| BDD10k | 28 | 0 | 0 | 0 |
| IDD-AW | 28 | 0 | 0 | 0 |
| MapillaryVistas | 14 | 0 | 14 | 0 |
| OUTSIDE15k | 14 | 0 | 14 | 0 |

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

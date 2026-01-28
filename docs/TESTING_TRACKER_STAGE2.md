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
| 🥇 | gen_VisualCloze | 47.9 | BDD10k | 52.3 | 4/4 |
| 🥈 | gen_cycleGAN | 47.9 | BDD10k | 52.2 | 4/4 |
| 🥉 | gen_step1x_v1p2 | 47.9 | BDD10k | 52.2 | 4/4 |
| 4. | gen_Attribute_Hallucination | 47.8 | BDD10k | 52.1 | 4/4 |
| 5. | gen_Qwen_Image_Edit | 47.8 | BDD10k | 52.1 | 4/4 |
| 6. | gen_TSIT | 47.8 | BDD10k | 52.2 | 4/4 |
| 7. | gen_step1x_new | 47.8 | BDD10k | 52.1 | 4/4 |
| 8. | gen_albumentations_weather | 47.8 | BDD10k | 52.1 | 4/4 |
| 9. | std_cutmix | 47.7 | BDD10k | 51.9 | 4/4 |
| 10. | gen_cyclediffusion | 47.7 | BDD10k | 51.8 | 4/4 |


### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| gen_Attribute_Hallucination | 52.1 | 47.0 | 41.2 | 51.0 | 47.8 |
| gen_augmenters | 51.8 | 46.8 | 41.3 | 50.4 | 47.6 |
| gen_automold | 52.0 | 46.8 | 41.0 | 50.6 | 47.6 |
| gen_CNetSeg | 51.8 | 46.8 | 41.2 | 50.8 | 47.6 |
| gen_CUT | 51.5 | 46.9 | 40.7 | 50.7 | 47.4 |
| gen_cyclediffusion | 51.8 | 46.9 | 41.5 | 50.7 | 47.7 |
| gen_cycleGAN | 52.2 | 47.0 | 41.5 | 50.9 | 47.9 |
| gen_flux_kontext | 51.2 | 46.9 | 41.4 | 50.9 | 47.6 |
| gen_Img2Img | 51.5 | 46.9 | 41.0 | 51.0 | 47.6 |
| gen_IP2P | 51.7 | 46.9 | 40.5 | 50.7 | 47.5 |
| gen_LANIT | 51.8 | 47.0 | 41.1 | 50.6 | 47.6 |
| gen_Qwen_Image_Edit | 52.1 | 46.9 | 41.4 | 50.9 | 47.8 |
| gen_stargan_v2 | 51.1 | 46.9 | 41.1 | 51.0 | 47.5 |
| gen_step1x_new | 52.1 | 47.0 | 41.5 | 50.7 | 47.8 |
| gen_step1x_v1p2 | 52.2 | 47.0 | 41.3 | 51.0 | 47.9 |
| gen_SUSTechGAN | 51.3 | 46.8 | 41.4 | 50.9 | 47.6 |
| gen_TSIT | 52.2 | 47.0 | 41.5 | 50.6 | 47.8 |
| gen_UniControl | 51.5 | 46.9 | 41.3 | 50.4 | 47.5 |
| gen_VisualCloze | 52.3 | 47.0 | 41.5 | 50.8 | 47.9 |
| gen_Weather_Effect_Generator | 51.9 | 46.8 | 41.3 | 50.7 | 47.7 |
| gen_albumentations_weather | 52.1 | 46.9 | 41.2 | 50.9 | 47.8 |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Avg |
|----------|-------:|-------:|-------:|-------:|-------:|
| baseline | 50.6 | 46.8 | 41.0 | 51.2 | 47.4 |
| std_photometric_distort | 52.3 | 46.9 | 41.0 | 50.6 | 47.7 |
| std_autoaugment | 52.0 | 46.9 | 41.0 | 50.8 | 47.7 |
| std_cutmix | 51.9 | 46.9 | 41.6 | 50.5 | 47.7 |
| std_mixup | 51.4 | 46.4 | 40.6 | 50.2 | 47.1 |
| std_randaugment | 51.3 | 46.8 | 40.4 | 51.1 | 47.4 |

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
| std_photometric_distort | ✅ | ✅ | ✅ | ✅ |
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
bjobs -u mima2416 | grep retest

# Count by status
bjobs -u mima2416 -o "JOB_NAME STAT" | grep retest | awk '{print $2}' | sort | uniq -c
```

### Submit Retest Jobs
```bash
cd scripts/retest_jobs_lsf
bash submit_all_retests.sh
```

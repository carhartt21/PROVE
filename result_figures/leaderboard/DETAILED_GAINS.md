# Detailed Per-Dataset and Per-Domain mIoU Gains

All gains are computed relative to **baseline_clear_day** (trained only on clear_day data).

**Important Notes:**
- **Avg Gain** = Average improvement across available datasets (from overall mIoU)
- **Normal Gain / Adverse Gain** = Improvement from per-domain detailed test results
- These metrics may come from different data subsets for strategies with incomplete coverage
- Strategies with '-' entries have missing data for those columns

---

## Per-Dataset mIoU Gains (Full-Trained Models)

Shows mIoU improvement over baseline_clear_day for each dataset. Positive = better than baseline.

**Columns:** Dataset gains from overall mIoU; Normal/Adverse Gain from per-domain metrics (may have different coverage).

| Strategy | Type | bdd10k | iddaw | mapillaryvistas | outside15k | Avg Gain | Normal Gain | Adverse Gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | Baseline Full | - | - | - | - | - | - | - |
| std_autoaugment | Standard Aug | - | - | - | - | - | - | - |
| std_cutmix | Standard Aug | - | - | - | - | - | - | - |
| std_minimal | Standard Aug | - | - | - | - | - | - | - |
| std_mixup | Standard Aug | - | - | - | - | - | - | - |
| std_photometric_distort | Standard Aug | - | - | - | - | - | - | - |
| std_randaugment | Standard Aug | - | - | - | - | - | - | - |

---

## Per-Domain mIoU Gains (Full-Trained Models)

Shows mIoU improvement over baseline_clear_day for each weather domain. Normal = clear_day + cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Gain | Adverse Gain | Overall Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | Baseline Full | - | - | - | - | - | - | - | - | - | - |
| std_autoaugment | Standard Aug | - | - | - | - | - | - | - | - | - | - |
| std_cutmix | Standard Aug | - | - | - | - | - | - | - | - | - | - |
| std_minimal | Standard Aug | - | - | - | - | - | - | - | - | - | - |
| std_mixup | Standard Aug | - | - | - | - | - | - | - | - | - | - |
| std_photometric_distort | Standard Aug | - | - | - | - | - | - | - | - | - | - |
| std_randaugment | Standard Aug | - | - | - | - | - | - | - | - | - | - |

---

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

| Strategy | Type | bdd10k | Avg Gain | Normal Gain | Adverse Gain |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | Baseline Full | - | - | - | - |

---

## Per-Dataset mIoU Gains (Clear-Day Trained Models)

Shows mIoU improvement over baseline_clear_day for models trained only on clear_day data.

| Strategy | Type | bdd10k_cd | idd-aw_cd | outside15k_cd | Avg Gain | Normal Gain | Adverse Gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | Baseline Full | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| std_cutmix | Standard Aug | +2.91 | +27.89 | -6.14 | +8.22 | +26.66 | +21.54 |
| std_mixup | Standard Aug | +2.03 | - | -5.91 | -1.94 | +20.09 | +19.25 |
| std_randaugment | Standard Aug | -40.96 | - | +22.51 | -9.22 | +11.34 | +10.12 |
| photometric_distort | Augmentation | -40.87 | - | +19.52 | -10.67 | -4.23 | -2.52 |
| std_autoaugment | Standard Aug | -40.97 | - | +19.35 | -10.81 | -4.70 | -2.94 |

---

## Per-Domain mIoU Gains (Clear-Day Trained Models)

Shows mIoU improvement over baseline_clear_day for models trained only on clear_day data.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Gain | Adverse Gain | Overall Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | Baseline Full | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| std_cutmix | Standard Aug | +27.52 | +25.79 | +21.88 | +25.61 | +15.51 | +21.68 | +23.33 | +26.66 | +21.54 | +23.05 |
| std_mixup | Standard Aug | +18.40 | +21.78 | +16.46 | +24.29 | +9.10 | +18.75 | +24.86 | +20.09 | +19.25 | +19.09 |
| std_randaugment | Standard Aug | +11.78 | +10.90 | +4.77 | +13.59 | +7.75 | +10.56 | +8.60 | +11.34 | +10.12 | +9.71 |
| photometric_distort | Augmentation | -4.96 | -3.49 | -2.94 | -2.83 | -1.85 | -2.57 | -2.84 | -4.23 | -2.52 | -3.07 |
| std_autoaugment | Standard Aug | -5.31 | -4.09 | -3.21 | -3.28 | -2.53 | -3.00 | -2.95 | -4.70 | -2.94 | -3.48 |
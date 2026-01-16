# Strategy Leaderboard (Full-Trained)

Generated from PROVE domain gap analysis pipeline.

**Reference Baseline: `baseline_clear_day`** (models trained only on clear_day data)

This is the proper baseline for measuring augmentation effectiveness, as it represents
models that never saw adverse weather conditions during training.

**Metrics:**
- **Overall mIoU**: Mean Intersection over Union across all domains
- **Gain vs Clear Day**: Overall mIoU improvement vs baseline_clear_day (positive = better)
- **Normal mIoU**: Performance on clear_day + cloudy conditions
- **Normal Gain**: Normal mIoU improvement vs baseline_clear_day (positive = better)
- **Adverse mIoU**: Performance on foggy, rainy, snowy, night conditions
- **Adverse Gain**: Adverse mIoU improvement vs baseline_clear_day (positive = better)
- **Domain Gap (Δ)**: Normal - Adverse (positive = worse on adverse)
- **Gap Reduction vs Clear Day**: Domain gap improvement vs baseline_clear_day (positive = smaller gap)

**Baseline Types:**
- `baseline_clear_day`: Trained only on clear_day data (THE REFERENCE)
- `baseline` / `baseline_full`: Trained on all weather conditions

---

| Strategy | Type | Overall mIoU | Gain vs Clear Day | Normal mIoU | Normal Gain | Adverse mIoU | Adverse Gain | Domain Gap (Δ) | Gap Reduction vs Clear Day |
|---|---|---|---|---|---|---|---|---|---|
| baseline_clear_day | Baseline Clear Day | 18.6% | 0.0% | 5.2% | 0.0% | 3.4% | 0.0% | +1.8% | 0.0% |
| baseline | Baseline Full | 44.5% | +25.9% | - | - | - | - | - | - |



# Strategy Leaderboard (Clear-Day Trained)

Generated from PROVE domain gap analysis pipeline.

**Reference Baseline: `baseline_clear_day`** (models trained only on clear_day data)

This is the proper baseline for measuring augmentation effectiveness, as it represents
models that never saw adverse weather conditions during training.

**Metrics:**
- **Overall mIoU**: Mean Intersection over Union across all domains
- **Gain vs Clear Day**: Overall mIoU improvement vs baseline_clear_day (positive = better)
- **Normal mIoU**: Performance on clear_day + cloudy conditions
- **Normal Gain**: Normal mIoU improvement vs baseline_clear_day (positive = better)
- **Adverse mIoU**: Performance on foggy, rainy, snowy, night conditions
- **Adverse Gain**: Adverse mIoU improvement vs baseline_clear_day (positive = better)
- **Domain Gap (Δ)**: Normal - Adverse (positive = worse on adverse)
- **Gap Reduction vs Clear Day**: Domain gap improvement vs baseline_clear_day (positive = smaller gap)

**Baseline Types:**
- `baseline_clear_day`: Trained only on clear_day data (THE REFERENCE)
- `baseline` / `baseline_full`: Trained on all weather conditions

---

| Strategy | Type | Overall mIoU | Gain vs Clear Day | Normal mIoU | Normal Gain | Adverse mIoU | Adverse Gain | Domain Gap (Δ) | Gap Reduction vs Clear Day |
|---|---|---|---|---|---|---|---|---|---|
| baseline_clear_day | Baseline Clear Day | 18.6% | 0.0% | 5.2% | 0.0% | 3.4% | 0.0% | +1.8% | 0.0% |
| std_cutmix | Standard Aug | 30.9% | +12.3% | 31.9% | +26.7% | 24.9% | +21.5% | +7.0% | -5.1% |
| std_mixup | Standard Aug | 23.6% | +5.0% | 25.3% | +20.1% | 22.6% | +19.2% | +2.7% | -0.8% |
| std_randaugment | Standard Aug | 16.4% | -2.3% | 16.6% | +11.3% | 13.5% | +10.1% | +3.1% | -1.2% |
| photometric_distort | Augmentation | 14.9% | -3.7% | 1.0% | -4.2% | 0.8% | -2.5% | +0.1% | +1.7% |
| std_autoaugment | Standard Aug | 10.1% | -8.6% | 0.5% | -4.7% | 0.4% | -2.9% | +0.1% | +1.8% |
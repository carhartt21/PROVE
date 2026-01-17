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
| baseline_clear_day | Baseline Clear Day | - | - | - | - | - | - | - | - |
| baseline | Baseline Full | 41.6% | - | 42.2% | - | 36.1% | - | +6.1% | - |
| std_autoaugment | Standard Aug | 45.5% | - | 46.1% | - | 40.4% | - | +5.8% | - |
| std_randaugment | Standard Aug | 44.1% | - | 44.6% | - | 38.7% | - | +5.9% | - |
| photometric_distort | Augmentation | 43.7% | - | 44.3% | - | 38.5% | - | +5.7% | - |
| std_mixup | Standard Aug | 42.9% | - | 43.5% | - | 37.4% | - | +6.2% | - |
| std_cutmix | Standard Aug | 42.6% | - | 43.3% | - | 37.1% | - | +6.2% | - |



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
| baseline_clear_day | Baseline Clear Day | - | - | - | - | - | - | - | - |
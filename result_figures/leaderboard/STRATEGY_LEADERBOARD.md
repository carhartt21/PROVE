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
| baseline | Baseline Full | 40.9% | - | 41.8% | - | 35.2% | - | +6.5% | - |
| std_minimal | Standard Aug | 41.3% | - | 42.2% | - | 35.3% | - | +6.8% | - |
| std_cutmix | Standard Aug | 40.9% | - | 41.8% | - | 35.0% | - | +6.8% | - |
| std_photometric_distort | Standard Aug | 40.9% | - | 41.8% | - | 35.5% | - | +6.4% | - |
| std_mixup | Standard Aug | 40.9% | - | 41.7% | - | 35.2% | - | +6.5% | - |
| std_autoaugment | Standard Aug | 40.8% | - | 41.7% | - | 34.8% | - | +6.9% | - |
| std_randaugment | Standard Aug | 40.7% | - | 41.5% | - | 34.9% | - | +6.5% | - |



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
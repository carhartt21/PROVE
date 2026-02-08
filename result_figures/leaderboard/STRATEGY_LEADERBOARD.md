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
| baseline | Baseline Full | 32.1% | - | 32.9% | - | 26.9% | - | +6.0% | - |
| std_photometric_distort | Standard Aug | 45.5% | - | 49.3% | - | 44.8% | - | +4.5% | - |
| std_minimal | Standard Aug | 44.9% | - | 49.2% | - | 42.7% | - | +6.5% | - |
| std_autoaugment | Standard Aug | 37.0% | - | 38.5% | - | 32.2% | - | +6.2% | - |
| std_cutmix | Standard Aug | 36.9% | - | 38.3% | - | 32.0% | - | +6.3% | - |
| std_randaugment | Standard Aug | 36.9% | - | 38.2% | - | 31.8% | - | +6.4% | - |
| std_mixup | Standard Aug | 36.8% | - | 38.4% | - | 31.9% | - | +6.4% | - |



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
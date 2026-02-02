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
| baseline | Baseline Full | 30.0% | - | 32.5% | - | 28.4% | - | +4.0% | - |
| std_autoaugment | Standard Aug | 45.5% | - | 49.4% | - | 44.3% | - | +5.1% | - |
| std_photometric_distort | Standard Aug | 45.5% | - | 49.3% | - | 44.8% | - | +4.5% | - |
| std_cutmix | Standard Aug | 45.3% | - | 48.9% | - | 43.5% | - | +5.3% | - |
| std_randaugment | Standard Aug | 45.2% | - | 49.3% | - | 43.5% | - | +5.8% | - |
| std_minimal | Standard Aug | 44.9% | - | 49.2% | - | 42.7% | - | +6.5% | - |
| std_mixup | Standard Aug | 44.8% | - | 49.5% | - | 42.7% | - | +6.8% | - |



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
# Gen+Std Combined Strategy Batch Submission

This document describes the `submit_gen_std_combinations.sh` script for submitting batch training jobs that combine generative augmentation strategies (gen_*) with standard augmentation strategies (std_*) to the LSF cluster.

## Overview

Based on comprehensive analysis using `test_result_analyzer.py`, the script identifies and submits training jobs for the most promising strategy combinations. The hypothesis is that combining the best-performing generative augmentation methods with standard augmentation techniques may yield further improvements.

## Top-Performing Strategies

### Generative Strategies (ranked by avg mIoU improvement over baseline)

| Rank | Strategy | Avg mIoU | Improvement vs Baseline |
|------|----------|----------|------------------------|
| 1 | gen_StyleID | 56.27% | +4.14% |
| 2 | gen_cycleGAN | 55.96% | +3.83% |
| 3 | gen_LANIT | 55.71% | +3.58% |
| 4 | gen_CUT | 55.70% | +3.56% |
| 5 | gen_step1x_new | 55.70% | +3.56% |
| 6 | gen_automold | 55.62% | +3.48% |

### Standard Augmentation Strategies (ranked by avg mIoU)

| Rank | Strategy | Avg mIoU |
|------|----------|----------|
| 1 | std_randaugment | 56.44% |
| 2 | std_mixup | 56.19% |
| 3 | std_cutmix | 55.93% |
| 4 | std_autoaugment | 55.36% |

## Usage

```bash
./scripts/submit_gen_std_combinations.sh <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `submit-all` | Submit all gen+std combinations for all datasets/models |
| `submit-top` | Submit only top 3 gen + top 2 std combinations (reduced set) |
| `submit-single` | Submit a single combination (requires `--gen`, `--std`, `--dataset`, `--model`) |
| `submit-dataset` | Submit all combinations for one specific dataset |
| `submit-model` | Submit all combinations for one specific model |
| `list` | List all combinations that would be submitted |
| `estimate` | Estimate total jobs and resource requirements |
| `help` | Show help message |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gen <strategy>` | Specific gen_* strategy | - |
| `--std <strategy>` | Specific std_* strategy | - |
| `--dataset <name>` | Specific dataset | - |
| `--model <name>` | Specific model | - |
| `--queue <name>` | LSF queue name | BatchGPU |
| `--gpu-mem <size>` | GPU memory requirement | 24G |
| `--gpu-mode <mode>` | GPU mode (shared/exclusive_process) | shared |
| `--dry-run` | Show commands without executing | - |
| `--delay <seconds>` | Delay between job submissions | 1 |
| `--domain-filter <domain>` | Add domain filter (e.g., clear_day) | - |
| `--with-domain-variants` | Submit both regular AND clear_day variants | - |
| `--with-baseline-std` | Also submit baseline+std combinations | - |

## Examples

### Preview All Jobs (Dry Run)

```bash
./scripts/submit_gen_std_combinations.sh submit-all --dry-run
```

### Submit Top Combinations

Submit the top 3 gen strategies × top 2 std strategies = 6 combinations, across all datasets and models:

```bash
# Without domain variants (90 jobs)
./scripts/submit_gen_std_combinations.sh submit-top

# With domain variants (180 jobs - both regular and clear_day)
./scripts/submit_gen_std_combinations.sh submit-top --with-domain-variants
```

### Submit All Combinations

Submit all 6 gen × 4 std = 24 combinations:

```bash
# Without domain variants (360 jobs)
./scripts/submit_gen_std_combinations.sh submit-all

# With domain variants (720 jobs)
./scripts/submit_gen_std_combinations.sh submit-all --with-domain-variants
```

### Submit for Specific Dataset

```bash
./scripts/submit_gen_std_combinations.sh submit-dataset --dataset ACDC --with-domain-variants
```

### Submit for Specific Model

```bash
./scripts/submit_gen_std_combinations.sh submit-model --model segformer_mit-b5 --with-domain-variants
```

### Submit Single Combination

```bash
./scripts/submit_gen_std_combinations.sh submit-single \
    --gen gen_cycleGAN \
    --std std_cutmix \
    --dataset ACDC \
    --model deeplabv3plus_r50
```

### Check Resource Estimates

```bash
./scripts/submit_gen_std_combinations.sh estimate
```

Output:
```
==============================================
Resource Estimation
==============================================

Configuration:
  Gen strategies: 6 (gen_StyleID gen_cycleGAN gen_LANIT gen_CUT gen_step1x_new gen_automold)
  Std strategies: 4 (std_randaugment std_mixup std_cutmix std_autoaugment)
  Datasets: 5 (ACDC BDD10k IDD-AW MapillaryVistas OUTSIDE15k)
  Models: 3 (deeplabv3plus_r50 pspnet_r50 segformer_mit-b5)

Calculations:
  Total strategy combinations: 24
  Total training jobs (no domain variants): 360
  Total training jobs (with domain variants): 720

Estimated Resources (assuming ~4-8 hours per job):
  GPU hours (no variants): 2160 hours (avg 6h per job)
  GPU hours (with variants): 4320 hours
...
```

## Job Counts Summary

| Command | Without `--with-domain-variants` | With `--with-domain-variants` |
|---------|----------------------------------|------------------------------|
| `submit-all` | 360 | 720 |
| `submit-top` | 90 | 180 |
| `submit-dataset` | 72 | 144 |
| `submit-model` | 120 | 240 |

## Configuration

The script uses the following default configurations:

### Datasets
- ACDC
- BDD10k
- IDD-AW
- MapillaryVistas
- OUTSIDE15k

### Models
- deeplabv3plus_r50
- pspnet_r50
- segformer_mit-b5

### LSF Settings
- Queue: BatchGPU
- GPU Memory: 24G
- GPU Mode: shared
- CPUs per job: 8

These can be customized via command-line options or by editing the script's configuration section.

## Output Directories

Jobs will create output in:
- Logs: `logs/prove_<dataset>_<model>_<gen_strategy>+<std_strategy>_<jobid>.out/err`
- Weights: `${AWARE_DATA_ROOT}/WEIGHTS/<gen_strategy>+<std_strategy>/`

## Domain Variants

When using `--with-domain-variants`, the script submits two versions of each job:

1. **Regular**: Training on all available data
2. **clear_day**: Training only on clear_day domain-filtered data

This allows comparing performance when training on weather-diverse data vs. clear conditions only.

## Related Documentation

- [Unified Training](UNIFIED_TRAINING.md) - Main training documentation
- [Unified Testing](UNIFIED_TESTING.md) - Testing trained models
- [Result Visualization](RESULT_VISUALIZATION.md) - Visualizing results

## Updating Top Strategies

To update the list of top strategies based on new results:

1. Run `python test_result_analyzer.py --comprehensive`
2. Identify new top performers from the output
3. Edit `TOP_GEN_STRATEGIES` and `STD_STRATEGIES` arrays in the script

# Ratio Ablation Study

This document describes the `submit_ratio_ablation.sh` script for running ablation studies on the ratio of generated to real images during training.

## Overview

The ratio ablation study investigates how the proportion of generated images in the training data affects model performance. By varying the `real_gen_ratio` parameter from 0.125 to 1.0, we can understand the optimal balance between real and generated training data.

## Top 5 Generative Strategies

The ablation uses the 5 best performing `gen_*` strategies based on average mIoU across all datasets (with complete 4/4 dataset results):

| Rank | Strategy | Avg mIoU |
|------|----------|----------|
| 1 | gen_TSIT | 48.8 |
| 2 | gen_albumentations_weather | 48.8 |
| 3 | gen_cycleGAN | 48.5 |
| 4 | gen_UniControl | 48.5 |
| 5 | gen_automold | 47.5 |

*Note: Updated 2026-01-14 based on TESTING_TRACKER.md results.*

## Ratio Values

The following ratios are tested (0.125 increments):

| Ratio | Real Images | Generated Images | Notes |
|-------|-------------|------------------|-------|
| 1.0 | 100% | 0% | Pure real images (baseline) |
| 0.875 | 87.5% | 12.5% | |
| 0.75 | 75% | 25% | |
| 0.625 | 62.5% | 37.5% | |
| 0.5 | 50% | 50% | Standard training (in WEIGHTS) |
| 0.375 | 37.5% | 62.5% | |
| 0.25 | 25% | 75% | |
| 0.125 | 12.5% | 87.5% | |
| 0.0 | 0% | 100% | Pure synthetic images |

**Note:** Ratio 0.5 is excluded from ablation since it's equivalent to standard `gen_*` strategy training already present in the `WEIGHTS` folder. Ratio 0.0 (100% synthetic) tests training with only generated images.

## Usage

### Basic Commands

```bash
# List all jobs that would be submitted
./scripts/submit_ratio_ablation.sh --list

# Preview commands without executing (dry run)
./scripts/submit_ratio_ablation.sh --dry-run

# Submit all jobs
./scripts/submit_ratio_ablation.sh

# Submit with a limit
./scripts/submit_ratio_ablation.sh --limit 50
```

### Filtering Options

```bash
# Filter by dataset
./scripts/submit_ratio_ablation.sh --dataset ACDC

# Filter by model
./scripts/submit_ratio_ablation.sh --model deeplabv3plus_r50

# Filter by strategy
./scripts/submit_ratio_ablation.sh --strategy gen_LANIT

# Filter by ratio
./scripts/submit_ratio_ablation.sh --ratio 0.5

# Combine filters
./scripts/submit_ratio_ablation.sh --dataset ACDC --model deeplabv3plus_r50 --strategy gen_LANIT
```

### LSF Options

```bash
# Custom queue
./scripts/submit_ratio_ablation.sh --queue BatchGPU

# Custom GPU memory
./scripts/submit_ratio_ablation.sh --gpu-mem 32G

# Custom GPU mode
./scripts/submit_ratio_ablation.sh --gpu-mode exclusive_process

# Custom number of CPUs
./scripts/submit_ratio_ablation.sh --num-cpus 4
```

### Custom Output Location

```bash
# Specify custom weights output directory
./scripts/submit_ratio_ablation.sh --weights-root /path/to/custom/output
```

## Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Show commands without executing | - |
| `--list` | List all jobs that would be submitted | - |
| `--dataset <name>` | Filter to specific dataset | All |
| `--model <name>` | Filter to specific model | All |
| `--strategy <name>` | Filter to specific strategy | All top 5 |
| `--ratio <value>` | Filter to specific ratio | All |
| `--queue <name>` | LSF queue name | BatchGPU |
| `--gpu-mem <size>` | GPU memory requirement | 24G |
| `--gpu-mode <mode>` | GPU mode (shared/exclusive_process) | shared |
| `--num-cpus <n>` | Number of CPUs per job | 8 |
| `--limit <n>` | Limit number of jobs to submit | - |
| `--weights-root <path>` | Custom weights output directory | /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION |

## Output Structure

Weights are saved to:
```
/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/
├── gen_LANIT/
│   ├── acdc/
│   │   ├── deeplabv3plus_r50_ratio0p12/
│   │   ├── deeplabv3plus_r50_ratio0p25/
│   │   ├── deeplabv3plus_r50_ratio0p50/
│   │   └── ...
│   ├── bdd10k/
│   └── ...
├── gen_step1x_new/
└── ...
```

The ratio is encoded in the model directory name as `_ratio{X}p{YY}` (e.g., `_ratio0p50` for 0.5).

## Total Jobs

- **5 strategies** × **8 ratios** × **5 datasets** × **3 models** = **600 jobs**

## Monitoring

```bash
# Check job status
bjobs -w

# View logs
ls -la logs/ratio_*.log
ls -la logs/ratio_*.err

# Tail a specific log
tail -f logs/ratio_ACDC_deeplabv3plus_r50_gen_LANIT_r50_*.log
```

## Analysis

After training completes, use `test_result_analyzer.py` to analyze results:

```bash
# Analyze ratio ablation results
python test_result_analyzer.py --root /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION --comprehensive
```

## See Also

- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - General training documentation
- [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) - Extended training ablation study
- [GEN_STD_BATCH_SUBMISSION.md](GEN_STD_BATCH_SUBMISSION.md) - Batch submission for gen+std combinations

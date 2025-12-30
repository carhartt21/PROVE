# Extended Training Ablation Study

This document describes the `submit_extended_training.sh` script for running ablation studies on extended training duration beyond the standard 80,000 iterations.

## Overview

The extended training ablation study investigates whether training for longer improves model performance. The script resumes training from the latest checkpoint and continues to a specified maximum iteration count with early stopping disabled.

## Top 15 Strategies

The ablation uses the 15 best performing strategies based on average mIoU across all datasets and models:

| Rank | Strategy | Avg mIoU |
|------|----------|----------|
| 1 | std_randaugment+std_mixup | 56.06 |
| 2 | gen_LANIT | 55.71 |
| 3 | gen_step1x_new | 55.70 |
| 4 | std_mixup+std_autoaugment | 55.67 |
| 5 | gen_automold | 55.62 |
| 6 | gen_TSIT | 55.61 |
| 7 | std_randaugment | 55.58 |
| 8 | gen_NST | 55.55 |
| 9 | gen_CUT | 55.52 |
| 10 | gen_Attribute_Hallucination | 55.46 |
| 11 | gen_UniControl | 55.46 |
| 12 | std_cutmix+std_autoaugment | 55.45 |
| 13 | gen_Img2Img | 55.43 |
| 14 | gen_flux1_kontext | 55.40 |
| 15 | gen_SUSTechGAN | 55.37 |

## Iteration Recommendations

| Target | Multiplier | Description |
|--------|------------|-------------|
| 160,000 | 2× | Moderate extension (default) |
| 240,000 | 3× | Significant extension |
| 320,000 | 4× | Recommended maximum |
| 400,000 | 5× | Very long, may see diminishing returns |

## Key Features

- **Resumes from latest checkpoint**: No wasted computation
- **Early stopping disabled**: Trains for full specified duration
- **Separate output directory**: Results saved to `WEIGHTS_EXTENDED` by default
- **Automatic checkpoint detection**: Finds the highest iteration checkpoint available

## Usage

### Basic Commands

```bash
# List all jobs that would be submitted
./submit_extended_training.sh --list

# Preview commands without executing (dry run)
./submit_extended_training.sh --dry-run

# Submit all jobs (default: 160,000 iterations)
./submit_extended_training.sh

# Submit with a limit
./submit_extended_training.sh --limit 50
```

### Specifying Iteration Target

```bash
# Train to 160,000 iterations (2× default)
./submit_extended_training.sh --max-iters 160000

# Train to 240,000 iterations (3×)
./submit_extended_training.sh --max-iters 240000

# Train to 320,000 iterations (4×)
./submit_extended_training.sh --max-iters 320000

# Train to 400,000 iterations (5×)
./submit_extended_training.sh --max-iters 400000
```

### Filtering Options

```bash
# Filter by dataset
./submit_extended_training.sh --dataset ACDC

# Filter by model
./submit_extended_training.sh --model deeplabv3plus_r50

# Filter by strategy (use quotes for combined strategies)
./submit_extended_training.sh --strategy gen_LANIT
./submit_extended_training.sh --strategy "std_randaugment+std_mixup"

# Combine filters
./submit_extended_training.sh --dataset ACDC --model deeplabv3plus_r50 --max-iters 240000
```

### LSF Options

```bash
# Custom queue
./submit_extended_training.sh --queue BatchGPU

# Custom GPU memory
./submit_extended_training.sh --gpu-mem 32G

# Custom GPU mode
./submit_extended_training.sh --gpu-mode exclusive_process

# Custom number of CPUs
./submit_extended_training.sh --num-cpus 4
```

### Custom Paths

```bash
# Specify source weights directory (where existing checkpoints are)
./submit_extended_training.sh --weights-root /path/to/existing/weights

# Specify output directory for extended training results
./submit_extended_training.sh --output-root /path/to/output
```

## Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Show commands without executing | - |
| `--list` | List all jobs that would be submitted | - |
| `--dataset <name>` | Filter to specific dataset | All |
| `--model <name>` | Filter to specific model | All |
| `--strategy <name>` | Filter to specific strategy | All top 15 |
| `--max-iters <n>` | Maximum training iterations | 160000 |
| `--queue <name>` | LSF queue name | BatchGPU |
| `--gpu-mem <size>` | GPU memory requirement | 24G |
| `--gpu-mode <mode>` | GPU mode (shared/exclusive_process) | shared |
| `--num-cpus <n>` | Number of CPUs per job | 8 |
| `--limit <n>` | Limit number of jobs to submit | - |
| `--weights-root <path>` | Source weights directory | /scratch/aaa_exchange/AWARE/WEIGHTS |
| `--output-root <path>` | Output weights directory | /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED |

## How It Works

1. **Checkpoint Discovery**: For each strategy/dataset/model combination, the script finds the latest `iter_*.pth` checkpoint file.

2. **Progress Check**: If the checkpoint iteration is already at or beyond `--max-iters`, the job is skipped.

3. **Resume Training**: Training resumes from the checkpoint with:
   - `--resume-from <checkpoint_path>`
   - `--max-iters <target>`
   - `--no-early-stop` (ensures full training duration)

4. **Output**: Results are saved to the output directory, preserving the same structure as the source.

## Output Structure

Extended weights are saved to:
```
/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/
├── gen_LANIT/
│   ├── acdc/
│   │   ├── deeplabv3plus_r50/
│   │   │   ├── iter_80000.pth      # Copied from source
│   │   │   ├── iter_90000.pth      # New checkpoint
│   │   │   ├── iter_100000.pth
│   │   │   └── iter_160000.pth     # Final checkpoint
│   │   └── ...
│   └── ...
├── std_randaugment+std_mixup/
└── ...
```

## Total Jobs

- **15 strategies** × **5 datasets** × **3 models** = **225 jobs**

## Monitoring

```bash
# Check job status
bjobs -w

# View logs
ls -la logs/ext_*.log
ls -la logs/ext_*.err

# Tail a specific log
tail -f logs/ext_ACDC_deeplabv3plus_r50_gen_LANIT_160k_*.log
```

## Analysis

After training completes, use `test_result_analyzer.py` to analyze results:

```bash
# Analyze extended training results
python test_result_analyzer.py --root /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED --comprehensive

# Compare with baseline results
python test_result_analyzer.py --root /scratch/aaa_exchange/AWARE/WEIGHTS --comprehensive
```

## Troubleshooting

### "No checkpoint found" for a configuration

The script requires existing checkpoints to resume from. Ensure the source weights directory contains trained models:

```bash
# Check if checkpoints exist
ls /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/acdc/deeplabv3plus_r50/iter_*.pth
```

### Job skipped (already at max_iters)

If a model has already been trained to or beyond the target iteration, it will be skipped. Use a higher `--max-iters` value or check the current checkpoint:

```bash
# Check current checkpoint
ls -la /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/acdc/deeplabv3plus_r50/iter_*.pth
```

## See Also

- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - General training documentation
- [RATIO_ABLATION.md](RATIO_ABLATION.md) - Ratio ablation study
- [GEN_STD_BATCH_SUBMISSION.md](GEN_STD_BATCH_SUBMISSION.md) - Batch submission for gen+std combinations

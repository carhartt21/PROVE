# Extended Training Ablation Study

This document describes the extended training ablation study for investigating performance gains from training beyond the standard 80,000 iterations.

> **Note:** The original `submit_extended_training.sh` shell script has been superseded by `batch_training_submission.py --stage extended` or direct `unified_training.py --resume` calls. The concepts below remain valid.

## Overview

The extended training ablation study investigates whether training for longer improves model performance. The script resumes training from the latest checkpoint and continues to a specified maximum iteration count with early stopping disabled.

## Top 15 Strategies

The ablation uses the 5 best performing strategies based on average mIoU across all datasets and models

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
./scripts/submit_extended_training.sh --list

# Preview commands without executing (dry run)
./scripts/submit_extended_training.sh --dry-run

# Submit all jobs (default: 160,000 iterations)
./scripts/submit_extended_training.sh

# Submit with a limit
./scripts/submit_extended_training.sh --limit 50
```

### Specifying Iteration Target

```bash
# Train to 160,000 iterations (2× default)
./scripts/submit_extended_training.sh --max-iters 160000

# Train to 240,000 iterations (3×)
./scripts/submit_extended_training.sh --max-iters 240000

# Train to 320,000 iterations (4×)
./scripts/submit_extended_training.sh --max-iters 320000

# Train to 400,000 iterations (5×)
./scripts/submit_extended_training.sh --max-iters 400000
```

### Filtering Options

```bash
# Filter by dataset
./scripts/submit_extended_training.sh --dataset ACDC

# Filter by model
./scripts/submit_extended_training.sh --model deeplabv3plus_r50

# Filter by strategy (use quotes for combined strategies)
./scripts/submit_extended_training.sh --strategy gen_LANIT
./scripts/submit_extended_training.sh --strategy "std_randaugment+std_mixup"

# Combine filters
./scripts/submit_extended_training.sh --dataset ACDC --model deeplabv3plus_r50 --max-iters 240000
```

### LSF Options

```bash
# Custom queue
./scripts/submit_extended_training.sh --queue BatchGPU

# Custom GPU memory
./scripts/submit_extended_training.sh --gpu-mem 32G

# Custom GPU mode
./scripts/submit_extended_training.sh --gpu-mode exclusive_process

# Custom number of CPUs
./scripts/submit_extended_training.sh --num-cpus 4
```

### Custom Paths

```bash
# Specify source weights directory (where existing checkpoints are)
./scripts/submit_extended_training.sh --weights-root /path/to/existing/weights

# Specify output directory for extended training results
./scripts/submit_extended_training.sh --output-root /path/to/output
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
| `--weights-root <path>` | Source weights directory | ${AWARE_DATA_ROOT}/WEIGHTS |
| `--output-root <path>` | Output weights directory | ${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED |

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
${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/
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

After training completes, use the analysis scripts to evaluate results:

### Testing Extended Training Checkpoints

Use `submit_test_extended_training.sh` to submit evaluation jobs for all checkpoints:

```bash
# List all available test jobs
./scripts/submit_test_extended_training.sh --list

# Dry run to see commands
./scripts/submit_test_extended_training.sh --dry-run

# Submit all test jobs
./scripts/submit_test_extended_training.sh

# Submit tests for specific strategy
./scripts/submit_test_extended_training.sh --strategy gen_cyclediffusion

# Submit tests for specific iteration only
./scripts/submit_test_extended_training.sh --iteration 160000

# Skip already tested configurations
./scripts/submit_test_extended_training.sh --skip-tested
```

The testing script uses `fine_grained_test.py` which provides:
- Per-domain (weather condition) metrics
- Per-class IoU breakdown
- Timestamped output folders
- Results saved to `test_results_iter_{N}/` directories

### Analyzing Results

```bash
# Analyze extended training results (scans for results automatically)
python analysis_scripts/analyze_extended_training.py --verbose

# Export to CSV and JSON
python analysis_scripts/analyze_extended_training.py --auto-export

# Filter by strategy/dataset/model
python analysis_scripts/analyze_extended_training.py --strategy gen_cyclediffusion --dataset BDD10k
```

### Generating Publication Figures

```bash
# Generate all IEEE-style figures
python analysis_scripts/generate_ieee_figures_extended_training.py

# Output structure:
# result_figures/extended_training/
# ├── ieee/           # PDF figures for publication
# │   ├── fig_learning_curves.pdf
# │   ├── fig_convergence_heatmap.pdf
# │   ├── fig_improvement_distribution.pdf
# │   ├── fig_best_iteration.pdf
# │   ├── fig_strategy_comparison.pdf
# │   ├── fig_model_dataset_comparison.pdf
# │   └── fig_diminishing_returns.pdf
# ├── preview/        # PNG previews
# └── data/           # CSV and JSON data
#     ├── extended_training_results.csv
#     └── extended_training_summary.json
```

### Visualizing Results

```bash
# Generate all visualizations
python analysis_scripts/visualize_extended_training.py

# Generate specific plots
python analysis_scripts/visualize_extended_training.py --plots learning convergence improvement
```

## Troubleshooting

### "No checkpoint found" for a configuration

The script requires existing checkpoints to resume from. Ensure the source weights directory contains trained models:

```bash
# Check if checkpoints exist
ls ${AWARE_DATA_ROOT}/WEIGHTS/gen_LANIT/acdc/deeplabv3plus_r50/iter_*.pth
```

### Job skipped (already at max_iters)

If a model has already been trained to or beyond the target iteration, it will be skipped. Use a higher `--max-iters` value or check the current checkpoint:

```bash
# Check current checkpoint
ls -la ${AWARE_DATA_ROOT}/WEIGHTS/gen_LANIT/acdc/deeplabv3plus_r50/iter_*.pth
```

## See Also

- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - General training documentation
- [RATIO_ABLATION.md](RATIO_ABLATION.md) - Ratio ablation study
- [GEN_STD_BATCH_SUBMISSION.md](GEN_STD_BATCH_SUBMISSION.md) - Batch submission for gen+std combinations

## Technical Notes

### Resumption Logic Fix (MMEngine 1.x)

A critical fix was applied to `unified_training.py` to ensure that training resumes correctly from the specified iteration count. 

**Issue**: Using the deprecated `resume_from` parameter in MMEngine 1.x caused the iteration counter to reset to 0, even if weights were loaded. This resulted in training starting from "Iter 0" instead of "Iter 80,000".

**Fix**: The pipeline now uses the modern MMEngine resumption pattern:
```python
config['load_from'] = checkpoint_path
config['resume'] = True
```
This ensures that the iteration counter, optimizer state, and scheduler state are all correctly restored.

### Automated Sequential Testing

To evaluate the performance across the entire training trajectory, a new script `scripts/submit_all_tests.sh` was introduced.

**Features**:
- Finds all `iter_*.pth` checkpoints for a given strategy.
- Groups tests by model and runs them **sequentially** in a single LSF job.
- Significantly reduces queue load (from thousands of jobs to ~175).
- Saves results to `test_results/` within each model directory.

**Usage**:
```bash
# Submit sequential tests for a strategy
./scripts/submit_all_tests.sh gen_LANIT
```

## Results Analysis (320k Iterations)

A comprehensive analysis of 22 configurations trained to 320k iterations reveals **clear diminishing returns**:

### Marginal Gains by Training Phase

| Training Phase | Average Gain | % of Initial |
|----------------|-------------|--------------|
| 90k → 160k | **+0.75 mIoU** | 100% (baseline) |
| 160k → 240k | **+0.39 mIoU** | 52% |
| 240k → 320k | **+0.10 mIoU** | 13% |

### Key Findings

1. **First 80k of extended training** (90k→160k) captures ~60% of total improvement
2. **Training beyond 240k** provides marginal gains (~0.1 mIoU average)
3. **Peak performance** typically occurs around 190k-450k depending on configuration
4. **gen_TSIT** saturates fastest (peak at 190k, zero gains after 160k)
5. **IDD-AW configurations** benefit most from extended training

### Recommendations

| Use Case | Duration | Rationale |
|----------|----------|-----------|
| **Production** | 160k (2×) | 75% of gains at 50% compute |
| **Research** | 240k (3×) | 92% of gains, good cost-benefit |
| **Benchmarks** | 320k (4×) | Maximum performance |

### Visualizations

See the full analysis with figures at:
- 📊 [EXTENDED_TRAINING_ANALYSIS.md](EXTENDED_TRAINING_ANALYSIS.md) - Detailed report
- 📈 `result_figures/extended_training_analysis.png` - Main analysis plots
- 📉 `result_figures/extended_training_by_strategy.png` - Per-strategy breakdown

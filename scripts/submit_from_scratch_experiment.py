#!/usr/bin/env python3
"""
From-Scratch Training Experiment

Purpose: Determine whether pretrained backbone weights mask the difference
between noise-trained and real-image-trained models.

Hypothesis:
  - With pretrained backbone: noise ≈ baseline (~41 mIoU) because backbone
    features dominate and the model learns spatial label priors
  - Without pretrained backbone: noise → ~5% mIoU (random guessing for 19 classes),
    baseline → ~25-30% mIoU, clearly separating them

Experiment matrix (4 jobs):
  - 2 models: pspnet_r50, segformer_mit-b3
  - 2 strategies: baseline (clear_day only), gen_random_noise (100% noise, ratio 0.0)
  - All on BDD10k dataset
  - 80k iterations with checkpoints every 10k
  - No pretrained backbone (--no-pretrained flag)

Output directory: ${AWARE_DATA_ROOT}/WEIGHTS_FROM_SCRATCH/

Usage:
    # Dry run (preview jobs)
    python scripts/submit_from_scratch_experiment.py --dry-run
    
    # Submit all 4 jobs
    python scripts/submit_from_scratch_experiment.py --submit
    
    # Submit specific model only
    python scripts/submit_from_scratch_experiment.py --submit --models pspnet_r50
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = "${AWARE_DATA_ROOT}/WEIGHTS_FROM_SCRATCH"

# Experiment configuration
DATASET = "BDD10k"
MAX_ITERS = 80000
CHECKPOINT_INTERVAL = 10000
EVAL_INTERVAL = 10000
BATCH_SIZE = 2  # Conservative for from-scratch (no pretrained = harder optimization)
WARMUP_ITERS = 1000

MODELS = ["pspnet_r50", "segformer_mit-b3"]

# Each "experiment" is a (strategy, ratio, domain_filter, label) tuple
EXPERIMENTS = [
    {
        "strategy": "baseline",
        "ratio": 1.0,
        "domain_filter": "clear_day",
        "label": "baseline",
    },
    {
        "strategy": "gen_random_noise",
        "ratio": 0.0,
        "domain_filter": "clear_day",
        "label": "gen_random_noise",
    },
]

# GPU memory requirements per model
MODEL_GMEM = {
    "pspnet_r50": "20G",
    "segformer_mit-b3": "20G",
}


def generate_lsf_script(model: str, experiment: dict) -> tuple:
    """Generate LSF job script. Returns (job_name, work_dir, script)."""
    strategy = experiment["strategy"]
    label = experiment["label"]
    ratio = experiment["ratio"]
    domain_filter = experiment["domain_filter"]

    job_name = f"scratch_{DATASET.lower()}_{model}_{label}"
    work_dir = f"{WEIGHTS_DIR}/{label}/{DATASET.lower()}/{model}"
    gmem = MODEL_GMEM.get(model, "20G")

    # Build training command
    cmd_parts = [
        "python", str(PROJECT_ROOT / "unified_training.py"),
        "--dataset", DATASET,
        "--model", model,
        "--strategy", strategy,
        "--no-pretrained",
        "--max-iters", str(MAX_ITERS),
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--eval-interval", str(EVAL_INTERVAL),
        "--batch-size", str(BATCH_SIZE),
        "--warmup-iters", str(WARMUP_ITERS),
        "--work-dir", work_dir,
    ]
    
    if domain_filter:
        cmd_parts.extend(["--domain-filter", domain_filter])
    
    if strategy.startswith("gen_"):
        cmd_parts.extend(["--real-gen-ratio", str(ratio)])

    training_cmd = " ".join(cmd_parts)

    script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -o {work_dir}/train_%J.out
#BSUB -e {work_dir}/train_%J.err
#BSUB -n 2,4
#BSUB -gpu "num=1:gmem={gmem}"

# ============================================================================
# From-Scratch Training Experiment
# ============================================================================

umask 002
export PYTHONDONTWRITEBYTECODE=1

echo "=========================================="
echo "FROM-SCRATCH EXPERIMENT"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "User: $USER"
echo "Started: $(date)"
echo "Strategy: {strategy}"
echo "Dataset: {DATASET}"
echo "Model: {model}"
echo "Label: {label}"
echo "Max Iters: {MAX_ITERS}"
echo "Batch Size: {BATCH_SIZE}"
echo "No Pretrained: YES"
echo "Work Dir: {work_dir}"
echo "=========================================="

# Create work directory
mkdir -p {work_dir}
chmod 775 {work_dir}

# Activate conda environment
source ~/.bashrc
mamba activate prove

# Pre-flight check
CHECKPOINT="{work_dir}/iter_{MAX_ITERS}.pth"
if [ -f "$CHECKPOINT" ]; then
    SIZE=$(stat -c%s "$CHECKPOINT" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000 ]; then
        echo "WARNING: Checkpoint already exists (size: $SIZE bytes). Skipping."
        exit 0
    fi
fi

# Training
echo ""
echo "Starting training..."
echo "Command: {training_cmd}"
echo ""

{training_cmd}

TRAIN_EXIT_CODE=$?

# Testing (if training succeeded)
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    CHECKPOINT="{work_dir}/iter_{MAX_ITERS}.pth"
    CONFIG="{work_dir}/training_config.py"
    TEST_OUTPUT="{work_dir}/test_results_detailed"

    if [ -f "$CHECKPOINT" ] && [ -f "$CONFIG" ]; then
        echo ""
        echo "=========================================="
        echo "Starting fine-grained testing..."
        echo "=========================================="

        python {PROJECT_ROOT}/fine_grained_test.py \\
            --config "$CONFIG" \\
            --checkpoint "$CHECKPOINT" \\
            --output-dir "$TEST_OUTPUT" \\
            --dataset {DATASET} \\
            --data-root "${AWARE_DATA_ROOT}/FINAL_SPLITS" \\
            --test-split "test" \\
            --batch-size 10

        TEST_EXIT_CODE=$?
        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo "Testing completed successfully"
        else
            echo "WARNING: Testing failed with exit code: $TEST_EXIT_CODE"
        fi
    fi
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi

# Permissions
find {work_dir} -type d -exec chmod 775 {{}} \\; 2>/dev/null || true
find {work_dir} -type f -exec chmod 664 {{}} \\; 2>/dev/null || true

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
"""
    return job_name, work_dir, script


def main():
    parser = argparse.ArgumentParser(description="Submit from-scratch training experiment")
    parser.add_argument("--dry-run", action="store_true", help="Preview jobs without submitting")
    parser.add_argument("--submit", action="store_true", help="Submit jobs to LSF")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help=f"Models to run (default: {MODELS})")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    if not args.dry_run and not args.submit:
        parser.print_help()
        print("\nPlease specify --dry-run or --submit")
        return

    models = args.models or MODELS
    
    # Validate models
    for m in models:
        if m not in MODELS:
            print(f"ERROR: Unknown model '{m}'. Available: {MODELS}")
            return

    jobs = []
    for model in models:
        for exp in EXPERIMENTS:
            job_name, work_dir, script = generate_lsf_script(model, exp)
            jobs.append({
                "job_name": job_name,
                "work_dir": work_dir, 
                "script": script,
                "model": model,
                "label": exp["label"],
            })

    # Print summary
    print(f"\n{'='*60}")
    print(f"FROM-SCRATCH TRAINING EXPERIMENT")
    print(f"{'='*60}")
    print(f"Dataset:    {DATASET}")
    print(f"Models:     {', '.join(models)}")
    print(f"Strategies: baseline, gen_random_noise (100% noise)")
    print(f"Max iters:  {MAX_ITERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Pretrained: NO (training from scratch)")
    print(f"Output:     {WEIGHTS_DIR}/")
    print(f"Total jobs: {len(jobs)}")
    print(f"{'='*60}\n")

    for i, job in enumerate(jobs, 1):
        # Check if already completed
        checkpoint = Path(job["work_dir"]) / f"iter_{MAX_ITERS}.pth"
        status = "DONE" if checkpoint.exists() else "PENDING"
        print(f"  [{i}] {job['job_name']:50s} [{status}]")
        print(f"      → {job['work_dir']}")

    if args.dry_run:
        print(f"\n[DRY RUN] {len(jobs)} jobs would be submitted.")
        # Show first script as example
        if jobs:
            print(f"\nExample script ({jobs[0]['job_name']}):")
            print("-" * 60)
            print(jobs[0]["script"][:2000])
            print("...")
        return
    
    if args.submit:
        # Filter out already-completed jobs
        pending = [j for j in jobs if not (Path(j["work_dir"]) / f"iter_{MAX_ITERS}.pth").exists()]
        if not pending:
            print("\nAll jobs already completed!")
            return
        
        print(f"\n{len(pending)} jobs to submit ({len(jobs) - len(pending)} already completed)")
        
        if not args.yes:
            confirm = input("\nSubmit these jobs? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return
        
        submitted = 0
        for job in pending:
            # Write script to work_dir
            os.makedirs(job["work_dir"], exist_ok=True)
            script_path = Path(job["work_dir"]) / "submit_job.sh"
            with open(script_path, "w") as f:
                f.write(job["script"])
            os.chmod(script_path, 0o775)
            
            # Submit
            result = subprocess.run(
                f"bsub < {script_path}",
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✅ Submitted: {job['job_name']}")
                submitted += 1
            else:
                print(f"  ❌ Failed: {job['job_name']}: {result.stderr.strip()}")
        
        print(f"\n{submitted}/{len(pending)} jobs submitted successfully.")


if __name__ == "__main__":
    main()

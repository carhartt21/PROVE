#!/usr/bin/env python3
"""
Run all sample inference locally on the available GPU.

This script executes all 48 batch inference scripts sequentially,
with progress tracking, skip logic for completed work, and error recovery.

Usage:
    python scripts/run_local_inference.py              # Run all
    python scripts/run_local_inference.py --stage 1    # Stage 1 only
    python scripts/run_local_inference.py --stage 2    # Stage 2 only
    python scripts/run_local_inference.py --dataset BDD10k  # One dataset
    python scripts/run_local_inference.py --dry-run    # Preview only
"""

import os
import sys
import json
import time
import glob
import argparse
import subprocess
import signal
from pathlib import Path
from datetime import datetime, timedelta

SCRIPT_DIR = Path("${AWARE_DATA_ROOT}/SAMPLE_EXTRACTION/inference_scripts")
LOG_DIR = Path("${AWARE_DATA_ROOT}/SAMPLE_EXTRACTION/logs/local")
PRED_ROOT = Path("${AWARE_DATA_ROOT}/SAMPLE_EXTRACTION/testing_samples")


def count_completed_in_batch(script_path):
    """Parse a batch script to find how many of its models already have results."""
    completed = 0
    total = 0
    
    # Quick parse: extract pred_dir paths from the script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find all pred_dir values
    import re
    pred_dirs = re.findall(r'"pred_dir":\s*"([^"]+)"', content)
    total = len(pred_dirs)
    
    for pd in pred_dirs:
        results_file = Path(pd) / "inference_results.json"
        if results_file.exists():
            try:
                with open(results_file) as rf:
                    data = json.load(rf)
                    results = data.get("results", {})
                    if all(r.get("status") == "ok" for r in results.values()) and len(results) > 0:
                        completed += 1
            except (json.JSONDecodeError, KeyError):
                pass
    
    return completed, total


def run_batch_script(script_path, log_file):
    """Run a single batch inference script and capture output."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    cmd = [sys.executable, str(script_path)]
    
    with open(log_file, 'w') as lf:
        lf.write(f"=== Started: {datetime.now().isoformat()} ===\n")
        lf.write(f"Script: {script_path}\n\n")
        lf.flush()
        
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1
        )
        
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            # Print key progress lines
            if line.startswith('[') or 'Done:' in line or 'FAILED' in line or 'All done' in line:
                print(f"  {line.rstrip()}")
        
        proc.wait()
        lf.write(f"\n=== Finished: {datetime.now().isoformat()} (exit code: {proc.returncode}) ===\n")
    
    return proc.returncode


def format_duration(seconds):
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h:.0f}h {m:.0f}m"


def main():
    parser = argparse.ArgumentParser(description="Run inference locally on GPU")
    parser.add_argument('--stage', type=int, choices=[1, 2], help="Run only this stage")
    parser.add_argument('--dataset', type=str, help="Run only this dataset")
    parser.add_argument('--dry-run', action='store_true', help="Preview only")
    parser.add_argument('--skip-completed', action='store_true', default=True,
                        help="Skip batch scripts where all models are done (default: True)")
    parser.add_argument('--no-skip', action='store_true', help="Don't skip any scripts")
    args = parser.parse_args()
    
    if args.no_skip:
        args.skip_completed = False
    
    # Discover batch scripts
    pattern = "batch_stage*.py"
    scripts = sorted(glob.glob(str(SCRIPT_DIR / pattern)))
    
    if not scripts:
        print(f"No batch scripts found in {SCRIPT_DIR}")
        return
    
    # Filter by stage
    if args.stage:
        scripts = [s for s in scripts if f"batch_stage{args.stage}_" in s]
    
    # Filter by dataset
    if args.dataset:
        scripts = [s for s in scripts if args.dataset in s]
    
    print(f"Found {len(scripts)} batch scripts to run")
    print(f"Skip completed: {args.skip_completed}")
    
    # Pre-scan completion status
    scripts_to_run = []
    total_models = 0
    already_done = 0
    
    for script in scripts:
        name = Path(script).stem
        completed, total = count_completed_in_batch(script)
        total_models += total
        already_done += completed
        
        if args.skip_completed and completed == total and total > 0:
            print(f"  SKIP {name}: {completed}/{total} models already done")
        else:
            remaining = total - completed
            scripts_to_run.append((script, completed, total))
            print(f"  QUEUE {name}: {completed}/{total} done, {remaining} remaining")
    
    models_remaining = total_models - already_done
    print(f"\nSummary: {len(scripts_to_run)}/{len(scripts)} scripts to run")
    print(f"  Total models: {total_models}")
    print(f"  Already done: {already_done}")
    print(f"  Remaining: {models_remaining}")
    
    if models_remaining > 0:
        est_seconds = models_remaining * 40  # ~40s per model on A5000
        print(f"  Estimated time: {format_duration(est_seconds)}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would run the above scripts. Use without --dry-run to execute.")
        return
    
    if not scripts_to_run:
        print("\nAll inference is already complete!")
        return
    
    # Setup
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle Ctrl+C gracefully
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n!!! Interrupted by user. Finishing current model... !!!\n")
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run
    start_time = time.time()
    completed_scripts = 0
    failed_scripts = []
    
    for i, (script, done, total) in enumerate(scripts_to_run):
        if interrupted:
            break
        
        name = Path(script).stem
        elapsed = time.time() - start_time
        
        # ETA calculation
        if completed_scripts > 0:
            avg_per_script = elapsed / completed_scripts
            remaining_scripts = len(scripts_to_run) - i
            eta = format_duration(avg_per_script * remaining_scripts)
        else:
            eta = "calculating..."
        
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(scripts_to_run)}] Running: {name}")
        print(f"  Models in batch: {total} ({done} already done)")
        print(f"  Elapsed: {format_duration(elapsed)} | ETA: {eta}")
        print(f"{'='*70}")
        
        log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%H%M%S')}.log"
        
        script_start = time.time()
        returncode = run_batch_script(script, log_file)
        script_duration = time.time() - script_start
        
        if returncode == 0:
            completed_scripts += 1
            print(f"  ✓ Completed in {format_duration(script_duration)}")
        else:
            failed_scripts.append(name)
            print(f"  ✗ Failed (exit code {returncode}) after {format_duration(script_duration)}")
            print(f"    Log: {log_file}")
    
    # Final summary
    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"LOCAL INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"  Duration: {format_duration(total_elapsed)}")
    print(f"  Scripts completed: {completed_scripts}/{len(scripts_to_run)}")
    if failed_scripts:
        print(f"  Failed scripts: {', '.join(failed_scripts)}")
    print(f"  Logs: {LOG_DIR}")
    
    # Count final results
    total_results = 0
    ok_results = 0
    for ds in ['BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k']:
        for stage in ['stage1', 'stage2']:
            pred_base = PRED_ROOT / ds / 'predictions' / stage
            if pred_base.exists():
                for rfile in pred_base.rglob('inference_results.json'):
                    try:
                        with open(rfile) as f:
                            data = json.load(f)
                        for r in data.get('results', {}).values():
                            total_results += 1
                            if r.get('status') == 'ok':
                                ok_results += 1
                    except:
                        pass
    print(f"\n  Total predictions: {ok_results}/{total_results} images OK")


if __name__ == '__main__':
    main()

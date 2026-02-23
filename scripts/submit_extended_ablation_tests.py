#!/usr/bin/env python3
"""
Submit test jobs for ALL checkpoints in extended training ablation study.

Tests every intermediate checkpoint across:
- S1 base (WEIGHTS/): 2k-15k iterations, S1 strategies × 2 datasets × 2 models
- CG base (WEIGHTS_CITYSCAPES_GEN/): 2k-20k iterations, CG strategies × cityscapes × 2 models
- Extended S1 (WEIGHTS_EXTENDED_ABLATION/stage1/): 20k-45k iterations
- Extended CG (WEIGHTS_EXTENDED_ABLATION/cityscapes_gen/): 25k-60k iterations

Jobs are GROUPED by (strategy, dataset, model) so that all iterations for a
given configuration run sequentially within a single LSF job. This minimizes
queue impact (~30 grouped jobs instead of ~434 individual ones).

Usage:
    # Count all untested checkpoints (dry run)
    python scripts/submit_extended_ablation_tests.py --dry-run

    # Submit all grouped test jobs
    python scripts/submit_extended_ablation_tests.py -y

    # Only S1-related sources (base + extended)
    python scripts/submit_extended_ablation_tests.py --source s1-base ext-s1 --dry-run

    # Only extended CG checkpoints
    python scripts/submit_extended_ablation_tests.py --source ext-cg --dry-run

    # Limit grouped job submissions
    python scripts/submit_extended_ablation_tests.py --limit 10 -y
"""

import os
import sys
import json
import argparse
import subprocess
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_S1 = Path('${AWARE_DATA_ROOT}/WEIGHTS')
WEIGHTS_CG = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')
WEIGHTS_EXT = Path('${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED_ABLATION')
DATA_ROOT = '${AWARE_DATA_ROOT}/FINAL_SPLITS'
LOG_DIR = PROJECT_ROOT / 'logs'

# Extended training ablation strategies
S1_STRATEGIES = ['baseline', 'gen_Img2Img', 'gen_augmenters', 'gen_cycleGAN', 'std_randaugment']
CG_STRATEGIES = ['baseline', 'gen_augmenters', 'gen_Img2Img', 'gen_CUT', 'std_randaugment']

S1_DATASETS = ['bdd10k', 'iddaw']
CG_DATASETS = ['cityscapes']

MODELS = ['pspnet_r50', 'segformer_mit-b3']

# Dataset display names for fine_grained_test.py --dataset argument
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'iddaw': 'IDD-AW',
    'cityscapes': 'Cityscapes',
}

# Datasets requiring --use-native-classes
NATIVE_CLASS_DATASETS = {'mapillaryvistas', 'outside15k'}

# GPU memory per model
MODEL_GMEM = {
    'pspnet_r50': '16G',
    'segformer_mit-b3': '16G',
}

# Time budget: minutes per checkpoint test (conservative estimate)
MINUTES_PER_CHECKPOINT = 20
# Maximum walltime for any single grouped job (hours)
MAX_WALLTIME_HOURS = 12


@dataclass
class CheckpointEntry:
    """A single checkpoint within a grouped job."""
    source: str          # 's1-base', 'cg-base', 'ext-s1', 'ext-cg'
    checkpoint_path: Path
    config_path: Path
    output_dir: Path     # test_results_detailed/iter_XXXXX/
    iteration: int
    already_tested: bool = False


@dataclass
class GroupedTestJob:
    """A grouped test job: all iterations for one (strategy, dataset, model) combo."""
    strategy: str
    dataset: str
    model: str
    model_dir: str       # directory name (may include _ratio0p50)
    checkpoints: List[CheckpointEntry] = field(default_factory=list)

    @property
    def group_key(self) -> str:
        return f"{self.strategy}/{self.dataset}/{self.model}"

    @property
    def job_name(self) -> str:
        strat_short = self.strategy[:15]
        model_short = self.model.split('_')[0]
        n = len(self.actionable_checkpoints)
        return f"tExt_{strat_short}_{self.dataset}_{model_short}_{n}ckpt"[:80]

    @property
    def actionable_checkpoints(self) -> List[CheckpointEntry]:
        return [c for c in self.checkpoints if not c.already_tested]

    @property
    def total_checkpoints(self) -> int:
        return len(self.checkpoints)

    @property
    def num_actionable(self) -> int:
        return len(self.actionable_checkpoints)

    @property
    def num_skipped(self) -> int:
        return len(self.checkpoints) - self.num_actionable

    @property
    def iter_range_str(self) -> str:
        """Human-readable iteration range."""
        iters = sorted(c.iteration for c in self.actionable_checkpoints)
        if not iters:
            return "none"
        return f"{iters[0]//1000}k-{iters[-1]//1000}k ({len(iters)} ckpts)"

    @property
    def walltime_str(self) -> str:
        """Compute walltime based on number of checkpoints."""
        n = self.num_actionable
        minutes = n * MINUTES_PER_CHECKPOINT
        hours = min(math.ceil(minutes / 60), MAX_WALLTIME_HOURS)
        return f"{hours:02d}:00"


def get_model_dir_name(strategy: str, model: str) -> str:
    """Get the model directory name (with _ratio0p50 for gen strategies)."""
    if strategy in ('baseline', 'std_randaugment'):
        return model
    return f'{model}_ratio0p50'


def find_config(model_dir: Path) -> Optional[Path]:
    """Find the training config."""
    cfg = model_dir / 'training_config.py'
    if cfg.exists():
        return cfg
    cfg2 = model_dir / 'configs' / 'training_config.py'
    if cfg2.exists():
        return cfg2
    return None


def has_test_result(test_results_dir: Path, iteration: int) -> bool:
    """Check if a specific iteration already has valid test results."""
    # Check iter-specific subdirectory
    iter_dir = test_results_dir / f'iter_{iteration}'
    if iter_dir.exists():
        for ts_dir in iter_dir.iterdir():
            rj = ts_dir / 'results.json'
            if rj.exists():
                try:
                    with open(rj) as f:
                        data = json.load(f)
                    miou = data.get('overall', {}).get('mIoU', 0)
                    if miou and miou > 5:
                        return True
                except Exception:
                    continue

    # Also check top-level timestamp dirs (for backward compat)
    if not test_results_dir.exists():
        return False
    for ts_dir in test_results_dir.iterdir():
        if not ts_dir.is_dir() or not ts_dir.name.startswith('202'):
            continue
        rj = ts_dir / 'results.json'
        if rj.exists():
            try:
                with open(rj) as f:
                    data = json.load(f)
                miou = data.get('overall', {}).get('mIoU', 0)
                ckpt = data.get('checkpoint', '')
                if f'iter_{iteration}' in str(ckpt) and miou and miou > 5:
                    return True
            except Exception:
                continue
    return False


def discover_checkpoints(model_dir: Path) -> List[Tuple[int, Path]]:
    """Find all checkpoints in a model directory, return (iteration, path) pairs."""
    ckpts = []
    for f in model_dir.glob('iter_*.pth'):
        match = re.match(r'iter_(\d+)\.pth', f.name)
        if match:
            iteration = int(match.group(1))
            if f.stat().st_size > 1000:
                ckpts.append((iteration, f))
    return sorted(ckpts, key=lambda x: x[0])


def generate_grouped_jobs(
    sources: Optional[List[str]] = None,
    force: bool = False,
) -> List[GroupedTestJob]:
    """Generate grouped test jobs: one per (strategy, dataset, model), containing
    all checkpoints across base and extended sources."""
    if sources is None:
        sources = ['s1-base', 'cg-base', 'ext-s1', 'ext-cg']

    # Accumulate checkpoints into groups keyed by (strategy, dataset, model)
    groups: Dict[str, GroupedTestJob] = {}

    def get_or_create_group(strategy: str, dataset: str, model: str, model_dir_name: str) -> GroupedTestJob:
        key = f"{strategy}/{dataset}/{model}"
        if key not in groups:
            groups[key] = GroupedTestJob(
                strategy=strategy,
                dataset=dataset,
                model=model,
                model_dir=model_dir_name,
            )
        return groups[key]

    # S1 base checkpoints (WEIGHTS/)
    if 's1-base' in sources:
        for strat in S1_STRATEGIES:
            for ds in S1_DATASETS:
                for model in MODELS:
                    model_dir_name = get_model_dir_name(strat, model)
                    model_dir = WEIGHTS_S1 / strat / ds / model_dir_name
                    if not model_dir.exists():
                        continue
                    config = find_config(model_dir)
                    if not config:
                        continue

                    test_results_dir = model_dir / 'test_results_detailed'
                    group = get_or_create_group(strat, ds, model, model_dir_name)
                    for iteration, ckpt_path in discover_checkpoints(model_dir):
                        if iteration > 15000:
                            continue
                        output_subdir = test_results_dir / f'iter_{iteration}'
                        already = not force and has_test_result(test_results_dir, iteration)
                        group.checkpoints.append(CheckpointEntry(
                            source='s1-base',
                            checkpoint_path=ckpt_path,
                            config_path=config,
                            output_dir=output_subdir,
                            iteration=iteration,
                            already_tested=already,
                        ))

    # CG base checkpoints (WEIGHTS_CITYSCAPES_GEN/)
    if 'cg-base' in sources:
        for strat in CG_STRATEGIES:
            for model in MODELS:
                model_dir_name = get_model_dir_name(strat, model)
                model_dir = WEIGHTS_CG / strat / 'cityscapes' / model_dir_name
                if not model_dir.exists():
                    continue
                config = find_config(model_dir)
                if not config:
                    continue

                test_results_dir = model_dir / 'test_results_detailed'
                group = get_or_create_group(strat, 'cityscapes', model, model_dir_name)
                for iteration, ckpt_path in discover_checkpoints(model_dir):
                    if iteration > 20000:
                        continue
                    output_subdir = test_results_dir / f'iter_{iteration}'
                    already = not force and has_test_result(test_results_dir, iteration)
                    group.checkpoints.append(CheckpointEntry(
                        source='cg-base',
                        checkpoint_path=ckpt_path,
                        config_path=config,
                        output_dir=output_subdir,
                        iteration=iteration,
                        already_tested=already,
                    ))

    # Extended S1 checkpoints (WEIGHTS_EXTENDED_ABLATION/stage1/)
    if 'ext-s1' in sources:
        for strat in S1_STRATEGIES:
            for ds in S1_DATASETS:
                for model in MODELS:
                    model_dir_name = get_model_dir_name(strat, model)
                    model_dir = WEIGHTS_EXT / 'stage1' / strat / ds / model_dir_name
                    if not model_dir.exists():
                        continue
                    config = find_config(model_dir)
                    if not config:
                        continue

                    test_results_dir = model_dir / 'test_results_detailed'
                    group = get_or_create_group(strat, ds, model, model_dir_name)
                    for iteration, ckpt_path in discover_checkpoints(model_dir):
                        output_subdir = test_results_dir / f'iter_{iteration}'
                        already = not force and has_test_result(test_results_dir, iteration)
                        group.checkpoints.append(CheckpointEntry(
                            source='ext-s1',
                            checkpoint_path=ckpt_path,
                            config_path=config,
                            output_dir=output_subdir,
                            iteration=iteration,
                            already_tested=already,
                        ))

    # Extended CG checkpoints (WEIGHTS_EXTENDED_ABLATION/cityscapes_gen/)
    if 'ext-cg' in sources:
        for strat in CG_STRATEGIES:
            for model in MODELS:
                model_dir_name = get_model_dir_name(strat, model)
                model_dir = WEIGHTS_EXT / 'cityscapes_gen' / strat / 'cityscapes' / model_dir_name
                if not model_dir.exists():
                    continue
                config = find_config(model_dir)
                if not config:
                    continue

                test_results_dir = model_dir / 'test_results_detailed'
                group = get_or_create_group(strat, 'cityscapes', model, model_dir_name)
                for iteration, ckpt_path in discover_checkpoints(model_dir):
                    output_subdir = test_results_dir / f'iter_{iteration}'
                    already = not force and has_test_result(test_results_dir, iteration)
                    group.checkpoints.append(CheckpointEntry(
                        source='ext-cg',
                        checkpoint_path=ckpt_path,
                        config_path=config,
                        output_dir=output_subdir,
                        iteration=iteration,
                        already_tested=already,
                    ))

    # Sort checkpoints within each group by iteration
    for group in groups.values():
        group.checkpoints.sort(key=lambda c: c.iteration)

    # Return only groups that have actionable checkpoints
    return sorted(groups.values(), key=lambda g: g.group_key)


def generate_grouped_job_script(group: GroupedTestJob) -> str:
    """Generate a single LSF job script that tests all checkpoints sequentially."""
    gmem = MODEL_GMEM.get(group.model, '16G')
    test_dataset = DATASET_DISPLAY.get(group.dataset, group.dataset)
    walltime = group.walltime_str

    native_flag = ''
    if group.dataset.lower() in NATIVE_CLASS_DATASETS:
        native_flag = '    --use-native-classes \\\n'

    ckpts = group.actionable_checkpoints

    # Build sequential test commands
    test_commands = []
    for i, ckpt in enumerate(ckpts, 1):
        test_commands.append(f'''
echo ""
echo "--- Checkpoint {i}/{len(ckpts)}: iter_{ckpt.iteration} ({ckpt.source}) ---"
echo "Config:     {ckpt.config_path}"
echo "Checkpoint: {ckpt.checkpoint_path}"
echo "Output:     {ckpt.output_dir}"
echo "Started:    $(date)"

python {PROJECT_ROOT}/fine_grained_test.py \\
    --config "{ckpt.config_path}" \\
    --checkpoint "{ckpt.checkpoint_path}" \\
    --output-dir "{ckpt.output_dir}" \\
    --dataset {test_dataset} \\
    --data-root "{DATA_ROOT}" \\
    --test-split test \\
{native_flag}    --batch-size 10

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ iter_{ckpt.iteration} completed successfully"
    PASSED=$((PASSED + 1))
    find "{ckpt.output_dir}" -type d -exec chmod 775 {{}} \\; 2>/dev/null || true
    find "{ckpt.output_dir}" -type f -exec chmod 664 {{}} \\; 2>/dev/null || true
else
    echo "  ✗ iter_{ckpt.iteration} FAILED (exit code: $EXIT_CODE)"
    FAILED=$((FAILED + 1))
fi''')

    test_block = '\n'.join(test_commands)

    script = f'''#!/bin/bash
#BSUB -J {group.job_name}
#BSUB -q BatchGPU
#BSUB -n 4
#BSUB -M 32000
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1:gmem={gmem}"
#BSUB -W {walltime}
#BSUB -o {LOG_DIR}/{group.job_name}_%J.out
#BSUB -e {LOG_DIR}/{group.job_name}_%J.err

umask 002
source ~/.bashrc
mamba activate prove
cd {PROJECT_ROOT}

PASSED=0
FAILED=0

echo "=========================================="
echo "Extended Training Ablation - GROUPED Checkpoint Test"
echo "Job ID:   $LSB_JOBID"
echo "Host:     $(hostname)"
echo "Started:  $(date)"
echo "Strategy: {group.strategy}"
echo "Dataset:  {group.dataset}"
echo "Model:    {group.model_dir}"
echo "Checkpoints to test: {len(ckpts)}"
echo "Iterations: {', '.join(str(c.iteration) for c in ckpts)}"
echo "Walltime: {walltime}"
echo "=========================================="
{test_block}

echo ""
echo "=========================================="
echo "SUMMARY: $PASSED passed, $FAILED failed out of {len(ckpts)} checkpoints"
echo "Finished: $(date)"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
exit 0
'''
    return script


def get_running_jobs() -> Set[str]:
    """Get currently running/pending job names."""
    running = set()
    try:
        result = subprocess.run(
            ['bjobs', '-u', os.environ.get('USER', '${USER}'), '-w'],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) >= 7 and parts[2] in ('RUN', 'PEND'):
                running.add(parts[6].lower())
    except Exception:
        pass
    return running


def submit_grouped_job(group: GroupedTestJob, dry_run: bool = False) -> Optional[str]:
    """Submit a grouped test job. Returns job ID or None."""
    script = generate_grouped_job_script(group)

    if dry_run:
        return 'DRY-RUN'

    # Write script to logs directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = LOG_DIR / f'{group.job_name}.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    # Also create output dirs for all checkpoints
    for ckpt in group.actionable_checkpoints:
        ckpt.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ['bsub'], input=script, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            match = re.search(r'Job <(\d+)>', result.stdout)
            if match:
                return match.group(1)
        print(f"  ERROR submitting {group.job_name}: {result.stderr.strip()}")
    except Exception as e:
        print(f"  ERROR submitting {group.job_name}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Submit GROUPED test jobs for all extended training ablation checkpoints'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be submitted without submitting')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--source', nargs='+',
                        choices=['s1-base', 'cg-base', 'ext-s1', 'ext-cg'],
                        help='Only test checkpoints from specific source(s)')
    parser.add_argument('--force', action='store_true',
                        help='Force retest even if results exist')
    parser.add_argument('--limit', type=int,
                        help='Maximum number of grouped jobs to submit')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extended Training Ablation - GROUPED Checkpoint Testing")
    print("=" * 70)

    # Generate grouped jobs
    all_groups = generate_grouped_jobs(sources=args.source, force=args.force)

    # Compute totals
    total_ckpts = sum(g.total_checkpoints for g in all_groups)
    total_actionable = sum(g.num_actionable for g in all_groups)
    total_skipped = sum(g.num_skipped for g in all_groups)

    print(f"\nTotal groups (strategy/dataset/model combos): {len(all_groups)}")
    print(f"Total checkpoints found: {total_ckpts}")
    print(f"Already tested (skip): {total_skipped}")
    print(f"Need testing: {total_actionable}")

    # Filter to groups with actionable checkpoints
    actionable_groups = [g for g in all_groups if g.num_actionable > 0]

    # Breakdown by source
    print("\nBreakdown by source:")
    for src in ['s1-base', 'cg-base', 'ext-s1', 'ext-cg']:
        src_total = sum(
            len([c for c in g.checkpoints if c.source == src])
            for g in all_groups
        )
        src_action = sum(
            len([c for c in g.actionable_checkpoints if c.source == src])
            for g in actionable_groups
        )
        if src_total > 0:
            print(f"  {src:>8}: {src_total:3d} total, {src_action:3d} to test")

    # Breakdown by strategy
    print("\nBreakdown by strategy (actionable):")
    strat_map = defaultdict(lambda: {'groups': 0, 'ckpts': 0})
    for g in actionable_groups:
        strat_map[g.strategy]['groups'] += 1
        strat_map[g.strategy]['ckpts'] += g.num_actionable
    for strat in sorted(strat_map.keys()):
        info = strat_map[strat]
        print(f"  {strat:<25}: {info['groups']:2d} jobs, {info['ckpts']:3d} checkpoints")

    if not actionable_groups:
        print("\n✅ All checkpoints already tested!")
        return

    # Detailed group listing
    print(f"\nGrouped jobs to submit ({len(actionable_groups)}):")
    print(f"  {'Group':<50} {'Ckpts':>5} {'Range':<25} {'Walltime':>8}")
    print(f"  {'-'*50} {'-'*5} {'-'*25} {'-'*8}")
    for g in actionable_groups:
        print(f"  {g.group_key:<50} {g.num_actionable:>5} {g.iter_range_str:<25} {g.walltime_str:>8}")

    total_walltime_h = sum(
        min(math.ceil(g.num_actionable * MINUTES_PER_CHECKPOINT / 60), MAX_WALLTIME_HOURS)
        for g in actionable_groups
    )
    print(f"\n  Total: {len(actionable_groups)} grouped jobs, "
          f"{total_actionable} checkpoints, "
          f"~{total_walltime_h} GPU-hours requested")

    # Apply limit
    to_submit = actionable_groups
    if args.limit:
        to_submit = actionable_groups[:args.limit]
        if len(actionable_groups) > args.limit:
            print(f"\n⚠️  Limited to {args.limit} of {len(actionable_groups)} grouped jobs")

    # Check for running duplicates
    running = get_running_jobs()
    final_submit = []
    for g in to_submit:
        if g.job_name.lower() in running:
            print(f"  ⚠️  Skipping {g.job_name} (already running/pending)")
        else:
            final_submit.append(g)

    if len(final_submit) < len(to_submit):
        print(f"\n⚠️  {len(to_submit) - len(final_submit)} jobs already running/pending")

    n_ckpts_submit = sum(g.num_actionable for g in final_submit)
    print(f"\nWill submit: {len(final_submit)} grouped jobs ({n_ckpts_submit} checkpoints)")

    if args.dry_run:
        print("\n--- DRY RUN (nothing submitted) ---")
        return

    if not args.yes:
        response = input(f"\nSubmit {len(final_submit)} grouped jobs? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return

    # Submit
    submitted = 0
    failed = 0
    job_ids = []

    for i, group in enumerate(final_submit, 1):
        job_id = submit_grouped_job(group, dry_run=False)
        if job_id:
            submitted += 1
            job_ids.append(job_id)
            print(f"  [{i}/{len(final_submit)}] {group.group_key} "
                  f"({group.num_actionable} ckpts) → Job {job_id}")
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"Submitted: {submitted} grouped jobs | Failed: {failed}")
    if job_ids:
        print(f"Job ID range: {job_ids[0]} - {job_ids[-1]}")
    total_ckpts_submitted = sum(g.num_actionable for g in final_submit[:submitted])
    print(f"Total checkpoints queued: {total_ckpts_submitted}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

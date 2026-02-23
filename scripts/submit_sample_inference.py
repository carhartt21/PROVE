#!/usr/bin/env python3
"""
Batch Inference Submission for Sample Extraction

Submits LSF jobs for running inference on the 10 extracted test images
per dataset using all available iter_15000 checkpoints.

Groups multiple models into single jobs (by dataset) to reduce LSF overhead.
Each job processes multiple strategy/model combinations for one dataset.

Usage:
    # Dry run - see what would be submitted
    python scripts/submit_sample_inference.py --dry-run

    # Submit all jobs
    python scripts/submit_sample_inference.py --submit

    # Submit limited number of jobs
    python scripts/submit_sample_inference.py --submit --limit 10

    # Submit only for specific dataset
    python scripts/submit_sample_inference.py --submit --dataset BDD10k

    # Submit only for specific stage
    python scripts/submit_sample_inference.py --submit --stage stage1
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = Path('${AWARE_DATA_ROOT}/SAMPLE_EXTRACTION')
WEIGHTS_STAGE1 = Path('${AWARE_DATA_ROOT}/WEIGHTS')
WEIGHTS_STAGE2 = Path('${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2')
CHECKPOINT_ITER = 'iter_15000'

# Map weights dir names to dataset names used in testing_manifest
DIR_TO_DATASET = {
    'bdd10k': 'BDD10k',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}

# Maximum checkpoints to process per job (keeps job under 1h wall time)
MODELS_PER_JOB = 20


def load_testing_manifest():
    """Load the testing manifest created by extract phase."""
    manifest_path = OUTPUT_ROOT / 'metadata' / 'testing_manifest.json'
    if not manifest_path.exists():
        print(f"ERROR: Testing manifest not found at {manifest_path}")
        print("Run: python scripts/extract_samples_and_predictions.py --phase extract")
        sys.exit(1)
    with open(manifest_path) as f:
        return json.load(f)


def discover_checkpoints():
    """Discover all iter_15000 checkpoints, grouped by dataset."""
    checkpoints = defaultdict(list)  # {(stage, dataset_name): [ckpt_info, ...]}

    for stage_name, weights_root in [('stage1', WEIGHTS_STAGE1), ('stage2', WEIGHTS_STAGE2)]:
        if not weights_root.exists():
            continue

        for ckpt_path in sorted(weights_root.rglob(f'{CHECKPOINT_ITER}.pth')):
            rel_path = ckpt_path.relative_to(weights_root)
            parts = rel_path.parts

            if len(parts) < 4:
                continue

            strategy = parts[0]
            dataset_dir = parts[1]
            model_dir = parts[2]

            dataset_name = DIR_TO_DATASET.get(dataset_dir)
            if not dataset_name:
                continue

            config_path = ckpt_path.parent / 'training_config.py'
            if not config_path.exists():
                continue

            # Check if already done
            pred_dir = OUTPUT_ROOT / 'testing_samples' / dataset_name / 'predictions' / stage_name / strategy / model_dir
            if (pred_dir / 'inference_results.json').exists():
                continue

            checkpoints[(stage_name, dataset_name)].append({
                'strategy': strategy,
                'model_dir': model_dir,
                'config_path': str(config_path),
                'checkpoint_path': str(ckpt_path),
                'pred_dir': str(pred_dir),
            })

    return checkpoints


def generate_batch_inference_script(dataset_name: str, stage: str,
                                      ckpt_batch: list,
                                      test_images: list) -> str:
    """Generate a Python script that runs inference for multiple checkpoints."""

    script = f'''#!/usr/bin/env python3
"""Auto-generated batch inference: {stage} / {dataset_name} ({len(ckpt_batch)} models)"""
import os, sys, json, traceback
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, "{PROJECT_ROOT}")
import custom_transforms
import custom_losses

from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import mmseg.models
import warnings
warnings.filterwarnings('ignore')
register_all_modules(init_default_scope=True)

PALETTES = {{
    19: {json.dumps([
        [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
        [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
        [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
        [0,60,100],[0,80,100],[0,0,230],[119,11,32]
    ])},
    24: {json.dumps([
        [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
        [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
        [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
        [0,60,100],[0,80,100],[0,0,230],[119,11,32],
        [100,100,100],[165,42,42],[0,170,30],[140,140,140],[0,0,0]
    ])},
}}
# Generate 66-class palette
p66 = list(PALETTES[19])
for i in range(19, 66):
    p66.append([(i*67+37)%256, (i*113+59)%256, (i*179+83)%256])
PALETTES[66] = p66


def colorize_mask(mask, num_classes):
    palette = PALETTES.get(num_classes, PALETTES[19])
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(palette))):
        colored[mask == c] = palette[c]
    return colored


def create_side_by_side(img, pred_colored, save_path):
    h = max(img.shape[0], pred_colored.shape[0])
    w1, w2 = img.shape[1], pred_colored.shape[1]
    canvas = np.zeros((h, w1 + w2 + 10, 3), dtype=np.uint8)
    canvas[:img.shape[0], :w1] = img
    canvas[:pred_colored.shape[0], w1+10:] = pred_colored
    cv2.imwrite(str(save_path), canvas)


def detect_num_classes(cfg):
    if hasattr(cfg, 'model') and 'decode_head' in cfg.model:
        dh = cfg.model.decode_head
        if isinstance(dh, list):
            for h in dh:
                if 'num_classes' in h:
                    return h['num_classes']
        elif isinstance(dh, dict):
            return dh.get('num_classes', 19)
    return 19


def run_inference_for_checkpoint(config_path, checkpoint_path, pred_dir,
                                   test_images, test_names, device):
    """Run inference for one checkpoint on all test images."""
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(config_path)
    model_num_classes = detect_num_classes(cfg)
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    results = {{}}
    for img_path_str, img_name in zip(test_images, test_names):
        try:
            img = cv2.imread(img_path_str)
            if img is None:
                results[img_name] = {{"status": "error", "error": "imread returned None"}}
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            # Center crop: resize shorter side to 512, then center crop to 512x512
            scale = 512.0 / min(h_orig, w_orig)
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            img_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            y_off = (new_h - 512) // 2
            x_off = (new_w - 512) // 2
            img_cropped = img_scaled[y_off:y_off+512, x_off:x_off+512]

            img_norm = (img_cropped.astype(np.float32) - mean) / std
            img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            with torch.no_grad():
                img_metas = [dict(ori_shape=(h_orig, w_orig), img_shape=(512, 512),
                                  pad_shape=(512, 512), scale_factor=(1.0, 1.0))]
                result = model.inference(img_tensor, img_metas)

            if isinstance(result, torch.Tensor):
                pred = result[0].argmax(dim=0).cpu().numpy() if result.ndim >= 3 else result.cpu().numpy()
            elif isinstance(result, list):
                r = result[0]
                if hasattr(r, 'pred_sem_seg'):
                    pred = r.pred_sem_seg.data.squeeze()
                    if pred.ndim == 3:
                        pred = pred.argmax(dim=0)
                    pred = pred.cpu().numpy()
                else:
                    pred = r.cpu().numpy() if isinstance(r, torch.Tensor) else np.array(r)
            else:
                pred = result.cpu().numpy() if isinstance(result, torch.Tensor) else np.array(result)

            pred = pred.squeeze().astype(np.uint8)
            pred_colored = colorize_mask(pred, model_num_classes)

            safe_name = img_name.replace('.', '_')
            cv2.imwrite(str(pred_dir / f"{{safe_name}}_pred_raw.png"), pred)
            cv2.imwrite(str(pred_dir / f"{{safe_name}}_pred_color.png"),
                        cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))

            img_display = img_cropped.copy()
            create_side_by_side(
                cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR),
                pred_dir / f"{{safe_name}}_sbs.png"
            )
            results[img_name] = {{"status": "ok"}}
        except Exception as e:
            results[img_name] = {{"status": "error", "error": str(e)}}

    with open(pred_dir / "inference_results.json", 'w') as f:
        json.dump({{
            "config": config_path,
            "checkpoint": checkpoint_path,
            "model_num_classes": model_num_classes,
            "results": results
        }}, f, indent=2)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    return ok, len(results)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {{device}}")

    test_images = {json.dumps([img['original_path'] for img in test_images])}
    test_names = {json.dumps([img['filename'] for img in test_images])}

    checkpoints = {json.dumps(ckpt_batch)}

    total_ok = 0
    total_imgs = 0
    for i, ckpt in enumerate(checkpoints):
        print(f"\\n[{{i+1}}/{{len(checkpoints)}}] {{ckpt['strategy']}}/{{ckpt['model_dir']}}")
        try:
            ok, n = run_inference_for_checkpoint(
                ckpt['config_path'], ckpt['checkpoint_path'],
                ckpt['pred_dir'], test_images, test_names, device
            )
            total_ok += ok
            total_imgs += n
            print(f"  Done: {{ok}}/{{n}} images OK")
        except Exception as e:
            print(f"  FAILED: {{e}}")
            traceback.print_exc()

    print(f"\\nAll done: {{total_ok}}/{{total_imgs}} total images processed across {{len(checkpoints)}} models")


if __name__ == '__main__':
    main()
'''
    return script


def main():
    parser = argparse.ArgumentParser(description='Submit batch inference jobs for sample extraction')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be submitted')
    parser.add_argument('--submit', action='store_true', help='Actually submit LSF jobs')
    parser.add_argument('--generate-only', action='store_true', help='Generate scripts without submitting')
    parser.add_argument('--limit', type=int, default=None, help='Max number of jobs')
    parser.add_argument('--dataset', type=str, default=None, help='Only process specific dataset')
    parser.add_argument('--stage', type=str, default=None, choices=['stage1', 'stage2'],
                       help='Only process specific stage')
    parser.add_argument('-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    if not args.dry_run and not args.submit and not args.generate_only:
        print("Specify --dry-run, --submit, or --generate-only")
        sys.exit(1)

    testing_manifest = load_testing_manifest()
    checkpoints = discover_checkpoints()

    # Filter if requested
    if args.dataset or args.stage:
        filtered = {}
        for (stage, dataset), ckpts in checkpoints.items():
            if args.dataset and dataset != args.dataset:
                continue
            if args.stage and stage != args.stage:
                continue
            filtered[(stage, dataset)] = ckpts
        checkpoints = filtered

    # Group into batches for job submission
    jobs = []
    for (stage, dataset), ckpts in sorted(checkpoints.items()):
        if not ckpts:
            continue
        test_images = testing_manifest.get(dataset, {}).get('images', [])
        if not test_images:
            continue

        # Split into batches of MODELS_PER_JOB
        for batch_idx in range(0, len(ckpts), MODELS_PER_JOB):
            batch = ckpts[batch_idx:batch_idx + MODELS_PER_JOB]
            jobs.append({
                'stage': stage,
                'dataset': dataset,
                'batch_idx': batch_idx // MODELS_PER_JOB,
                'checkpoints': batch,
                'test_images': test_images,
            })

    total_models = sum(len(j['checkpoints']) for j in jobs)
    print(f"\nCheckpoints to process: {total_models} across {len(jobs)} batch jobs")
    print(f"  (Each job processes up to {MODELS_PER_JOB} models on {NUM_SAMPLES} test images)")

    if args.limit:
        jobs = jobs[:args.limit]
        print(f"  Limited to {len(jobs)} jobs")

    # Print summary by stage/dataset
    from collections import Counter
    stage_counts = Counter()
    dataset_counts = Counter()
    for j in jobs:
        stage_counts[j['stage']] += len(j['checkpoints'])
        dataset_counts[j['dataset']] += len(j['checkpoints'])
    print(f"\n  By stage: {dict(stage_counts)}")
    print(f"  By dataset: {dict(dataset_counts)}")

    if args.dry_run:
        print("\n[DRY-RUN] Would submit these jobs:")
        for j in jobs[:20]:
            strategies = [c['strategy'] + '/' + c['model_dir'] for c in j['checkpoints']]
            print(f"  {j['stage']}/{j['dataset']} batch{j['batch_idx']}: {len(j['checkpoints'])} models")
            for s in strategies[:3]:
                print(f"    - {s}")
            if len(strategies) > 3:
                print(f"    ... and {len(strategies)-3} more")
        if len(jobs) > 20:
            print(f"  ... and {len(jobs)-20} more jobs")
        print(f"\nTotal: {len(jobs)} jobs, {total_models} model inferences")
        print("Re-run with --submit or --generate-only")
        return

    # Generate scripts
    scripts_dir = OUTPUT_ROOT / 'inference_scripts'
    jobs_dir = OUTPUT_ROOT / 'jobs'
    logs_dir = OUTPUT_ROOT / 'logs'
    for d in [scripts_dir, jobs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    generated = 0
    for job in jobs:
        stage = job['stage']
        dataset = job['dataset']
        batch_idx = job['batch_idx']
        ckpt_batch = job['checkpoints']
        test_images = job['test_images']

        script_content = generate_batch_inference_script(
            dataset, stage, ckpt_batch, test_images
        )

        script_name = f"batch_{stage}_{dataset}_{batch_idx:03d}.py"
        script_path = scripts_dir / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)
        generated += 1

    print(f"\nGenerated {generated} batch scripts in {scripts_dir}")

    if args.generate_only:
        print("Scripts generated. Run locally with: python scripts/run_local_inference.py")
        return

    # Confirm submission
    if not args.y:
        print(f"\nAbout to submit {len(jobs)} LSF jobs ({total_models} model inferences)")
        response = input("Continue? [y/N] ")
        if response.lower() != 'y':
            print("Aborted")
            return

    submitted = 0
    for job in jobs:
        stage = job['stage']
        dataset = job['dataset']
        batch_idx = job['batch_idx']
        ckpt_batch = job['checkpoints']

        script_name = f"batch_{stage}_{dataset}_{batch_idx:03d}.py"
        script_path = scripts_dir / script_name

        # Wall time: ~3 min per model × MODELS_PER_JOB = up to 60 min
        wall_time = min(120, max(30, len(ckpt_batch) * 4))
        wall_hours = wall_time // 60
        wall_mins = wall_time % 60

        job_name = f"sinf_{stage[:2]}_{dataset[:5]}_{batch_idx}"
        job_script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {logs_dir}/{script_name.replace('.py', '.out')}
#BSUB -e {logs_dir}/{script_name.replace('.py', '.err')}
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:gmem=10G"
#BSUB -W {wall_hours:02d}:{wall_mins:02d}

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd {PROJECT_ROOT}
python {script_path}
"""
        job_path = jobs_dir / script_name.replace('.py', '.sh')
        with open(job_path, 'w') as f:
            f.write(job_script)

        # Submit via bsub
        result = subprocess.run(
            ['bsub'],
            input=job_script,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            submitted += 1
            if submitted <= 5 or submitted % 10 == 0:
                job_id = result.stdout.strip()
                print(f"  [{submitted}] {job_name}: {job_id}")
        else:
            print(f"  ERROR submitting {job_name}: {result.stderr}")

    print(f"\nSubmitted {submitted}/{len(jobs)} jobs")
    print(f"Monitor with: bjobs -w | grep sinf_")


NUM_SAMPLES = 10  # Must match extract script

if __name__ == '__main__':
    main()

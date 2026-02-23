#!/usr/bin/env python3
"""
Fill Missing Predictions for Publication Figures

Identifies and submits inference jobs for strategy×dataset×stage combinations
that are missing from the SAMPLE_EXTRACTION predictions directory.

This script complements submit_sample_inference.py by targeting specific gaps
rather than discovering all possible checkpoints.

Usage:
    python scripts/fill_missing_predictions.py --dry-run
    python scripts/fill_missing_predictions.py --submit
    python scripts/fill_missing_predictions.py --submit -y
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

# All strategies we want predictions for
ALL_STRATEGIES = [
    'baseline',
    'std_autoaugment', 'std_cutmix', 'std_mixup', 'std_randaugment', 'std_photometric_distort',
    'gen_albumentations_weather', 'gen_Attribute_Hallucination', 'gen_augmenters', 'gen_automold',
    'gen_CNetSeg', 'gen_CUT', 'gen_cyclediffusion', 'gen_cycleGAN', 'gen_flux_kontext',
    'gen_Img2Img', 'gen_IP2P', 'gen_LANIT', 'gen_Qwen_Image_Edit', 'gen_stargan_v2',
    'gen_step1x_new', 'gen_step1x_v1p2', 'gen_SUSTechGAN', 'gen_TSIT', 'gen_UniControl',
    'gen_VisualCloze', 'gen_Weather_Effect_Generator',
]

DATASETS = {
    'BDD10k': 'bdd10k',
    'MapillaryVistas': 'mapillaryvistas',
    'OUTSIDE15k': 'outside15k',
}

MODEL = 'segformer_mit-b3'

# Mapping from dataset_dir name to dataset configs (for num_classes detection)
DATASET_CONFIGS = {
    'BDD10k': {'num_classes': 19},
    'MapillaryVistas': {'num_classes': 66},
    'OUTSIDE15k': {'num_classes': 24},
}


def find_checkpoint(strategy, dataset_dir, weights_root):
    """Find iter_15000 checkpoint, trying both model dir naming conventions."""
    # Try direct model dir
    ckpt = weights_root / strategy / dataset_dir / MODEL / f'{CHECKPOINT_ITER}.pth'
    config = weights_root / strategy / dataset_dir / MODEL / 'training_config.py'
    if ckpt.exists() and config.exists():
        return str(ckpt), str(config), MODEL

    # Try ratio0p50 naming
    model_ratio = f'{MODEL}_ratio0p50'
    ckpt = weights_root / strategy / dataset_dir / model_ratio / f'{CHECKPOINT_ITER}.pth'
    config = weights_root / strategy / dataset_dir / model_ratio / 'training_config.py'
    if ckpt.exists() and config.exists():
        return str(ckpt), str(config), model_ratio

    return None, None, None


def load_testing_manifest():
    """Load the testing manifest."""
    manifest_path = OUTPUT_ROOT / 'metadata' / 'testing_manifest.json'
    if not manifest_path.exists():
        print(f"ERROR: Testing manifest not found at {manifest_path}")
        sys.exit(1)
    with open(manifest_path) as f:
        return json.load(f)


def identify_gaps():
    """Identify all missing prediction directories."""
    gaps = []

    for dataset_name, dataset_dir in DATASETS.items():
        for stage_name, weights_root in [('stage1', WEIGHTS_STAGE1), ('stage2', WEIGHTS_STAGE2)]:
            for strategy in ALL_STRATEGIES:
                # Check if prediction already exists
                pred_dir = OUTPUT_ROOT / 'testing_samples' / dataset_name / 'predictions' / stage_name / strategy
                # Look for any model subdir with inference_results.json
                has_prediction = False
                if pred_dir.exists():
                    for model_dir in pred_dir.iterdir():
                        if (model_dir / 'inference_results.json').exists():
                            has_prediction = True
                            break

                if has_prediction:
                    continue

                # Try to find the checkpoint
                ckpt_path, config_path, model_dir = find_checkpoint(strategy, dataset_dir, weights_root)
                if ckpt_path is None:
                    gaps.append({
                        'dataset': dataset_name,
                        'stage': stage_name,
                        'strategy': strategy,
                        'status': 'NO_CHECKPOINT',
                    })
                else:
                    target_pred_dir = str(OUTPUT_ROOT / 'testing_samples' / dataset_name / 'predictions' / stage_name / strategy / model_dir)
                    gaps.append({
                        'dataset': dataset_name,
                        'stage': stage_name,
                        'strategy': strategy,
                        'status': 'FILLABLE',
                        'checkpoint_path': ckpt_path,
                        'config_path': config_path,
                        'model_dir': model_dir,
                        'pred_dir': target_pred_dir,
                    })

    return gaps


def generate_batch_script(dataset_name, stage, ckpt_batch, test_images):
    """Generate an inference script for a batch of checkpoints."""
    script = f'''#!/usr/bin/env python3
"""Auto-generated fill-gap inference: {stage} / {dataset_name} ({len(ckpt_batch)} models)"""
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
    parser = argparse.ArgumentParser(description='Fill missing prediction gaps for publication figures')
    parser.add_argument('--dry-run', action='store_true', help='Show gaps without submitting')
    parser.add_argument('--submit', action='store_true', help='Submit inference jobs')
    parser.add_argument('-y', action='store_true', help='Skip confirmation')
    args = parser.parse_args()

    if not args.dry_run and not args.submit:
        print("Specify --dry-run or --submit")
        sys.exit(1)

    print("Scanning for missing predictions...")
    gaps = identify_gaps()

    fillable = [g for g in gaps if g['status'] == 'FILLABLE']
    unfillable = [g for g in gaps if g['status'] == 'NO_CHECKPOINT']

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(fillable)} fillable gaps, {len(unfillable)} unfillable (no checkpoint)")
    print(f"{'='*60}")

    if unfillable:
        print(f"\nUnfillable ({len(unfillable)} — checkpoint doesn't exist yet):")
        for g in sorted(unfillable, key=lambda x: (x['stage'], x['dataset'], x['strategy'])):
            print(f"  {g['stage']:6s} {g['dataset']:20s} {g['strategy']}")

    if fillable:
        print(f"\nFillable ({len(fillable)} — will submit inference jobs):")
        for g in sorted(fillable, key=lambda x: (x['stage'], x['dataset'], x['strategy'])):
            print(f"  {g['stage']:6s} {g['dataset']:20s} {g['strategy']:40s} → {g['model_dir']}")

    if args.dry_run or not fillable:
        if not fillable:
            print("\nNo gaps to fill!")
        return

    # Load testing manifest for test image paths
    manifest = load_testing_manifest()

    # Group fillable gaps by (stage, dataset)
    grouped = defaultdict(list)
    for g in fillable:
        grouped[(g['stage'], g['dataset'])].append({
            'strategy': g['strategy'],
            'model_dir': g['model_dir'],
            'config_path': g['config_path'],
            'checkpoint_path': g['checkpoint_path'],
            'pred_dir': g['pred_dir'],
        })

    # Generate and submit jobs
    scripts_dir = OUTPUT_ROOT / 'inference_scripts'
    logs_dir = OUTPUT_ROOT / 'logs'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.y:
        response = input(f"\nSubmit {len(grouped)} inference jobs ({len(fillable)} model inferences)? [y/N] ")
        if response.lower() != 'y':
            print("Aborted")
            return

    submitted = 0
    for (stage, dataset), ckpt_batch in sorted(grouped.items()):
        test_images = manifest.get(dataset, {}).get('images', [])
        if not test_images:
            print(f"  WARNING: No test images for {dataset}, skipping")
            continue

        script_content = generate_batch_script(dataset, stage, ckpt_batch, test_images)

        script_name = f"fill_{stage}_{dataset}.py"
        script_path = scripts_dir / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)

        wall_time = max(30, len(ckpt_batch) * 5)  # 5 min per model
        wall_hours = wall_time // 60
        wall_mins = wall_time % 60

        job_name = f"fill_{stage[:2]}_{dataset[:5]}"
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
        result = subprocess.run(['bsub'], input=job_script, capture_output=True, text=True)
        if result.returncode == 0:
            submitted += 1
            strategies_list = [c['strategy'] for c in ckpt_batch]
            print(f"  [{submitted}] {job_name}: {len(ckpt_batch)} models — {', '.join(strategies_list[:5])}")
            if len(strategies_list) > 5:
                print(f"      ... and {len(strategies_list)-5} more")
        else:
            print(f"  ERROR submitting {job_name}: {result.stderr}")

    print(f"\nSubmitted {submitted} jobs")
    print("Monitor with: bjobs -w | grep fill_")


if __name__ == '__main__':
    main()

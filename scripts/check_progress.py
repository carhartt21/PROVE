#!/usr/bin/env python3
"""Quick progress checker for all stages."""
import os, glob, json, sys
from collections import defaultdict

def check_stage(base_dir, target_iter, label):
    if not os.path.isdir(base_dir):
        print(f"\n{label}: Directory not found")
        return
    
    # Find all model dirs (strategy/dataset/model)
    model_dirs = []
    for strategy in sorted(os.listdir(base_dir)):
        s_path = os.path.join(base_dir, strategy)
        if not os.path.isdir(s_path):
            continue
        for dataset in sorted(os.listdir(s_path)):
            d_path = os.path.join(s_path, dataset)
            if not os.path.isdir(d_path):
                continue
            for model in sorted(os.listdir(d_path)):
                m_path = os.path.join(d_path, model)
                if not os.path.isdir(m_path):
                    continue
                model_dirs.append((strategy, dataset, model, m_path))
    
    complete = 0
    partial = 0
    empty = 0
    progress_dist = defaultdict(int)
    
    for strategy, dataset, model, m_path in model_dirs:
        target_ckpt = os.path.join(m_path, f'iter_{target_iter}.pth')
        if os.path.exists(target_ckpt):
            complete += 1
            progress_dist[target_iter] += 1
        else:
            iters = glob.glob(os.path.join(m_path, 'iter_*.pth'))
            if iters:
                partial += 1
                max_iter = max(int(os.path.basename(f).replace('iter_', '').replace('.pth', '')) for f in iters)
                progress_dist[max_iter] += 1
            else:
                empty += 1
    
    total = len(model_dirs)
    print(f"\n{'='*60}")
    print(f"{label} (target: {target_iter} iters)")
    print(f"{'='*60}")
    print(f"  Total model dirs:  {total}")
    print(f"  ✅ Complete:        {complete} ({100*complete/total:.1f}%)" if total else "")
    print(f"  🔄 Partial:         {partial}")
    print(f"  ⏳ Empty:           {empty}")
    
    # Check valid test results
    cs_results = glob.glob(os.path.join(base_dir, '**/test_results_detailed/**/results.json'), recursive=True)
    valid_tests = 0
    for r in cs_results:
        try:
            with open(r) as f:
                d = json.load(f)
            miou = d.get('overall', {}).get('mIoU', None)
            if miou and float(miou) > 0:
                valid_tests += 1
        except:
            pass
    print(f"  📊 Valid test results: {valid_tests}")
    
    # Show progress distribution for top iterations
    if progress_dist:
        print(f"\n  Progress distribution:")
        for k in sorted(progress_dist.keys(), reverse=True)[:10]:
            bar = '█' * min(progress_dist[k], 40)
            print(f"    iter {k:>6}: {progress_dist[k]:>3} {bar}")

# Check all stages
check_stage('/scratch/aaa_exchange/AWARE/WEIGHTS', 80000, 'STAGE 1 (Clear Day)')
check_stage('/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2', 80000, 'STAGE 2 (All Domains)')
check_stage('/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES_GEN', 20000, 'CITYSCAPES-GEN')

#!/usr/bin/env python3
"""
Generate LSF job scripts for re-testing with CORRECT cluster settings.

Based on existing retrain_jobs scripts:
- Queue: BatchGPU
- GPU: "num=1:mode=exclusive_process:gmem=20G"
- CPUs: 4 (less than training)
- Memory: 16GB
- Time: 2:00 (testing is faster than training)
- Environment: mamba activate prove
"""

import os
import pandas as pd
from pathlib import Path

# LSF Template matching cluster settings
LSF_TEMPLATE = '''#!/bin/bash
#BSUB -J {job_name}
#BSUB -o /home/mima2416/repositories/PROVE/logs/retest/{job_name}_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retest/{job_name}_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 2:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
mamba activate prove

cd /home/mima2416/repositories/PROVE

echo "========================================"
echo "Re-test job: {job_name}"
echo "Strategy: {strategy}"
echo "Dataset: {dataset}"
echo "Model: {model_name}"
echo "Started: $(date)"
echo "========================================"

# Run fine-grained test with batch processing
python fine_grained_test.py \\
    --config "{config_path}" \\
    --checkpoint "{checkpoint_path}" \\
    --output-dir "{output_dir}" \\
    --dataset {dataset_class} \\
    --data-root /scratch/aaa_exchange/AWARE/FINAL_SPLITS \\
    --batch-size 8

echo "========================================"
echo "Test completed at $(date)"
echo "========================================"
'''


def get_dataset_class(dataset_name):
    """Map dataset directory name to test folder name.
    
    These MUST match the actual folder names in:
    /scratch/aaa_exchange/AWARE/FINAL_SPLITS/test/images/
    """
    mapping = {
        'bdd10k': 'BDD10k',
        'bdd10k_cd': 'BDD10k',
        'idd-aw': 'IDD-AW',
        'idd-aw_cd': 'IDD-AW',
        'mapillaryvistas': 'MapillaryVistas',
        'mapillaryvistas_cd': 'MapillaryVistas',
        'outside15k': 'OUTSIDE15k',
        'outside15k_cd': 'OUTSIDE15k',
    }
    return mapping.get(dataset_name.lower(), dataset_name)


def generate_lsf_scripts():
    """Generate LSF scripts for all tests that need re-running."""
    
    df = pd.read_csv('/home/mima2416/repositories/PROVE/downstream_results.csv')
    
    # Filter to tests that need re-running (mIoU <= 10%)
    bad_tests = df[df['mIoU'] <= 10.0].copy()
    
    output_dir = Path('/home/mima2416/repositories/PROVE/scripts/retest_jobs_lsf')
    
    # Create logs directory
    logs_dir = Path('/home/mima2416/repositories/PROVE/logs/retest')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear old scripts
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track jobs per dataset
    jobs_per_dataset = {}
    
    for idx, row in bad_tests.iterrows():
        dataset = row['dataset']
        strategy = row['strategy']
        model = row['model']
        result_dir = row['result_dir']
        
        # Derive paths from result_dir
        # e.g., /scratch/.../WEIGHTS/strategy/dataset/model/test_results_detailed/timestamp
        # model_dir is parent of test_results_detailed
        parts = result_dir.split('/')
        # Find "test_results" in path to determine model_dir
        for i, p in enumerate(parts):
            if 'test_results' in p:
                model_dir = '/'.join(parts[:i])
                break
        else:
            model_dir = '/'.join(parts[:-2])  # fallback: 2 levels up
        
        config_path = f"{model_dir}/training_config.py"
        checkpoint_path = f"{model_dir}/iter_80000.pth"
        
        # Create job name (truncate to 50 chars for LSF)
        job_name = f"retest_{strategy}_{dataset}_{model}"[:50]
        
        # Output directory for new results
        new_output_dir = f"{model_dir}/test_results_detailed_fixed"
        
        # Verify checkpoint exists before generating script
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue
        
        # Generate script content
        script_content = LSF_TEMPLATE.format(
            job_name=job_name,
            strategy=strategy,
            dataset=dataset,
            model_name=model,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=new_output_dir,
            dataset_class=get_dataset_class(dataset)
        )
        
        # Create dataset subdirectory
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Write script
        script_path = dataset_dir / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        # Track
        if dataset not in jobs_per_dataset:
            jobs_per_dataset[dataset] = []
        jobs_per_dataset[dataset].append(script_path)
    
    # Create submission scripts for each dataset
    for dataset, scripts in jobs_per_dataset.items():
        submit_script = output_dir / f'submit_{dataset}_retests.sh'
        with open(submit_script, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Submit all {dataset} re-test jobs using LSF\n\n')
            f.write(f'echo "Submitting {len(scripts)} jobs for {dataset}..."\n\n')
            for script in sorted(scripts):
                f.write(f'bsub < {script}\n')
                f.write('sleep 1\n')
            f.write(f'\necho "Done submitting {len(scripts)} jobs for {dataset}"\n')
        os.chmod(submit_script, 0o755)
    
    # Create master submission script
    submit_all = output_dir / 'submit_all_retests.sh'
    with open(submit_all, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Submit all re-test jobs using LSF\n')
        f.write('# Total: 170 jobs\n\n')
        f.write('cd /home/mima2416/repositories/PROVE/scripts/retest_jobs_lsf\n\n')
        for dataset in sorted(jobs_per_dataset.keys()):
            f.write(f'echo "=== Submitting {dataset} jobs ({len(jobs_per_dataset[dataset])}) ==="\n')
            f.write(f'bash submit_{dataset}_retests.sh\n')
            f.write('echo ""\n\n')
        f.write('echo "All jobs submitted!"\n')
        f.write('echo "Monitor with: bjobs -a"\n')
    os.chmod(submit_all, 0o755)
    
    # Summary
    print("=" * 80)
    print("LSF SCRIPTS GENERATED")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nJobs per dataset:")
    total = 0
    for dataset in sorted(jobs_per_dataset.keys()):
        count = len(jobs_per_dataset[dataset])
        total += count
        print(f"  {dataset:25s}: {count} jobs")
    print(f"  {'TOTAL':25s}: {total} jobs")
    
    print("\nSubmission commands:")
    print(f"  cd {output_dir}")
    print("  bash submit_all_retests.sh           # Submit all jobs")
    print("  bash submit_bdd10k_cd_retests.sh     # Submit only BDD10K_CD")
    
    print("\nMonitoring commands:")
    print("  bjobs -a         # View all jobs")
    print("  bjobs -w         # Wide format")
    print("  bkill <jobid>    # Kill a job")
    print("  bkill 0          # Kill all your jobs")
    
    # Show example script
    print("\n" + "=" * 80)
    print("Example LSF script:")
    print("=" * 80)
    example_scripts = list((output_dir / 'bdd10k_cd').glob('*baseline*.sh'))
    if example_scripts:
        with open(example_scripts[0]) as f:
            print(f.read())


if __name__ == '__main__':
    generate_lsf_scripts()

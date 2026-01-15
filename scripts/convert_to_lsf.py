#!/usr/bin/env python3
"""
Convert SLURM job scripts to LSF format.

SLURM -> LSF mapping:
- #SBATCH --job-name=X     -> #BSUB -J X
- #SBATCH --output=X       -> #BSUB -o X
- #SBATCH --error=X        -> #BSUB -e X
- #SBATCH --partition=gpu  -> #BSUB -q gpu
- #SBATCH --gres=gpu:1     -> #BSUB -R "rusage[ngpus_excl_p=1]"
- #SBATCH --cpus-per-task=4 -> #BSUB -n 4
- #SBATCH --mem=32G        -> #BSUB -R "rusage[mem=32000]"
- #SBATCH --time=4:00:00   -> #BSUB -W 4:00
"""

import os
import re
from pathlib import Path

def convert_slurm_to_lsf(slurm_content):
    """Convert SLURM script content to LSF format."""
    
    lines = slurm_content.split('\n')
    lsf_lines = []
    
    for line in lines:
        # Shebang
        if line.startswith('#!/bin/bash'):
            lsf_lines.append(line)
            continue
            
        # Job name
        if '#SBATCH --job-name=' in line:
            job_name = line.split('=')[1]
            lsf_lines.append(f'#BSUB -J {job_name}')
            continue
            
        # Output file - LSF uses %J for job ID (SLURM uses %j)
        if '#SBATCH --output=' in line:
            output = line.split('=')[1].replace('%j', '%J')
            lsf_lines.append(f'#BSUB -o {output}')
            continue
            
        # Error file
        if '#SBATCH --error=' in line:
            error = line.split('=')[1].replace('%j', '%J')
            lsf_lines.append(f'#BSUB -e {error}')
            continue
            
        # Partition/Queue
        if '#SBATCH --partition=' in line:
            queue = line.split('=')[1]
            lsf_lines.append(f'#BSUB -q {queue}')
            continue
            
        # GPU resources
        if '#SBATCH --gres=gpu:' in line:
            num_gpus = line.split(':')[-1]
            lsf_lines.append(f'#BSUB -R "rusage[ngpus_excl_p={num_gpus}]"')
            continue
            
        # CPUs
        if '#SBATCH --cpus-per-task=' in line:
            cpus = line.split('=')[1]
            lsf_lines.append(f'#BSUB -n {cpus}')
            continue
            
        # Memory - convert from G to MB
        if '#SBATCH --mem=' in line:
            mem = line.split('=')[1]
            if 'G' in mem:
                mem_mb = int(mem.replace('G', '')) * 1000
            else:
                mem_mb = int(mem.replace('M', ''))
            lsf_lines.append(f'#BSUB -R "rusage[mem={mem_mb}]"')
            continue
            
        # Time limit - convert HH:MM:SS to HH:MM
        if '#SBATCH --time=' in line:
            time_str = line.split('=')[1]
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, mins, _ = parts
                lsf_lines.append(f'#BSUB -W {hours}:{mins}')
            else:
                lsf_lines.append(f'#BSUB -W {time_str}')
            continue
            
        # Skip any other SBATCH lines
        if line.startswith('#SBATCH'):
            continue
            
        # Keep everything else
        lsf_lines.append(line)
    
    return '\n'.join(lsf_lines)


def convert_directory(source_dir, output_dir):
    """Convert all SLURM scripts in a directory to LSF format."""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    
    for slurm_file in source_path.rglob('*.sh'):
        # Read SLURM script
        with open(slurm_file, 'r') as f:
            slurm_content = f.read()
        
        # Skip if it's a submission script (not a job script)
        if 'submit_' in slurm_file.name:
            continue
            
        # Convert to LSF
        lsf_content = convert_slurm_to_lsf(slurm_content)
        
        # Determine output path
        relative_path = slurm_file.relative_to(source_path)
        lsf_file = output_path / relative_path
        lsf_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write LSF script
        with open(lsf_file, 'w') as f:
            f.write(lsf_content)
        
        # Make executable
        os.chmod(lsf_file, 0o755)
        converted_count += 1
        
    return converted_count


def create_submission_scripts(output_dir):
    """Create bsub submission scripts for each dataset."""
    
    output_path = Path(output_dir)
    
    datasets = ['bdd10k_cd', 'idd-aw_cd', 'mapillaryvistas_cd', 'outside15k_cd']
    
    for dataset in datasets:
        dataset_dir = output_path / dataset
        if not dataset_dir.exists():
            continue
            
        scripts = sorted(dataset_dir.glob('*.sh'))
        
        # Create per-dataset submission script
        submit_script = output_path / f'submit_{dataset}_retests.sh'
        with open(submit_script, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Submit all {dataset} re-test jobs using LSF\n\n')
            f.write(f'echo "Submitting {len(scripts)} jobs for {dataset}..."\n\n')
            for script in scripts:
                f.write(f'bsub < {script}\n')
                f.write('sleep 1\n')
            f.write(f'\necho "Done submitting {len(scripts)} jobs for {dataset}"\n')
        os.chmod(submit_script, 0o755)
    
    # Create master submission script
    submit_all = output_path / 'submit_all_retests.sh'
    with open(submit_all, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Submit all re-test jobs using LSF\n\n')
        for dataset in datasets:
            script = output_path / f'submit_{dataset}_retests.sh'
            if script.exists():
                f.write(f'echo "=== Submitting {dataset} jobs ==="\n')
                f.write(f'bash {script}\n')
                f.write('echo ""\n\n')
        f.write('echo "All jobs submitted!"\n')
    os.chmod(submit_all, 0o755)


def main():
    source_dir = '/home/mima2416/repositories/PROVE/scripts/retest_jobs'
    output_dir = '/home/mima2416/repositories/PROVE/scripts/retest_jobs_lsf'
    
    print("Converting SLURM scripts to LSF format...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    
    count = convert_directory(source_dir, output_dir)
    print(f"  Converted {count} job scripts")
    
    print("\nCreating submission scripts...")
    create_submission_scripts(output_dir)
    
    print("\nDone! LSF scripts are in:", output_dir)
    print("\nTo submit jobs:")
    print(f"  cd {output_dir}")
    print("  bash submit_all_retests.sh           # Submit all 170 jobs")
    print("  bash submit_bdd10k_cd_retests.sh     # Submit only BDD10K_CD jobs")
    
    # Show example LSF script
    print("\n" + "=" * 80)
    print("Example LSF script:")
    print("=" * 80)
    example = Path(output_dir) / 'bdd10k_cd' / 'retest_baseline_bdd10k_cd_deeplabv3plus_r50.sh'
    if example.exists():
        with open(example) as f:
            print(f.read())


if __name__ == '__main__':
    main()

# Retraining Job Scripts

This directory contains auto-generated LSF job scripts for retraining models.

## ⚠️ Important Notes

**These scripts are user-specific and should NOT be committed to version control.**

The scripts contain hardcoded paths that are automatically detected based on where the PROVE repository is located on your system.

## Generating Scripts

To generate or regenerate the job scripts for your environment:

```bash
cd /path/to/PROVE
python scripts/retrain_affected_models.py --generate-scripts
```

This will create job scripts with correct paths for your system.

## Usage

After generating scripts, you can:

```bash
# Submit all jobs
python scripts/retrain_affected_models.py --submit-all

# Submit only pending jobs (skips completed)
python scripts/retrain_affected_models.py --submit-pending

# Submit specific strategy
python scripts/retrain_affected_models.py --submit-strategy baseline

# Submit specific strategy × dataset
python scripts/retrain_affected_models.py --submit-strategy baseline --dataset bdd10k
```

## Configuration

You can override default paths:

```bash
python scripts/retrain_affected_models.py \
    --project-root /custom/path/to/PROVE \
    --weights-root /custom/weights/path \
    --generate-scripts
```

Or use environment variables:

```bash
export PROVE_WEIGHTS_ROOT=/custom/weights/path
python scripts/retrain_affected_models.py --generate-scripts
```

# Training Lock Management

## Overview

The training lock system prevents duplicate training of the same model configuration when running multiple jobs in parallel. This is especially important on HPC clusters where multiple jobs might be submitted for the same configuration.

## Lock Mechanisms

PROVE uses **two different locking mechanisms**:

### 1. File-Based Locks (`.training_lock`)

Used by the retrain shell scripts (e.g., `scripts/retrain_jobs/*.sh`). These locks are:
- Created in each model's weights directory (e.g., `${AWARE_DATA_ROOT}/WEIGHTS/{strategy}/{dataset}_cd/{model}/.training_lock`)
- Contain the PID of the training process
- Automatically removed when training completes (via `trap` in bash)

### 2. fcntl-Based Locks (`TrainingLock` class)

A Python class using `fcntl` file locks, stored in `${AWARE_DATA_ROOT}/training_locks/`. These locks are:
- Process-safe (kernel-level locking)
- Automatically released when the process exits
- Include metadata (job ID, hostname, timestamp)

## Command-Line Interface

### List Active Locks

```bash
# List all active locks (both types)
python training_lock.py list

# List only file-based locks
python training_lock.py list --file-only

# List only fcntl-based locks
python training_lock.py list --fcntl-only

# Skip PID verification (faster)
python training_lock.py list --no-verify-pid
```

**Example output:**
```
=== File-based locks (.training_lock) (3) ===
  - gen_VisualCloze/idd-aw/segformer_mit-b5_ratio0p50 [UNKNOWN - remote process]
    PID: 2233310, Started: 2026-01-12T16:49:29.536651
    Lock file: ${AWARE_DATA_ROOT}/WEIGHTS/gen_VisualCloze/idd-aw_cd/segformer_mit-b5_ratio0p50/.training_lock
  - photometric_distort/idd-aw/deeplabv3plus_r50 [ACTIVE]
    PID: 1277413, Started: 2026-01-12T16:52:55.136674
    Lock file: ${AWARE_DATA_ROOT}/WEIGHTS/photometric_distort/idd-aw_cd/deeplabv3plus_r50/.training_lock

Total locks: 3
```

**Lock Status Indicators:**
- `[ACTIVE]` - PID is running locally
- `[STALE - PID not running]` - PID confirmed not running (safe to clear)
- `[STALE - no PID]` - Lock file exists but has no PID (safe to clear)
- `[UNKNOWN - remote process]` - PID is from a compute node; can't verify from login node

### Clear Stale Locks

```bash
# Dry run - show what would be cleared
python training_lock.py clear

# Clear only file-based stale locks
python training_lock.py clear --file-only

# Actually remove stale locks
python training_lock.py clear --no-dry-run

# Force clear (including unknown status locks - USE WITH CAUTION!)
python training_lock.py clear --force --no-dry-run
```

**Warning:** Use `--force` only when you're certain no jobs are running, as it will remove locks even for processes that might be running on compute nodes.

### Pre-flight Check

Check if a configuration is locked before starting training:

```bash
python training_lock.py check --strategy baseline --dataset IDD-AW --model deeplabv3plus_r50
```

Returns exit code 0 if safe to proceed, 1 if locked.

## Usage in Training Scripts

### Shell Script Method (Recommended for LSF jobs)

```bash
#!/bin/bash
WEIGHTS_PATH="${AWARE_DATA_ROOT}/WEIGHTS/{strategy}/{dataset}_cd/{model}"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"

# Create lock directory
mkdir -p "$WEIGHTS_PATH"

# Try to acquire lock (atomic operation)
if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
    trap "rm -f '$LOCK_FILE'" EXIT
    echo "Lock acquired. Starting training..."
    
    # Training code here
    python unified_training.py ...
    
    rm -f "$LOCK_FILE"
else
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
fi
```

### Python Method (Using TrainingLock class)

```python
from training_lock import TrainingLock

lock = TrainingLock('baseline', 'IDD-AW', 'deeplabv3plus_r50')

if lock.acquire():
    try:
        # Training code here
        pass
    finally:
        lock.release()
else:
    print("Configuration is locked by another process")
```

Or using context manager:

```python
from training_lock import TrainingLock

with TrainingLock('baseline', 'IDD-AW', 'deeplabv3plus_r50') as lock:
    # Training code here
    pass  # Lock automatically released on exit
```

## Troubleshooting

### "No active training locks" but jobs are running

This usually means:
1. The lock directory doesn't exist yet (will be created when first lock is acquired)
2. The training jobs use file-based locks in the weights directory, not fcntl locks

Run `python training_lock.py list` to see both types of locks.

### Stale locks after job failure

If a job crashes without cleaning up its lock:
1. Check if the job is actually running: `bjobs -w`
2. View locks: `python training_lock.py list`
3. If confirmed stale: `python training_lock.py clear --no-dry-run`

### Locks show as "UNKNOWN - remote process"

This happens when:
- The lock was created by a compute node (e.g., makalu94)
- You're checking from a login node (e.g., makalu48)
- SSH verification to compute node failed/timed out

These are likely active locks from running jobs. Use `bjobs -w` to verify if jobs are running.

## API Reference

### `TrainingLock` Class

```python
TrainingLock(
    strategy: str,          # Training strategy name
    dataset: str,           # Dataset name
    model: str,             # Model name
    ratio: float = None,    # Optional ratio for generative strategies
    lock_dir: str = DEFAULT_LOCK_DIR
)
```

Methods:
- `acquire(blocking=False, timeout=60)` - Acquire the lock
- `release()` - Release the lock
- `is_locked()` - Check if locked by another process
- `get_lock_holder()` - Get info about current lock holder

### Utility Functions

```python
from training_lock import (
    list_active_locks,       # List fcntl-based locks
    list_weights_dir_locks,  # List file-based locks  
    list_all_locks,          # List both types
    clear_stale_locks,       # Clear stale fcntl locks
    clear_stale_file_locks,  # Clear stale file locks
    preflight_check,         # Pre-flight safety check
    is_pid_running,          # Check if PID is running
)
```

#!/usr/bin/env python3
"""
Training lock mechanism to prevent parallel training of the same configuration.

This module provides a file-based locking mechanism to prevent multiple jobs
from training the same model configuration simultaneously.

Usage:
    from training_lock import TrainingLock
    
    lock = TrainingLock(strategy='baseline', dataset='IDD-AW', model='deeplabv3plus_r50')
    if lock.acquire():
        try:
            # Do training
            pass
        finally:
            lock.release()
    else:
        print("Another job is already training this configuration")
"""

import os
import sys
import fcntl
import time
import json
import socket
import atexit
from pathlib import Path
from datetime import datetime
from typing import Optional, Union


# Default lock directory
DEFAULT_LOCK_DIR = '${AWARE_DATA_ROOT}/training_locks'


class TrainingLock:
    """File-based lock for training configurations."""
    
    def __init__(self, strategy: str, dataset: str, model: str, 
                 ratio: Optional[float] = None,
                 aux_loss: Optional[str] = None,
                 stage: Optional[Union[int, str]] = None,
                 lock_dir: str = DEFAULT_LOCK_DIR):
        """
        Initialize a training lock.
        
        Args:
            strategy: Training strategy (e.g., 'baseline', 'gen_CUT')
            dataset: Dataset name (e.g., 'IDD-AW', 'BDD10k')
            model: Model name (e.g., 'deeplabv3plus_r50')
            ratio: Optional ratio for generative strategies
            aux_loss: Optional auxiliary loss
            stage: Training stage (1, 2, or string like 'cityscapes-gen') for lock file naming
            lock_dir: Directory to store lock files
        """
        self.strategy = strategy
        self.dataset = dataset.lower().replace('-', '_')
        self.model = model
        self.ratio = ratio
        self.aux_loss = aux_loss
        self.stage = stage
        self.lock_dir = Path(lock_dir)
        
        # Build lock filename with stage prefix
        # Integer stages get 's{N}_' prefix, string stages use themselves as prefix
        if stage is None:
            stage_prefix = ''
        elif isinstance(stage, int):
            stage_prefix = f's{stage}_'
        else:
            stage_prefix = f'{stage}_'
        lock_name = f'{stage_prefix}{strategy}_{self.dataset}_{model}'
        if ratio is not None:
            lock_name += f'_ratio{ratio:.2f}'.replace('.', 'p')
        if aux_loss:
            lock_name += f'_aux-{aux_loss}'
        self.lock_file = self.lock_dir / f'{lock_name}.lock'
        
        self._file_handle = None
        self._acquired = False
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
    
    def _get_lock_info(self) -> dict:
        """Generate lock info metadata."""
        return {
            'strategy': self.strategy,
            'dataset': self.dataset,
            'model': self.model,
            'ratio': self.ratio,
            'aux_loss': self.aux_loss,
            'job_id': os.environ.get('LSB_JOBID', 'unknown'),
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'user': os.environ.get('USER', 'unknown'),
            'acquired_at': datetime.now().isoformat(),
        }
    
    def acquire(self, blocking: bool = False, timeout: int = 60) -> bool:
        """
        Acquire the training lock.
        
        Args:
            blocking: If True, wait for lock. If False, fail immediately if locked.
            timeout: Timeout in seconds when blocking (default: 60)
            
        Returns:
            True if lock acquired, False otherwise
        """
        if self._acquired:
            return True
        
        # Ensure lock directory exists
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Open/create lock file
            self._file_handle = open(self.lock_file, 'w')
            
            if blocking:
                # Try to acquire with timeout
                start_time = time.time()
                while True:
                    try:
                        fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except IOError:
                        if time.time() - start_time > timeout:
                            self._file_handle.close()
                            self._file_handle = None
                            return False
                        time.sleep(1)
            else:
                # Non-blocking acquire
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write lock info
            lock_info = self._get_lock_info()
            self._file_handle.write(json.dumps(lock_info, indent=2))
            self._file_handle.flush()
            
            self._acquired = True
            return True
            
        except IOError:
            # Lock is held by another process
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
            return False
        except Exception as e:
            print(f"Error acquiring lock: {e}", file=sys.stderr)
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
            return False
    
    def release(self):
        """Release the training lock."""
        if not self._acquired:
            return
        
        try:
            if self._file_handle:
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
                self._file_handle.close()
                self._file_handle = None
            
            # Remove lock file
            if self.lock_file.exists():
                self.lock_file.unlink()
                
            self._acquired = False
        except Exception as e:
            print(f"Error releasing lock: {e}", file=sys.stderr)
    
    def _cleanup(self):
        """Cleanup on exit."""
        if self._acquired:
            self.release()
    
    def is_locked(self) -> bool:
        """
        Check if the configuration is currently locked by another process.
        
        Returns:
            True if locked by another process, False otherwise
        """
        if not self.lock_file.exists():
            return False
        
        try:
            with open(self.lock_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            # Lock was available, not locked
            return False
        except IOError:
            # Lock is held
            return True
    
    def get_lock_holder(self) -> Optional[dict]:
        """
        Get information about the process holding the lock.
        
        Returns:
            Lock info dict if locked, None if not locked
        """
        if not self.lock_file.exists():
            return None
        
        try:
            with open(self.lock_file, 'r') as f:
                content = f.read()
                if content:
                    return json.loads(content)
        except (json.JSONDecodeError, IOError):
            pass
        return None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            holder = self.get_lock_holder()
            if holder:
                raise RuntimeError(
                    f"Configuration already being trained by job {holder.get('job_id')} "
                    f"on {holder.get('hostname')} since {holder.get('acquired_at')}"
                )
            raise RuntimeError("Configuration is locked by another process")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def check_training_lock(strategy: str, dataset: str, model: str, 
                        ratio: Optional[float] = None,
                        aux_loss: Optional[str] = None) -> bool:
    """
    Convenience function to check if a training configuration is locked.
    
    Returns:
        True if NOT locked (can proceed), False if locked (should skip)
    """
    lock = TrainingLock(strategy, dataset, model, ratio, aux_loss=aux_loss)
    return not lock.is_locked()


def preflight_check(strategy: str, dataset: str, model: str,
                   ratio: Optional[float] = None,
                   aux_loss: Optional[str] = None) -> bool:
    """
    Pre-flight check before starting training.
    
    Checks if another job is already training this configuration.
    
    Args:
        strategy: Training strategy
        dataset: Dataset name  
        model: Model name
        ratio: Optional ratio for generative strategies
        
    Returns:
        True if safe to proceed, False if should skip
    """
    lock = TrainingLock(strategy, dataset, model, ratio, aux_loss=aux_loss)
    
    if lock.is_locked():
        holder = lock.get_lock_holder()
        print(f"[PRE-FLIGHT CHECK FAILED]", file=sys.stderr)
        print(f"Configuration is already being trained:", file=sys.stderr)
        print(f"  Strategy: {strategy}", file=sys.stderr)
        print(f"  Dataset: {dataset}", file=sys.stderr)
        print(f"  Model: {model}", file=sys.stderr)
        if ratio:
            print(f"  Ratio: {ratio}", file=sys.stderr)
        if aux_loss:
            print(f"  Aux Loss: {aux_loss}", file=sys.stderr)
        if holder:
            print(f"  Held by: Job {holder.get('job_id')} on {holder.get('hostname')}", file=sys.stderr)
            print(f"  Since: {holder.get('acquired_at')}", file=sys.stderr)
        print(f"Skipping to avoid duplicate training.", file=sys.stderr)
        return False
    
    return True


def list_active_locks(lock_dir: str = DEFAULT_LOCK_DIR) -> list:
    """List all currently active training locks."""
    lock_path = Path(lock_dir)
    if not lock_path.exists():
        return []
    
    active_locks = []
    for lock_file in lock_path.glob('*.lock'):
        try:
            with open(lock_file, 'r') as f:
                try:
                    # Check if actually locked
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    # Not locked, stale lock file
                    continue
                except IOError:
                    # Actually locked
                    pass
                
                f.seek(0)
                content = f.read()
                if content:
                    lock_info = json.loads(content)
                    lock_info['lock_file'] = str(lock_file)
                    active_locks.append(lock_info)
        except Exception:
            pass
    
    return active_locks


def clear_stale_locks(lock_dir: str = DEFAULT_LOCK_DIR, dry_run: bool = True) -> list:
    """
    Clear stale lock files (locks not held by any process).
    
    Args:
        lock_dir: Lock directory
        dry_run: If True, only report stale locks without removing
        
    Returns:
        List of stale lock files
    """
    lock_path = Path(lock_dir)
    if not lock_path.exists():
        return []
    
    stale_locks = []
    for lock_file in lock_path.glob('*.lock'):
        try:
            with open(lock_file, 'r') as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    # Lock is available = stale
                    stale_locks.append(str(lock_file))
                    if not dry_run:
                        lock_file.unlink()
                except IOError:
                    # Lock is held = active
                    pass
        except Exception:
            pass
    
    return stale_locks


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Training lock management')
    parser.add_argument('command', choices=['list', 'clear', 'check'],
                       help='Command to execute')
    parser.add_argument('--strategy', help='Strategy name (for check)')
    parser.add_argument('--dataset', help='Dataset name (for check)')
    parser.add_argument('--model', help='Model name (for check)')
    parser.add_argument('--ratio', type=float, help='Ratio (for check)')
    parser.add_argument('--aux-loss', help='Auxiliary loss (for check)')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='Actually clear stale locks')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        locks = list_active_locks()
        if locks:
            print(f"Active training locks ({len(locks)}):")
            for lock in locks:
                print(f"  - Job {lock.get('job_id')} ({lock.get('strategy')}/{lock.get('dataset')}/{lock.get('model')})")
                print(f"    Host: {lock.get('hostname')}, Started: {lock.get('acquired_at')}")
        else:
            print("No active training locks")
            
    elif args.command == 'clear':
        stale = clear_stale_locks(dry_run=not args.no_dry_run)
        if stale:
            if args.no_dry_run:
                print(f"Cleared {len(stale)} stale locks")
            else:
                print(f"Found {len(stale)} stale locks (dry run):")
                for lock in stale:
                    print(f"  - {lock}")
                print("\nRun with --no-dry-run to remove")
        else:
            print("No stale locks found")
            
    elif args.command == 'check':
        if not all([args.strategy, args.dataset, args.model]):
            print("Error: --strategy, --dataset, and --model required for check")
            sys.exit(1)
        
        can_proceed = preflight_check(
            args.strategy,
            args.dataset,
            args.model,
            args.ratio,
            args.aux_loss,
        )
        sys.exit(0 if can_proceed else 1)

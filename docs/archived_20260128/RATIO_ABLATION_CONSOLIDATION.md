# Ratio Ablation Directory Consolidation Plan

**Created:** 2026-01-26
**Status:** ✅ Fix Applied - Script Updated

## Summary

A bug in `submit_ratio_ablation_training.py` was causing all jobs to write checkpoints to the same root-level directory, resulting in checkpoint collisions.

### Actions Taken

1. ✅ **Fixed `submit_ratio_ablation_training.py`** - Now builds proper nested paths:
   ```
   {WEIGHTS_DIR}/{strategy}/{dataset_lower}/{model}_ratio{ratio}
   ```

2. ✅ **Killed misconfigured jobs** - All 113 pending/running jobs using flat paths were terminated

3. ✅ **Cleaned up root-level files** - Archived timestamp directories, removed loose checkpoints

4. ⚠️ **User chge7185** is still running jobs with the old buggy configuration - cannot control their jobs

### What Was Lost

- Training jobs submitted Jan 25-26 for `gen_Attribute_Hallucination`, `gen_cycleGAN`, `gen_Img2Img`, etc.
- Checkpoints were overwriting each other at root level

### What's Still Valid

| Strategy | Models | Tests |
|----------|--------|-------|
| gen_cyclediffusion | 9 | 9 |
| gen_cycleGAN | 28 | 28 |
| gen_stargan_v2 | 9 | 9 |
| gen_step1x_new | 56 | 28 |
| gen_step1x_v1p2 | 46 | 28 |
| **Total** | **148** | **102** |

# Manifest Generation Status Report

**Date:** 2026-02-05
**Author:** chge7185

## Summary

The manifest generation script (`tools/generate_manifests.py`) has been updated to support Cityscapes images and follow symbolic links. However, permission issues prevent regenerating most manifests.

## Changes Made

### 1. Added Cityscapes to Known Datasets

**File:** `tools/generate_manifests.py` (line ~74)

```python
KNOWN_DATASETS = {'ACDC', 'BDD100k', 'BDD10k', 'Cityscapes', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'}
```

### 2. Created Cityscapes Symlink

Created symlink to include Cityscapes originals in the standard location:

```bash
ln -s ${AWARE_DATA_ROOT}/CLEAR_DAY_TRAIN/Cityscapes \
      ${AWARE_DATA_ROOT}/FINAL_SPLITS/train/images/Cityscapes
```

### 3. Fixed Symlink Following in `find_image_files()`

The original implementation used `pathlib.rglob()` which does not follow symlinks by default. Updated to use `os.walk(followlinks=True)`:

**Before:**
```python
def find_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all supported image files in a directory."""
    image_files = []
    try:
        glob_func = directory.rglob if recursive else directory.glob
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(glob_func(f"*{ext}"))
            image_files.extend(glob_func(f"*{ext.upper()}"))
    except PermissionError:
        pass
    return sorted(image_files)
```

**After:**
```python
def find_image_files(directory: Path, recursive: bool = True, follow_symlinks: bool = True) -> List[Path]:
    """Find all supported image files in a directory.
    
    Args:
        directory: Directory to search in
        recursive: Whether to search recursively in subdirectories
        follow_symlinks: Whether to follow symbolic links (default True)
    """
    image_files = []
    try:
        if recursive and follow_symlinks:
            # os.walk follows symlinks by default, unlike pathlib.rglob
            for root, dirs, files in os.walk(directory, followlinks=True):
                root_path = Path(root)
                for f in files:
                    if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                        image_files.append(root_path / f)
        else:
            glob_func = directory.rglob if recursive else directory.glob
            for ext in SUPPORTED_EXTENSIONS:
                image_files.extend(glob_func(f"*{ext}"))
                image_files.extend(glob_func(f"*{ext.upper()}"))
    except PermissionError:
        pass
    return sorted(image_files)
```

## Results

### Original Image Index
- **Total indexed:** 74,646 images (up from 71,671 before Cityscapes)
- **Cityscapes:** 2,975 images
- **Generation index:** 72,872 unique stems

### Successfully Regenerated

| Method | Total Images | Match Rate | Cityscapes Images |
|--------|-------------|------------|-------------------|
| cycleGAN | 205,248 | 100.0% | 17,850 (2,975 × 6 conditions) |

### Permission Blocked

The following directories have manifest.json files owned by `${USER}` with 600 permissions, preventing updates:

| Directory | Structure | Manifest Owner | Permission |
|-----------|-----------|----------------|------------|
| albumentations_weather | dataset_domain | ${USER} | 600 |
| Attribute_Hallucination | unknown | ${USER} | 600 |
| augmenters | flat_dataset | ${USER} | 600 |
| automold | flat_dataset | ${USER} | 600 |
| CNetSeg | flat_dataset | ${USER} | 600 |
| CUT | domain_dataset | ${USER} | 600 |
| cyclediffusion | dataset_domain | ${USER} | 600 |
| EDICT | flat_domain | ${USER} | 600 |
| flux2 | domain_dataset | ${USER} | 644 |
| flux_kontext | dataset_domain | ${USER} | 600 |
| LANIT | flat_domain | ${USER} | 600 |
| magicbrush | dataset_domain | ${USER} | 600 |
| Qwen-Image-Edit | dataset_domain | ${USER} | 600 |
| stargan_v2 | unknown | ${USER} | 600 |
| step1x_new | dataset_domain | ${USER} | 600 |
| step1x_v1p2 | dataset_domain | ${USER} | 600 |
| StyleID | domain_dataset | ${USER} | 600 |
| SUSTechGAN | dataset_domain | ${USER} | 600 |
| TSIT | flat_dataset | ${USER} | 600 |
| VisualCloze | dataset_domain | ${USER} | 644 |
| visualcloze | dataset_domain | ${USER} | 600 |
| Weather_Effect_Generator | flat_dataset | ${USER} | 600 |

### No Manifest Yet

| Directory | Status |
|-----------|--------|
| qwen | Jobs still running (owned by chge7185) |
| IP2P | No images found |
| Img2Img | No images found |
| UniControl | No images found |

## Resolution Options

### Option 1: Permission Fix (Recommended)
Have `${USER}` run:
```bash
chmod 666 ${AWARE_DATA_ROOT}/GENERATED_IMAGES/*/manifest.json
```

### Option 2: Delete and Regenerate
For directories owned by chge7185, remove existing manifest.json and regenerate:
```bash
rm ${AWARE_DATA_ROOT}/GENERATED_IMAGES/*/manifest.json
python tools/generate_manifests.py --all
```

### Option 3: Run as ${USER}
Have `${USER}` run the regeneration:
```bash
cd /home/chge7185/repositories/PROVE
python tools/generate_manifests.py --all --verbose
```

## Directories with Cityscapes Data

The following directories contain Cityscapes generated images:

```
${AWARE_DATA_ROOT}/GENERATED_IMAGES/
├── albumentations_weather/Cityscapes/
├── Attribute_Hallucination/Cityscapes_HD/
├── augmenters/Cityscapes/
├── automold/Cityscapes/
├── CNetSeg/Cityscapes/
├── CUT/{weather}/Cityscapes/
├── cyclediffusion/Cityscapes/
├── cycleGAN/{weather}/Cityscapes/  ✅ Manifest updated
├── flux_kontext/Cityscapes/
├── magicbrush/Cityscapes/
├── qwen/CITYSCAPES/  (in progress, needs rename to Cityscapes)
├── stargan_v2/Cityscapes_from_lat/Cityscapes/
├── step1x_new/Cityscapes/
├── step1x_v1p2/Cityscapes/
├── SUSTechGAN/Cityscapes/
├── TSIT/Cityscapes/
├── visualcloze/Cityscapes/
└── Weather_Effect_Generator/Cityscapes/
```

## Next Steps

1. Fix permissions on manifest.json files
2. Regenerate all manifests with `--all` flag
3. Rename `qwen/CITYSCAPES` to `qwen/Cityscapes` after jobs complete
4. Generate manifest for `qwen` directory

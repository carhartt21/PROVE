"""
PROVE Configuration

This module provides configurable paths for the prove framework.
Set environment variables or modify this file before running.
"""

import os
from pathlib import Path

# ============================================================================
# Data Paths (Set via environment variables or edit here)
# ============================================================================

# Base data directory
DATA_ROOT = Path(os.environ.get("PROVE_DATA_ROOT", "/path/to/data"))

# Weights directory
WEIGHTS_ROOT = Path(os.environ.get("PROVE_WEIGHTS_ROOT", DATA_ROOT / "weights"))

# Generated images directory (for PRISM)
GENERATED_ROOT = Path(os.environ.get("GENERATED_IMAGES_ROOT", DATA_ROOT / "generated"))

# Original images directory
ORIGINAL_ROOT = Path(os.environ.get("ORIGINAL_IMAGES_ROOT", DATA_ROOT / "originals"))

# ============================================================================
# Per-Dataset Roots (for --dataset-layout standard)
# ============================================================================
# Set these to point directly to each dataset's root directory.
# Only needed when using --dataset-layout standard.

MAPILLARY_ROOT = Path(os.environ.get("PROVE_MAPILLARY_ROOT", DATA_ROOT / "mapillary_vistas"))
CITYSCAPES_ROOT = Path(os.environ.get("PROVE_CITYSCAPES_ROOT", DATA_ROOT / "cityscapes"))
BDD_ROOT = Path(os.environ.get("PROVE_BDD_ROOT", DATA_ROOT / "bdd100k"))
ACDC_ROOT = Path(os.environ.get("PROVE_ACDC_ROOT", DATA_ROOT / "ACDC"))

# ============================================================================
# Example Environment Setup
# ============================================================================
# 
# Add to your ~/.bashrc or run before executing:
#
#   export PROVE_DATA_ROOT="/path/to/your/data"
#   export PROVE_WEIGHTS_ROOT="/path/to/weights"
#
# For standard datasets (--dataset-layout standard):
#
#   export PROVE_MAPILLARY_ROOT="/path/to/mapillary_vistas"
#   export PROVE_CITYSCAPES_ROOT="/path/to/cityscapes"
#   export PROVE_BDD_ROOT="/path/to/bdd100k"
#   export PROVE_ACDC_ROOT="/path/to/ACDC"
#

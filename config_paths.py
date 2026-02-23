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
# Example Environment Setup
# ============================================================================
# 
# Add to your ~/.bashrc or run before executing:
#
#   export PROVE_DATA_ROOT="/path/to/your/data"
#   export PROVE_WEIGHTS_ROOT="/path/to/weights"
#

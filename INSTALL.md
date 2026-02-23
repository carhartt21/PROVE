# Installation Guide for PROVE

## Prerequisites

- Python 3.8+
- PyTorch 1.10+ (with CUDA support recommended)
- GPU with at least 24GB VRAM (for training)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/prove.git
cd prove

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Before running, configure data paths in `config_paths.py` or set environment variables:

```bash
export PROVE_DATA_ROOT="/path/to/your/data"
export PROVE_WEIGHTS_ROOT="/path/to/weights"
```

## Data Preparation

Prepare your data following the expected structure:
```
data/
├── train/
│   └── images/
├── val/
│   └── images/
└── test/
    └── images/
```

## Running Experiments

See README.md for detailed usage instructions.

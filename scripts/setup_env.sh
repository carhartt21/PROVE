#!/bin/bash
# PROVE Environment Setup Script
# This script creates and configures the mamba environment for PROVE

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Configuration
# ============================================================================

ENV_NAME="prove"
PYTHON_VERSION="3.10"

# Default data paths (can be overridden by environment variables)
DEFAULT_DATA_ROOT="${PROVE_DATA_ROOT:-${AWARE_DATA_ROOT}/FINAL_SPLITS}"
DEFAULT_GEN_ROOT="${PROVE_GEN_ROOT:-${AWARE_DATA_ROOT}/GENERATED_IMAGES}"
DEFAULT_WEIGHTS_ROOT="${PROVE_WEIGHTS_ROOT:-${AWARE_DATA_ROOT}/WEIGHTS}"

# ============================================================================
# Helper Functions
# ============================================================================

check_mamba() {
    if command -v mamba &> /dev/null; then
        echo "mamba"
    elif command -v conda &> /dev/null; then
        echo "conda"
    else
        echo ""
    fi
}

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                  PROVE Environment Setup                             ║"
    echo "║    Pipeline for Recognition & Object Vision Evaluation               ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --create        Create the mamba environment"
    echo "  --update        Update existing environment"
    echo "  --install-deps  Install pip dependencies only (after env activation)"
    echo "  --verify        Verify installation"
    echo "  --configure     Set up environment variables and paths"
    echo "  --all           Run full setup (create + verify + configure)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Full installation"
    echo "  $0 --create                 # Just create environment"
    echo "  $0 --verify                 # Check installation"
    echo ""
}

# ============================================================================
# Environment Creation
# ============================================================================

create_environment() {
    local PKG_MANAGER=$(check_mamba)
    
    if [ -z "$PKG_MANAGER" ]; then
        echo_error "Neither mamba nor conda found. Please install mamba or conda first."
        echo "  Install mamba: https://github.com/conda-forge/miniforge"
        exit 1
    fi
    
    echo_info "Using package manager: $PKG_MANAGER"
    
    # Check if environment already exists
    if $PKG_MANAGER env list | grep -q "^${ENV_NAME} "; then
        echo_warn "Environment '$ENV_NAME' already exists."
        read -p "Do you want to remove and recreate it? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo_info "Removing existing environment..."
            $PKG_MANAGER env remove -n $ENV_NAME -y
        else
            echo_info "Keeping existing environment. Use --update to update it."
            return 0
        fi
    fi
    
    echo_info "Creating mamba environment from environment.yml..."
    $PKG_MANAGER env create -f environment.yml
    
    echo_success "Environment '$ENV_NAME' created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  $PKG_MANAGER activate $ENV_NAME"
}

update_environment() {
    local PKG_MANAGER=$(check_mamba)
    
    if [ -z "$PKG_MANAGER" ]; then
        echo_error "Neither mamba nor conda found."
        exit 1
    fi
    
    echo_info "Updating environment from environment.yml..."
    $PKG_MANAGER env update -n $ENV_NAME -f environment.yml
    
    echo_success "Environment updated successfully!"
}

install_pip_deps() {
    echo_info "Installing pip dependencies..."
    
    # Ensure we're in the right environment
    if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
        echo_warn "Please activate the '$ENV_NAME' environment first:"
        echo "  mamba activate $ENV_NAME"
        exit 1
    fi
    
    # Install OpenMIM first
    pip install -U openmim
    
    # Use MIM to install mmengine
    mim install mmengine
    
    # Install MMSegmentation and MMDetection
    pip install mmsegmentation==1.2.2 mmdet==3.3.0
    
    # Install remaining requirements
    pip install -r requirements.txt
    
    echo_success "Pip dependencies installed!"
}

# ============================================================================
# Verification
# ============================================================================

verify_installation() {
    echo_info "Verifying installation..."
    
    local all_ok=true
    
    # Check Python
    echo -n "  Python: "
    if command -v python &> /dev/null; then
        python_ver=$(python --version 2>&1)
        echo_success "$python_ver"
    else
        echo_error "Not found"
        all_ok=false
    fi
    
    # Check PyTorch
    echo -n "  PyTorch: "
    if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
        :
    else
        echo_error "Not installed"
        all_ok=false
    fi
    
    # Check CUDA availability
    echo -n "  CUDA available: "
    if python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null; then
        :
    else
        echo_warn "Could not check"
    fi
    
    # Check MMCV
    echo -n "  MMCV: "
    if python -c "import mmcv; print(mmcv.__version__)" 2>/dev/null; then
        :
    else
        echo_error "Not installed"
        all_ok=false
    fi
    
    # Check MMEngine
    echo -n "  MMEngine: "
    if python -c "import mmengine; print(mmengine.__version__)" 2>/dev/null; then
        :
    else
        echo_error "Not installed"
        all_ok=false
    fi
    
    # Check MMSegmentation
    echo -n "  MMSegmentation: "
    if python -c "import mmseg; print(mmseg.__version__)" 2>/dev/null; then
        :
    else
        echo_error "Not installed"
        all_ok=false
    fi
    
    # Check MMDetection
    echo -n "  MMDetection: "
    if python -c "import mmdet; print(mmdet.__version__)" 2>/dev/null; then
        :
    else
        echo_error "Not installed"
        all_ok=false
    fi
    
    # Check unified training config
    echo -n "  PROVE modules: "
    if python -c "from unified_training_config import UnifiedTrainingConfig; print('OK')" 2>/dev/null; then
        :
    else
        echo_error "Import failed"
        all_ok=false
    fi
    
    echo ""
    if [ "$all_ok" = true ]; then
        echo_success "All components verified successfully!"
        return 0
    else
        echo_error "Some components failed verification."
        return 1
    fi
}

# ============================================================================
# Configuration
# ============================================================================

configure_paths() {
    echo_info "Configuring environment variables..."
    
    # Create a local .env file for project-specific settings
    cat > .env <<EOF
# PROVE Environment Configuration
# Source this file to set up environment variables:
#   source .env

# Data paths
export PROVE_DATA_ROOT="${DEFAULT_DATA_ROOT}"
export PROVE_GEN_ROOT="${DEFAULT_GEN_ROOT}"
export PROVE_WEIGHTS_ROOT="${DEFAULT_WEIGHTS_ROOT}"
export PROVE_CONFIG_ROOT="./multi_model_configs"

# CUDA configuration (adjust as needed)
# export CUDA_VISIBLE_DEVICES=0

# OpenMMLab configuration
export MPLCONFIGDIR=\${HOME}/.config/matplotlib

# Add project to PYTHONPATH
export PYTHONPATH="\${PYTHONPATH}:$(pwd)"
EOF
    
    echo_success "Created .env configuration file"
    echo ""
    echo "To load the configuration, run:"
    echo "  source .env"
    echo ""
    echo "You can modify .env to set custom data paths."
}

# ============================================================================
# Main
# ============================================================================

main() {
    print_banner
    
    case "${1:-}" in
        --create)
            create_environment
            ;;
        --update)
            update_environment
            ;;
        --install-deps)
            install_pip_deps
            ;;
        --verify)
            verify_installation
            ;;
        --configure)
            configure_paths
            ;;
        --all)
            create_environment
            echo ""
            configure_paths
            echo ""
            echo_info "After activating the environment, run:"
            echo "  mamba activate $ENV_NAME"
            echo "  source .env"
            echo "  ./setup_env.sh --verify"
            ;;
        --help|-h|"")
            print_usage
            ;;
        *)
            echo_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"

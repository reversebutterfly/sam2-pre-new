#!/bin/bash
# MemoryShield: Environment setup on remote GPU server
# Usage: bash setup_env.sh

set -e

ENV_NAME="memshield"
PYTHON_VER="3.10"

echo "=== MemoryShield Environment Setup ==="

# 1. Create conda env
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "[skip] conda env '$ENV_NAME' already exists"
else
    echo "[create] conda env '$ENV_NAME' with python $PYTHON_VER"
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

# 2. Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 3. Install PyTorch (CUDA 11.8 for V100 compatibility)
echo "[install] PyTorch with CUDA 11.8"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install SAM2
echo "[install] SAM2 from Meta"
pip install git+https://github.com/facebookresearch/sam2.git

# 5. Install other dependencies
echo "[install] Other dependencies"
pip install -r requirements.txt

# 6. Download SAM2 checkpoint if not present
CKPT_DIR="checkpoints"
CKPT_FILE="$CKPT_DIR/sam2.1_hiera_tiny.pt"
if [ -f "$CKPT_FILE" ]; then
    echo "[skip] Checkpoint already exists: $CKPT_FILE"
else
    echo "[download] SAM2.1 hiera_tiny checkpoint"
    mkdir -p "$CKPT_DIR"
    wget -q -O "$CKPT_FILE" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
fi

# 7. Check DAVIS dataset
if [ -d "data/davis/JPEGImages/480p" ]; then
    echo "[ok] DAVIS dataset found"
else
    echo "[warn] DAVIS dataset not found at data/davis/"
    echo "       Please symlink or download DAVIS 2017:"
    echo "       ln -s /path/to/DAVIS data/davis"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Run with: python run_memshield.py --help"

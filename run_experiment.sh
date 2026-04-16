#!/bin/bash
# MemoryShield Experiment Runner
# Usage: bash run_experiment.sh [pilot|full]
#
# Runs inside a screen session on the GPU server.
# Default: pilot mode (5 videos, 40 frames, ~30 min on V100)

set -e

MODE="${1:-pilot}"
DEVICE="cuda:0"
PROJECT_DIR="$HOME/sam2_pre_new"

cd "$PROJECT_DIR"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate memshield

echo "============================================"
echo "  MemoryShield Experiment: $MODE"
echo "  Device: $DEVICE"
echo "  Time: $(date)"
echo "============================================"

# Check GPU
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader -i 0

# Verify SAM2 loads
python -c "
from memshield.surrogate import SAM2Surrogate
import torch
s = SAM2Surrogate('checkpoints/sam2.1_hiera_tiny.pt', 'configs/sam2.1/sam2.1_hiera_t.yaml', torch.device('$DEVICE'))
print(f'SAM2 loaded: FIFO bank={s.num_maskmem}')
print('Smoke test passed!')
" 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: SAM2 smoke test failed. Check environment."
    exit 1
fi

echo ""
echo "Starting experiment..."
echo ""

# Run MemoryShield
if [ "$MODE" = "pilot" ]; then
    python run_memshield.py \
        --mode pilot \
        --max_frames 40 \
        --device "$DEVICE" \
        --n_steps 300 \
        --epsilon 8.0 \
        --output_dir results_memshield \
        2>&1 | tee results_memshield/pilot_log.txt
elif [ "$MODE" = "full" ]; then
    python run_memshield.py \
        --mode full \
        --max_frames 50 \
        --device "$DEVICE" \
        --n_steps 300 \
        --epsilon 8.0 \
        --output_dir results_memshield \
        2>&1 | tee results_memshield/full_log.txt
fi

echo ""
echo "============================================"
echo "  Experiment complete: $(date)"
echo "============================================"

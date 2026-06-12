#!/bin/bash
# IoV Secure Aggregation FL — Linux/macOS Setup Script
set -e

echo "=== Installing PyTorch + torchvision (CPU) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing remaining dependencies ==="
pip install "numpy<3" matplotlib pandas pyyaml scipy

echo "=== Verifying ==="
python -c "import torch, torchvision, numpy; print('torch:', torch.__version__, '| numpy:', numpy.__version__)"

echo "=== Setup complete. Run:  python main.py ==="

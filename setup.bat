@echo off
REM ============================================================
REM  IoV Secure Aggregation FL — Windows Setup Script
REM  Run this once from the project root (the folder that
REM  contains main.py) before running main.py.
REM ============================================================

echo === Step 1: Installing PyTorch + torchvision (CPU build) ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo === Step 2: Installing remaining dependencies ===
pip install numpy "numpy<3" matplotlib pandas pyyaml scipy

echo === Step 3: Verifying installation ===
python -c "import torch, torchvision, numpy; print('torch:', torch.__version__, '| torchvision:', torchvision.__version__, '| numpy:', numpy.__version__)"

echo === Setup complete. Run:  python main.py ===

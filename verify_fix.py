"""
verify_fix.py — run this AFTER applying the PRG scale fix.

Usage:
    cd iov_secure_agg
    python verify_fix.py

Checks:
  1. Mask norm before and after fix (should be ~1.0 after)
  2. Runs 2 rounds of exp3_masking and prints agg_norm
     (should be ~3-15, not ~2325)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("=" * 55)
print("  IoV Secure Agg — PRG Scale Verification")
print("=" * 55)

# ── 1. Import and check mask norm ─────────────────────────
from crypto import MaskEngine

model_dim = 21840   # LeNet parameter count
engine    = MaskEngine(model_dim)

seed     = b'A' * 32
round_id = b'test_round_0'
mask     = engine._prg.expand(seed, model_dim, round_id)
norm     = np.linalg.norm(mask)

print(f"\nModel dim : {model_dim}")
print(f"Mask norm : {norm:.4f}")

if norm > 10:
    print("\n  [FAIL] Mask norm is too large.")
    print("  The fix has NOT been applied yet.")
    print()
    print("  Open  crypto/__init__.py")
    print("  Find class PRG → method expand()")
    print("  Change the last return line to:")
    print()
    print("    return (raw / (2 ** 63)) / np.sqrt(dim)")
    print()
    print("  Save the file, then run this script again.")
    sys.exit(1)
else:
    print(f"\n  [PASS] Mask norm ≈ {norm:.4f} (target: ~1.0)")

# ── 2. Quick 2-round smoke test ───────────────────────────
print("\nRunning 2-round exp3_masking smoke test...")
print("(This may take 1-2 minutes)\n")

import warnings
warnings.filterwarnings("ignore")
import random
import torch
from config import Config

cfg = Config(
    seed=42, num_vehicles=8, num_rsus=2,
    rounds=2, local_epochs=1, batch_size=64,
    dataset="MNIST", iid=True, dropout_rate=0.0,
    transform_mode="none",
    results_dir="results_verify",
    figures_dir="results_verify/figures",
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from experiments import exp3_masking
logger = exp3_masking(cfg)

print()
print("=" * 55)
print("  Results")
print("=" * 55)

for r in logger.records:
    status = "OK" if r.agg_norm < 50 else "STILL BROKEN"
    print(f"  Round {r.round_idx+1}: acc={r.test_accuracy*100:.2f}%  "
          f"agg_norm={r.agg_norm:.2f}  [{status}]")

final = logger.records[-1]
norm_ok  = final.agg_norm < 50
acc_ok   = final.test_accuracy > 0.20

print()
if norm_ok and acc_ok:
    print("  [ALL PASS] Fix is working correctly.")
    print(f"  agg_norm = {final.agg_norm:.2f}  (was ~2325 before fix)")
    print(f"  accuracy = {final.test_accuracy*100:.2f}%  (was ~15% before fix)")
    print()
    print("  You can now run the full paper-scale experiments:")
    print("  python main.py --rounds 50 --num_vehicles 100")
else:
    if not norm_ok:
        print(f"  [FAIL] agg_norm = {final.agg_norm:.2f}  (still too large)")
        print("  The fix may not have been saved correctly.")
    if not acc_ok:
        print(f"  [WARN] accuracy = {final.test_accuracy*100:.2f}%")
        print("  Low accuracy is expected in only 2 rounds — run more rounds to confirm.")
    sys.exit(1)

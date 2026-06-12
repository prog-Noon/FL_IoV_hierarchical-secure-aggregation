"""
main.py — IoV Hierarchical Federated Learning Research Prototype

Entry point for running all experiments and generating paper figures.

Usage:
    python main.py                              # use config.yaml defaults
    python main.py --rounds 50                  # override specific parameters
    python main.py --experiments exp1_baseline exp7_full
    python main.py --dataset FashionMNIST --num_vehicles 50

IMPORTANT — run from the project root (the folder containing main.py):
    cd iov_secure_agg
    python main.py          ← correct
    python iov_secure_agg/main.py  ← also correct
"""

from __future__ import annotations
import os
import sys

# ── Ensure we're running from the right directory ────────────────────────────
# If the user ran `python main.py` from a parent directory, os.path.dirname
# gives an empty string.  Normalise it to the script's own directory so that
# relative imports (config/, datasets/, …) always resolve correctly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(_HERE)   # make relative paths (config.yaml, results/) work from anywhere

# ── Pre-flight environment check ─────────────────────────────────────────────
def _preflight() -> None:
    """
    Verify the Python environment before importing any project modules.

    Catches:
      1. NumPy ABI mismatch (NumPy 2.x + old torch compiled against NumPy 1.x)
      2. Missing torchvision
      3. Wrong Python version

    Prints a human-readable fix for every problem found.
    """
    errors: list[str] = []

    # ── Python version ────────────────────────────────────────────────────────
    if sys.version_info < (3, 9):
        errors.append(
            f"Python 3.9+ required. You have {sys.version_info.major}.{sys.version_info.minor}.\n"
            "  Fix: install Python 3.11 from https://www.python.org/downloads/"
        )

    # ── NumPy ─────────────────────────────────────────────────────────────────
    try:
        import numpy as np
        np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
    except ImportError:
        errors.append("NumPy not found.\n  Fix: pip install numpy")
        np_ver = (0, 0)

    # ── PyTorch ───────────────────────────────────────────────────────────────
    torch_ok = False
    try:
        import torch
        # Try a numpy interop call — this is what triggers the ABI crash
        _ = torch.zeros(1).numpy()
        torch_ok = True
    except ImportError:
        errors.append(
            "PyTorch not found.\n"
            "  Fix: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
    except Exception as e:
        # ABI mismatch produces RuntimeError or similar at numpy interop
        if np_ver >= (2, 0):
            errors.append(
                f"NumPy/PyTorch ABI mismatch detected.\n"
                f"  Your NumPy: {np.__version__}  (2.x)\n"
                f"  Your torch was compiled against NumPy 1.x and is incompatible.\n\n"
                f"  Fix — run these commands in order:\n"
                f"    pip uninstall torch torchvision torchaudio -y\n"
                f"    pip install torch torchvision "
                f"--index-url https://download.pytorch.org/whl/cpu\n\n"
                f"  Alternatively, downgrade NumPy:\n"
                f"    pip install \"numpy<2\"\n\n"
                f"  (Original error: {e})"
            )
        else:
            errors.append(f"PyTorch import failed: {e}")

    # ── torchvision ───────────────────────────────────────────────────────────
    if torch_ok:
        try:
            import torchvision   # noqa: F401
        except ImportError:
            errors.append(
                "torchvision not found.\n"
                "  It must be installed alongside torch — do NOT install them separately.\n"
                "  Fix:\n"
                "    pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu\n\n"
                "  Without torchvision the code falls back to a synthetic\n"
                "  random dataset (accuracy ~10%), which is fine for smoke testing.\n"
                "  For real MNIST/CIFAR10 results, torchvision is required."
            )
            # This is a WARNING not a hard error — we can still run with synthetic data.
            print(f"\n[WARNING] {errors.pop()}\n")

    # ── matplotlib ────────────────────────────────────────────────────────────
    try:
        import matplotlib   # noqa: F401
    except ImportError:
        errors.append("matplotlib not found.\n  Fix: pip install matplotlib")

    # ── Report ────────────────────────────────────────────────────────────────
    if errors:
        print("\n" + "=" * 65)
        print("  ENVIRONMENT ERROR — cannot start")
        print("=" * 65)
        for i, e in enumerate(errors, 1):
            print(f"\n  [{i}] {e}")
        print("\n" + "=" * 65)
        print("  Quick fix (Windows):\n")
        print("    Run setup.bat from the project directory.")
        print("    Or manually:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("    pip install numpy matplotlib pandas pyyaml scipy")
        print("=" * 65 + "\n")
        sys.exit(1)


_preflight()

# ── Normal imports (only reached if pre-flight passes) ────────────────────────
import random
import time
import traceback

import numpy as np
import torch

from config        import load_config, Config
from experiments   import (
    exp1_baseline, exp2_hierarchical, exp3_masking,
    exp4_dropout, exp5_dp, exp6_projection, exp7_full,
    scalability_sweep, exp8_centralised_v2x, run_3gpp_analysis,
)
from visualization import generate_all_figures


# =============================================================================
# Seed everything for reproducibility
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# Experiment dispatcher
# =============================================================================

EXPERIMENT_MAP = {
    "exp1_baseline":        exp1_baseline,
    "exp2_hierarchical":    exp2_hierarchical,
    "exp3_masking":         exp3_masking,
    "exp4_dropout":         exp4_dropout,
    "exp5_dp":              exp5_dp,
    "exp6_projection":      exp6_projection,
    "exp7_full":            exp7_full,
    "exp8_centralised_v2x": exp8_centralised_v2x,
}


def run_experiments(cfg: Config) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  IoV Secure Aggregation FL — Research Prototype")
    print(f"  Dataset:      {cfg.dataset}")
    print(f"  Vehicles:     {cfg.num_vehicles}   RSUs: {cfg.num_rsus}")
    print(f"  Rounds:       {cfg.rounds}   Epochs/round: {cfg.local_epochs}")
    print(f"  Partitioning: {'IID' if cfg.iid else f'Non-IID (α={cfg.dirichlet_alpha})'}")
    print(f"  Seed:         {cfg.seed}")
    print(f"  Results dir:  {os.path.abspath(cfg.results_dir)}")
    print(f"{'='*60}")

    overall_start = time.time()

    for exp_name in cfg.experiments:
        if exp_name not in EXPERIMENT_MAP and exp_name != "scalability":
            print(f"[WARN] Unknown experiment: {exp_name!r} — skipping")
            continue
        try:
            set_all_seeds(cfg.seed)
            if exp_name == "scalability":
                scalability_sweep(cfg)
            else:
                EXPERIMENT_MAP[exp_name](cfg)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user. Saving partial results...")
            break
        except Exception as e:
            print(f"\n[ERROR] {exp_name} failed: {e}")
            traceback.print_exc()
            print("Continuing with next experiment...\n")

    # Scalability sweep (for communication overhead + latency figures)
    if len(cfg.vehicle_counts) > 1:
        try:
            set_all_seeds(cfg.seed)
            scalability_sweep(cfg)
        except Exception as e:
            print(f"[WARN] Scalability sweep failed: {e}")

    # 3GPP NR-V2X communication analysis (always runs — no FL training needed)
    try:
        run_3gpp_analysis(cfg)
    except Exception as e:
        print(f"[WARN] 3GPP analysis failed: {e}")

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  All experiments complete in {elapsed:.1f}s")
    print(f"  Results:  {os.path.abspath(cfg.results_dir)}/")
    print(f"  Figures:  {os.path.abspath(cfg.figures_dir)}/")
    print(f"{'='*60}\n")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    cfg = load_config(cli_args=True)
    set_all_seeds(cfg.seed)

    run_experiments(cfg)

    print("Generating publication figures...")
    generate_all_figures(cfg.results_dir, cfg.figures_dir, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())

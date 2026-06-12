"""
config/__init__.py

Loads config.yaml and exposes a single Config dataclass.
CLI arguments override YAML values so experiments can be launched with:
  python main.py --rounds 50 --dropout_rate 0.2
"""

from __future__ import annotations
import argparse
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Federation
    num_vehicles:            int   = 20
    num_rsus:                int   = 4
    rounds:                  int   = 15
    local_epochs:            int   = 2
    learning_rate:           float = 0.01
    batch_size:              int   = 32
    participation_fraction:  float = 1.0

    # Dataset
    dataset:          str   = "MNIST"
    iid:              bool  = False
    dirichlet_alpha:  float = 0.5

    # Crypto
    shamir_threshold: int = 2
    shamir_n_shares:  int = 3

    # Dropout
    dropout_rate: float = 0.1

    # Transformation
    transform_mode: str   = "none"
    dp_epsilon:     float = 1.0
    dp_sensitivity: float = 1.0
    projection_dim: int   = 64

    # Communication
    float_bytes: int = 4

    # Sweep values
    dropout_rates:    List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    dp_epsilons:      List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    projection_dims:  List[int]   = field(default_factory=lambda: [16, 32, 64, 128, 256])
    vehicle_counts:   List[int]   = field(default_factory=lambda: [10, 20, 40])

    # Experiments
    experiments: List[str] = field(default_factory=lambda: [
        "exp1_baseline", "exp2_hierarchical", "exp3_masking",
        "exp4_dropout", "exp5_dp", "exp6_projection", "exp7_full",
    ])

    # Output
    results_dir: str = "results"
    figures_dir: str = "results/figures"

    def model_dim(self) -> int:
        """Flat parameter count for the dataset's default model."""
        if self.dataset == "CIFAR10":
            return 62006   # ResNet-small-like
        return 21840   # CNN for MNIST/FashionMNIST (set after model construction)

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(yaml_path: Optional[str] = None, cli_args: bool = True) -> Config:
    """
    Load configuration from YAML file, then apply CLI overrides.

    Priority: CLI > YAML > dataclass defaults.
    """
    cfg_dict: dict = {}

    # 1. Load YAML
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    yaml_path = os.path.abspath(yaml_path)
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            cfg_dict = yaml.safe_load(f) or {}

    # 2. Parse CLI arguments (only when running as script)
    if cli_args:
        parser = argparse.ArgumentParser(description="IoV Secure Aggregation FL")
        parser.add_argument("--config",          type=str,   default=None)
        parser.add_argument("--seed",            type=int,   default=None)
        parser.add_argument("--num_vehicles",    type=int,   default=None)
        parser.add_argument("--num_rsus",        type=int,   default=None)
        parser.add_argument("--rounds",          type=int,   default=None)
        parser.add_argument("--local_epochs",    type=int,   default=None)
        parser.add_argument("--learning_rate",   type=float, default=None)
        parser.add_argument("--batch_size",      type=int,   default=None)
        parser.add_argument("--dataset",         type=str,   default=None)
        parser.add_argument("--iid",             action="store_true", default=None)
        parser.add_argument("--dirichlet_alpha", type=float, default=None)
        parser.add_argument("--dropout_rate",    type=float, default=None)
        parser.add_argument("--transform_mode",  type=str,   default=None)
        parser.add_argument("--dp_epsilon",      type=float, default=None)
        parser.add_argument("--projection_dim",  type=int,   default=None)
        parser.add_argument("--experiments",     type=str,   nargs="+", default=None)
        args, _ = parser.parse_known_args()

        # If --config was given, reload YAML from that path
        if args.config:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        # Override with explicit CLI values
        for key, val in vars(args).items():
            if key != "config" and val is not None:
                cfg_dict[key] = val

    # 3. Build Config object, ignoring unknown YAML keys
    known = {f.name for f in Config.__dataclass_fields__.values()}
    filtered = {k: v for k, v in cfg_dict.items() if k in known}
    return Config(**filtered)

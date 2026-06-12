"""
metrics/__init__.py

RoundMetrics: per-round data container.
MetricsLogger: accumulates RoundMetrics and exports to CSV.
"""

from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class RoundMetrics:
    round_idx:   int   = 0

    # Accuracy / Loss
    test_accuracy: float = 0.0
    test_loss:     float = 0.0

    # Phase timings (seconds)
    phase1_time:    float = 0.0
    phase2_time:    float = 0.0
    phase3_time:    float = 0.0
    phase4_time:    float = 0.0
    phase5_time:    float = 0.0
    phase6_time:    float = 0.0
    round_time:     float = 0.0
    avg_train_time: float = 0.0

    # Participation
    num_selected: int = 0
    num_dropped:  int = 0
    n_active:     int = 0

    # Communication (bytes)
    vehicle_to_rsu_bytes:  int = 0
    rsu_to_coord_bytes:    int = 0
    coord_to_server_bytes: int = 0
    total_comm_bytes:      int = 0

    # Aggregate quality
    agg_norm: float = 0.0

    # Dropout
    dropout_recovery_success: bool = True

    # Privacy
    dp_epsilon:     float = 0.0
    projection_dim: int   = 0


class MetricsLogger:
    """
    Accumulates per-round metrics and writes them to a CSV file.
    """

    def __init__(self, experiment_name: str, results_dir: str):
        self.experiment_name = experiment_name
        self.results_dir     = results_dir
        self.records: List[RoundMetrics] = []
        os.makedirs(results_dir, exist_ok=True)

    def log(self, m: RoundMetrics) -> None:
        self.records.append(m)

    def save_csv(self) -> str:
        path = os.path.join(self.results_dir, f"{self.experiment_name}.csv")
        if not self.records:
            return path
        rows = [asdict(r) for r in self.records]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def final_accuracy(self) -> float:
        return self.records[-1].test_accuracy if self.records else 0.0

    def final_loss(self) -> float:
        return self.records[-1].test_loss if self.records else float("inf")

    def summary(self) -> dict:
        if not self.records:
            return {}
        last = self.records[-1]
        total_comm = sum(r.total_comm_bytes for r in self.records)
        return {
            "experiment":     self.experiment_name,
            "rounds":         len(self.records),
            "final_accuracy": last.test_accuracy,
            "final_loss":     last.test_loss,
            "total_comm_MB":  total_comm / 1e6,
            "avg_round_s":    sum(r.round_time for r in self.records) / len(self.records),
            "recovery_rate":  sum(r.dropout_recovery_success for r in self.records) / len(self.records),
        }

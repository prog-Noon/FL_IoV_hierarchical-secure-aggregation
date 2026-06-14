"""
visualization/__init__.py

Generate all paper figures from logged CSV data.

Figure list:
  Fig 1 — Accuracy vs Round (multiple experiments)
  Fig 2 — Loss vs Round
  Fig 3 — Communication Overhead vs Number of Vehicles
  Fig 4 — Latency (round time) vs Number of Vehicles
  Fig 5 — Recovery Success Rate vs Dropout Rate
  Fig 6 — Accuracy vs DP Epsilon
  Fig 7 — Accuracy vs Projection Dimension

Each figure is saved as PNG, PDF, and SVG in the configured figures directory.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from metrics import MetricsLogger


# ── Style ─────────────────────────────────────────────────────────────────────

STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "lines.linewidth":   2.0,
    "lines.markersize":  5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _savefig(fig, path_base: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path_base}.{ext}")
    plt.close(fig)


def _setup() -> None:
    plt.rcParams.update(STYLE)


# =============================================================================
# Figure 1 — Test Accuracy vs Round
# =============================================================================

def fig_accuracy_vs_round(
    loggers:     Dict[str, MetricsLogger],
    figures_dir: str,
    title:       str = "Test Accuracy vs. Communication Round",
) -> None:
    _setup()

    high = {k: v for k, v in loggers.items()
            if v.records and v.records[-1].test_accuracy >= 0.5}
    low  = {k: v for k, v in loggers.items()
            if v.records and v.records[-1].test_accuracy < 0.5}
    has_both = bool(high) and bool(low)

    if has_both:
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(title, fontsize=13)
    else:
        fig, ax_main = plt.subplots(figsize=(7, 4.5))
        ax_zoom = None
        ax_main.set_title(title)

    for i, (label, logger) in enumerate(loggers.items()):
        rounds = [m.round_idx + 1 for m in logger.records]
        accs   = [m.test_accuracy * 100 for m in logger.records]
        lw = 2.5 if logger.records[-1].test_accuracy >= 0.5 else 1.5
        ax_main.plot(rounds, accs,
                     color=COLORS[i % len(COLORS)],
                     marker=MARKERS[i % len(MARKERS)],
                     markevery=max(1, len(rounds) // 8),
                     linewidth=lw,
                     label=label)
    ax_main.set_xlabel("Round")
    ax_main.set_ylabel("Test Accuracy (%)")
    ax_main.set_xlim(left=1)
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if not has_both:
        ax_main.legend(loc="lower right")
    else:
        ax_main.set_title("All experiments")
        ax_main.legend(loc="center right", fontsize=9)

    if ax_zoom is not None and high:
        for i, (label, logger) in enumerate(loggers.items()):
            if label not in high:
                continue
            rounds = [m.round_idx + 1 for m in logger.records]
            accs   = [m.test_accuracy * 100 for m in logger.records]
            ax_zoom.plot(rounds, accs,
                         color=COLORS[i % len(COLORS)],
                         marker=MARKERS[i % len(MARKERS)],
                         markevery=max(1, len(rounds) // 8),
                         linewidth=2.5,
                         label=label)
        min_acc = min(
            m.test_accuracy * 100
            for lg in high.values() for m in lg.records
        )
        ax_zoom.set_ylim(bottom=max(0, min_acc - 5), top=101)
        ax_zoom.set_xlabel("Round")
        ax_zoom.set_ylabel("Test Accuracy (%)")
        ax_zoom.set_title("Zoomed: high-accuracy experiments")
        ax_zoom.set_xlim(left=1)
        ax_zoom.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax_zoom.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    _savefig(fig, os.path.join(figures_dir, "fig1_accuracy_vs_round"))

# =============================================================================
# Figure 2 — Training Loss vs Round
# =============================================================================

def fig_loss_vs_round(
    loggers:     Dict[str, MetricsLogger],
    figures_dir: str,
    title:       str = "Test Loss vs. Communication Round",
) -> None:
    _setup()

    low_loss  = {k: v for k, v in loggers.items()
                 if v.records and v.records[-1].test_loss <= 1.0}
    high_loss = {k: v for k, v in loggers.items()
                 if v.records and v.records[-1].test_loss > 1.0}
    has_both = bool(low_loss) and bool(high_loss)

    if has_both:
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(title, fontsize=13)
    else:
        fig, ax_main = plt.subplots(figsize=(7, 4.5))
        ax_zoom = None
        ax_main.set_title(title)

    for i, (label, logger) in enumerate(loggers.items()):
        rounds = [m.round_idx + 1 for m in logger.records]
        losses = [m.test_loss for m in logger.records]
        ax_main.plot(rounds, losses,
                     color=COLORS[i % len(COLORS)],
                     marker=MARKERS[i % len(MARKERS)],
                     markevery=max(1, len(rounds) // 8),
                     label=label)
    ax_main.set_xlabel("Round")
    ax_main.set_ylabel("Cross-Entropy Loss")
    ax_main.set_xlim(left=1)
    if not has_both:
        ax_main.legend(loc="upper right")
        ax_main.set_title(title)
    else:
        ax_main.set_title("All experiments (log scale)")
        ax_main.set_yscale("log")
        ax_main.legend(loc="upper right", fontsize=9)

    if ax_zoom is not None and low_loss:
        for i, (label, logger) in enumerate(loggers.items()):
            if label not in low_loss:
                continue
            rounds = [m.round_idx + 1 for m in logger.records]
            losses = [m.test_loss for m in logger.records]
            ax_zoom.plot(rounds, losses,
                         color=COLORS[i % len(COLORS)],
                         marker=MARKERS[i % len(MARKERS)],
                         markevery=max(1, len(rounds) // 8),
                         label=label)
        ax_zoom.set_xlabel("Round")
        ax_zoom.set_ylabel("Cross-Entropy Loss")
        ax_zoom.set_title("Zoomed: converged experiments (Exp 1 & 2)")
        ax_zoom.set_xlim(left=1)
        ax_zoom.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _savefig(fig, os.path.join(figures_dir, "fig2_loss_vs_round"))

# =============================================================================
# Figure 3 — Communication Overhead vs Number of Vehicles
# =============================================================================

def fig_comm_vs_vehicles(
    scalability_loggers: List[MetricsLogger],
    vehicle_counts:      List[int],
    figures_dir:         str,
) -> None:
    _setup()
    total_mb = [
        sum(r.total_comm_bytes for r in lg.records) / 1e6
        for lg in scalability_loggers
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vehicle_counts, total_mb,
            color=COLORS[0], marker="o", label="Total Comm. (MB)")
    ax.set_xlabel("Number of Vehicles")
    ax.set_ylabel("Total Communication (MB)")
    ax.set_title("Communication Overhead vs. Number of Vehicles")
    ax.legend()
    _savefig(fig, os.path.join(figures_dir, "fig3_comm_vs_vehicles"))


# =============================================================================
# Figure 4 — Latency vs Number of Vehicles
# =============================================================================

def fig_latency_vs_vehicles(
    scalability_loggers: List[MetricsLogger],
    vehicle_counts:      List[int],
    figures_dir:         str,
) -> None:
    _setup()
    avg_round_s = [
        np.mean([r.round_time for r in lg.records])
        for lg in scalability_loggers
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vehicle_counts, avg_round_s,
            color=COLORS[1], marker="s", label="Avg Round Time (s)")
    ax.set_xlabel("Number of Vehicles")
    ax.set_ylabel("Average Round Time (s)")
    ax.set_title("Round Latency vs. Number of Vehicles")
    ax.legend()
    _savefig(fig, os.path.join(figures_dir, "fig4_latency_vs_vehicles"))


# =============================================================================
# Figure 5 — Recovery Success Rate vs Dropout Rate
# =============================================================================

def fig_recovery_vs_dropout(
    dropout_loggers: List[MetricsLogger],
    dropout_rates:   List[float],
    figures_dir:     str,
) -> None:
    _setup()
    final_acc  = [lg.final_accuracy() * 100 for lg in dropout_loggers]
    rec_rate   = [
        100 * sum(r.dropout_recovery_success for r in lg.records) / len(lg.records)
        for lg in dropout_loggers
    ]
    rates_pct = [r * 100 for r in dropout_rates]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(rates_pct, final_acc,
             color=COLORS[0], marker="o", label="Final Accuracy (%)")
    ax1.set_xlabel("Dropout Rate (%)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy vs. Dropout Rate")
    ax1.legend()

    ax2.bar(rates_pct, rec_rate, color=COLORS[2], width=5)
    ax2.set_xlabel("Dropout Rate (%)")
    ax2.set_ylabel("Recovery Success Rate (%)")
    ax2.set_title("Recovery Success vs. Dropout Rate")
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    _savefig(fig, os.path.join(figures_dir, "fig5_recovery_vs_dropout"))


# =============================================================================
# Figure 6 — Accuracy vs DP Epsilon
# =============================================================================

def fig_accuracy_vs_epsilon(
    dp_loggers: List[MetricsLogger],
    epsilons:   List[float],
    figures_dir: str,
) -> None:
    _setup()
    final_acc = [lg.final_accuracy() * 100 for lg in dp_loggers]
    fig, ax   = plt.subplots(figsize=(6, 4))
    ax.semilogx(epsilons, final_acc,
                color=COLORS[3], marker="D", label="Final Accuracy (%)")
    ax.set_xlabel("Privacy Budget ε (log scale)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs. DP Privacy Budget (ε)")
    ax.legend()
    _savefig(fig, os.path.join(figures_dir, "fig6_accuracy_vs_epsilon"))


# =============================================================================
# Figure 7 — Accuracy vs Projection Dimension
# =============================================================================

def fig_accuracy_vs_projection(
    proj_loggers: List[MetricsLogger],
    proj_dims:    List[int],
    figures_dir:  str,
) -> None:
    _setup()
    final_acc = [lg.final_accuracy() * 100 for lg in proj_loggers]
    fig, ax   = plt.subplots(figsize=(6, 4))
    ax.plot(proj_dims, final_acc,
            color=COLORS[4], marker="^", label="Final Accuracy (%)")
    ax.set_xlabel("Projection Dimension")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs. Random Projection Dimension")
    ax.legend()
    _savefig(fig, os.path.join(figures_dir, "fig7_accuracy_vs_projection"))




# =============================================================================
# Figure 8 — 3GPP NR-V2X: Latency vs Number of Vehicles
# =============================================================================

def fig_3gpp_latency(
    vehicle_counts: List[int],
    figures_dir:    str,
    model_dim:      int = 21840,
    float_bytes:    int = 4,
) -> None:
    """
    Show V2RSU latency and total round latency vs number of vehicles
    for the three 3GPP scenarios (Urban, Highway, Suburban), with a
    horizontal reference line at the 3GPP 100 ms V2RSU budget.
    """
    _setup()
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from metrics.comm_model import (
            CommModel, URBAN_SCENARIO, HIGHWAY_SCENARIO, SUBURBAN_SCENARIO
        )
    except ImportError:
        return

    n_rsus = 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for (name, cfg, color, ls) in [
        ("Urban",    URBAN_SCENARIO,    COLORS[0], "-"),
        ("Highway",  HIGHWAY_SCENARIO,  COLORS[2], "--"),
        ("Suburban", SUBURBAN_SCENARIO, COLORS[1], "-."),
    ]:
        model   = CommModel(cfg)
        v2rsu   = [model.evaluate_round(model_dim, n, n_rsus, float_bytes).v2rsu_latency_ms
                   for n in vehicle_counts]
        total   = [model.evaluate_round(model_dim, n, n_rsus, float_bytes).total_comm_latency_ms
                   for n in vehicle_counts]
        ax1.plot(vehicle_counts, v2rsu,  color=color, ls=ls, marker="o", label=name)
        ax2.plot(vehicle_counts, total, color=color, ls=ls, marker="s", label=name)

    ax1.axhline(100, color="gray", ls=":", lw=1.5, label="3GPP budget (100 ms)")
    ax1.set_xlabel("Number of vehicles")
    ax1.set_ylabel("V2RSU latency (ms)")
    ax1.set_title("V2RSU Latency vs Vehicles (3GPP NR-V2X)")
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Number of vehicles")
    ax2.set_ylabel("Total round comm. latency (ms)")
    ax2.set_title("End-to-End Latency vs Vehicles")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, os.path.join(figures_dir, "fig8_3gpp_latency"))


# =============================================================================
# Figure 9 — Centralised vs Proposed: communication comparison
# =============================================================================

def fig_3gpp_comparison(
    vehicle_counts: List[int],
    figures_dir:    str,
    model_dim:      int = 21840,
    n_rsus:         int = 5,
    float_bytes:    int = 4,
) -> None:
    """
    Bar chart comparing centralised V2X baseline vs proposed hierarchical
    FL protocol on: latency, total bytes, and energy (per round).
    """
    _setup()
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from metrics.comm_model import CommModel, URBAN_SCENARIO
    except ImportError:
        return

    model   = CommModel(URBAN_SCENARIO)
    n_v_ref = vehicle_counts[-1]   # use the largest vehicle count for comparison
    cmp     = model.compare_to_centralised(model_dim, n_v_ref, n_rsus, float_bytes)

    metrics = ["Latency (ms)", "Data (MB)", "Energy (mJ)"]
    central = [cmp["centralised"]["latency_ms"],
               cmp["centralised"]["total_bytes_mb"],
               cmp["centralised"]["energy_mj"]]
    proposed = [cmp["proposed"]["latency_ms"],
                cmp["proposed"]["total_bytes_mb"],
                cmp["proposed"]["energy_mj"]]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, central,  w, label="Centralised V2X (no FL)",   color=COLORS[3], alpha=0.85)
    b2 = ax.bar(x + w/2, proposed, w, label="Proposed (hierarchical FL)", color=COLORS[0], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Centralised V2X vs Proposed FL ({n_v_ref} vehicles, Urban scenario)")
    ax.legend()

    # Annotate bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _savefig(fig, os.path.join(figures_dir, "fig9_3gpp_comparison"))


# =============================================================================
# Convenience: generate all figures from saved CSVs
# =============================================================================

def generate_all_figures(
    results_dir: str,
    figures_dir: str,
    cfg,
) -> None:
    """
    Load all CSVs from results_dir and generate every figure.
    Called at the end of main.py after all experiments complete.
    """
    os.makedirs(figures_dir, exist_ok=True)

    def load_csv(name: str) -> Optional[pd.DataFrame]:
        path = os.path.join(results_dir, f"{name}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def df_to_logger(df: pd.DataFrame, name: str) -> MetricsLogger:
        from metrics import RoundMetrics
        lg = MetricsLogger(name, results_dir)
        for _, row in df.iterrows():
            m = RoundMetrics(**{k: row[k] for k in row.index if hasattr(RoundMetrics, k)})
            lg.records.append(m)
        return lg

    # ── Fig 1 & 2: convergence curves ─────────────────────────────────────────
    conv_map = {}
    for exp_name, label in [
        ("exp1_baseline",   "Baseline FedAvg"),
        ("exp2_hierarchical", "Hierarchical FL"),
        ("exp3_masking",    "Masking (no dropout)"),
        ("exp4_dropout_10pct", "Masking + 10% dropout"),
        ("exp7_full",       "Full Framework"),
    ]:
        df = load_csv(exp_name)
        if df is not None:
            conv_map[label] = df_to_logger(df, exp_name)

    if conv_map:
        fig_accuracy_vs_round(conv_map, figures_dir)
        fig_loss_vs_round(conv_map, figures_dir)

    # ── Fig 3 & 4: scalability ─────────────────────────────────────────────────
    scale_loggers, scale_counts = [], []
    for n_v in cfg.vehicle_counts:
        df = load_csv(f"scalability_v{n_v}")
        if df is not None:
            scale_loggers.append(df_to_logger(df, f"scale_{n_v}"))
            scale_counts.append(n_v)
    if scale_loggers:
        fig_comm_vs_vehicles(scale_loggers, scale_counts, figures_dir)
        fig_latency_vs_vehicles(scale_loggers, scale_counts, figures_dir)

    # ── Fig 5: dropout recovery ────────────────────────────────────────────────
    drop_loggers, drop_rates = [], []
    for rate in cfg.dropout_rates:
        df = load_csv(f"exp4_dropout_{int(rate*100)}pct")
        if df is not None:
            drop_loggers.append(df_to_logger(df, f"drop_{rate}"))
            drop_rates.append(rate)
    if drop_loggers:
        fig_recovery_vs_dropout(drop_loggers, drop_rates, figures_dir)

    # ── Fig 6: DP epsilon ──────────────────────────────────────────────────────
    dp_loggers, dp_eps = [], []
    for eps in cfg.dp_epsilons:
        df = load_csv(f"exp5_dp_eps{eps}")
        if df is not None:
            dp_loggers.append(df_to_logger(df, f"dp_{eps}"))
            dp_eps.append(eps)
    if dp_loggers:
        fig_accuracy_vs_epsilon(dp_loggers, dp_eps, figures_dir)

    # ── Fig 7: projection dim ──────────────────────────────────────────────────
    proj_loggers, proj_dims = [], []
    for pdim in cfg.projection_dims:
        df = load_csv(f"exp6_proj_dim{pdim}")
        if df is not None:
            proj_loggers.append(df_to_logger(df, f"proj_{pdim}"))
            proj_dims.append(pdim)
    if proj_loggers:
        fig_accuracy_vs_projection(proj_loggers, proj_dims, figures_dir)

    # ── Fig 8 & 9: 3GPP analysis ─────────────────────────────────────────────
    try:
        from datasets.models import build_model, count_parameters
        model_dim = count_parameters(build_model(cfg.dataset))
    except Exception:
        model_dim = 21840   # LeNet fallback
    fig_3gpp_latency(cfg.vehicle_counts + [100, 200], figures_dir, model_dim, cfg.float_bytes)
    fig_3gpp_comparison(cfg.vehicle_counts + [100, 200], figures_dir, model_dim, cfg.num_rsus, cfg.float_bytes)

    print(f"\nAll figures saved to {figures_dir}/")

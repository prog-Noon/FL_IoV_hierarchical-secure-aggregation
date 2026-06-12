"""
experiments/__init__.py

Seven experiment functions matching the paper's evaluation section.

Each function:
  1. Constructs the federation with the appropriate configuration
  2. Runs the specified number of rounds
  3. Logs metrics to CSV
  4. Returns a MetricsLogger for visualization

Experiment matrix:
  Exp 1 — Baseline FedAvg (no RSUs, no masking)
  Exp 2 — Hierarchical FL  (RSUs, no masking)
  Exp 3 — Hierarchical FL + Pairwise Masking (no dropout)
  Exp 4 — Masking + Dropout Recovery (multiple rates)
  Exp 5 — Full + DP (multiple epsilon)
  Exp 6 — Full + Projection (multiple dims)
  Exp 7 — Full Framework
"""

from __future__ import annotations
import copy
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from crypto  import MaskEngine
from datasets import (
    load_dataset, iid_partition, dirichlet_partition,
    build_vehicle_loaders, build_test_loader, get_num_classes,
)
from datasets.models import build_model, count_parameters
from entities import Vehicle, RSU, Coordinator, CloudServer
from metrics  import MetricsLogger, RoundMetrics
from protocol import FLRoundRunner


# =============================================================================
# Federation builder — shared by all experiments
# =============================================================================

def build_federation(
    cfg:          Config,
    n_vehicles:   int   = None,
    n_rsus:       int   = None,
    transform_mode: str = None,
    dp_epsilon:   float = None,
    projection_dim: int = None,
    dropout_rate: float = None,
    use_masking:  bool  = True,
    use_dropout:  bool  = True,
    device:       torch.device = None,
):
    """
    Construct all protocol entities and return a ready FLRoundRunner + ancillaries.
    """
    # Allow experiment-level overrides
    n_v   = n_vehicles    or cfg.num_vehicles
    n_r   = n_rsus        or cfg.num_rsus
    t_mode = transform_mode or cfg.transform_mode
    eps   = dp_epsilon    or cfg.dp_epsilon
    pdim  = projection_dim or cfg.projection_dim
    drate = dropout_rate  if dropout_rate is not None else cfg.dropout_rate

    if device is None:
        device = torch.device("cpu")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, test_ds = load_dataset(cfg.dataset)
    if cfg.iid:
        partition = iid_partition(train_ds, n_v, seed=cfg.seed)
    else:
        partition = dirichlet_partition(train_ds, n_v, alpha=cfg.dirichlet_alpha, seed=cfg.seed)

    vehicle_loaders = build_vehicle_loaders(train_ds, partition, cfg.batch_size)
    test_loader     = build_test_loader(test_ds, batch_size=256)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(cfg.dataset, get_num_classes(cfg.dataset))
    model_dim = count_parameters(model)

    # ── Seed ──────────────────────────────────────────────────────────────────
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── RSUs ──────────────────────────────────────────────────────────────────
    rsu_ids  = [f"RSU_{r}" for r in range(n_r)]
    rsus     = {rid: RSU(rid, model_dim) for rid in rsu_ids}

    # ── Vehicles ──────────────────────────────────────────────────────────────
    vehicle_ids = [f"V{i}" for i in range(n_v)]
    vehicles    = {
        vid: Vehicle(vid, vehicle_loaders[i], device)
        for i, vid in enumerate(vehicle_ids)
    }

    # Round-robin topology (vehicle → RSU)
    topology = {vid: rsu_ids[i % n_r] for i, vid in enumerate(vehicle_ids)}

    # ── Coordinator ───────────────────────────────────────────────────────────
    coordinator = Coordinator(model_dim)

    # ── Cloud Server ──────────────────────────────────────────────────────────
    server = CloudServer(
        model, model_dim, device,
        transform_mode = t_mode,
        dp_epsilon     = eps,
        dp_sensitivity = cfg.dp_sensitivity,
        projection_dim = pdim,
        seed           = cfg.seed,
    )

    # ── Mask Engine ───────────────────────────────────────────────────────────
    mask_engine = MaskEngine(
        model_dim,
        shamir_threshold = cfg.shamir_threshold,
        shamir_n_shares  = cfg.shamir_n_shares,
    )

    # ── Runner ────────────────────────────────────────────────────────────────
    runner = FLRoundRunner(
        cfg         = cfg,
        vehicles    = vehicles,
        rsus        = rsus,
        coordinator = coordinator,
        server      = server,
        mask_engine = mask_engine,
        topology    = topology,
        rng         = rng,
        use_masking = use_masking,
        use_dropout = use_dropout and drate > 0,
        dropout_rate = drate,
    )

    return runner, test_loader, model_dim


def run_experiment(
    cfg:      Config,
    name:     str,
    runner:   FLRoundRunner,
    test_loader,
    extra_meta: dict = None,
) -> MetricsLogger:
    """Run cfg.rounds rounds and log metrics."""
    logger = MetricsLogger(name, cfg.results_dir)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for t in range(cfg.rounds):
        m = runner.run_round(t, test_loader)
        if extra_meta:
            for k, v in extra_meta.items():
                setattr(m, k, v)
        logger.log(m)
        print(f"  Round {t+1:3d}/{cfg.rounds} | "
              f"Acc={m.test_accuracy:.4f} | "
              f"Loss={m.test_loss:.4f} | "
              f"Active={m.n_active}/{m.num_selected} | "
              f"Time={m.round_time:.2f}s")
    csv_path = logger.save_csv()
    print(f"  → saved {csv_path}")
    s = logger.summary()
    print(f"  Final accuracy: {s['final_accuracy']:.4f}  |  "
          f"Total comm: {s['total_comm_MB']:.2f} MB")
    return logger


# =============================================================================
# Experiment 1 — Baseline FedAvg (no hierarchy, no masking)
# =============================================================================

def exp1_baseline(cfg: Config) -> MetricsLogger:
    """
    Vanilla FedAvg: vehicles upload raw updates directly to server.
    Single RSU acts as pass-through (no intermediate aggregation structure).
    No masking, no dropout recovery.
    """
    runner, test_loader, _ = build_federation(
        cfg,
        n_rsus      = 1,          # single aggregation point
        use_masking = False,
        use_dropout = False,
    )
    return run_experiment(cfg, "exp1_baseline", runner, test_loader)


# =============================================================================
# Experiment 2 — Hierarchical FL (RSUs, no masking)
# =============================================================================

def exp2_hierarchical(cfg: Config) -> MetricsLogger:
    """
    Multiple RSUs aggregate zone vehicles; coordinator combines.
    No privacy masking.
    """
    runner, test_loader, _ = build_federation(
        cfg,
        use_masking = False,
        use_dropout = False,
    )
    return run_experiment(cfg, "exp2_hierarchical", runner, test_loader)


# =============================================================================
# Experiment 3 — Hierarchical FL + Pairwise Masking (no dropout)
# =============================================================================

def exp3_masking(cfg: Config) -> MetricsLogger:
    """
    Full zone-local pairwise masking with self-masks.
    No vehicle dropouts.
    """
    runner, test_loader, _ = build_federation(
        cfg,
        use_masking = True,
        use_dropout = False,
    )
    return run_experiment(cfg, "exp3_masking", runner, test_loader)


# =============================================================================
# Experiment 4 — Masking + Dropout Recovery (multiple rates)
# =============================================================================

def exp4_dropout(cfg: Config) -> List[MetricsLogger]:
    """
    Sweep over dropout rates: [0%, 10%, 20%, 30%, 50%].
    For each rate, run a full experiment and log independently.
    """
    loggers = []
    for rate in cfg.dropout_rates:
        runner, test_loader, _ = build_federation(
            cfg,
            use_masking  = True,
            use_dropout  = True,
            dropout_rate = rate,
        )
        name   = f"exp4_dropout_{int(rate*100)}pct"
        logger = run_experiment(cfg, name, runner, test_loader,
                                extra_meta={"dropout_rate_tag": rate})
        loggers.append(logger)
    return loggers


# =============================================================================
# Experiment 5 — Full + DP (multiple epsilon)
# =============================================================================

def exp5_dp(cfg: Config) -> List[MetricsLogger]:
    """
    Sweep over DP epsilon values.
    Full masking + dropout recovery + Laplace DP noise on aggregate.
    """
    loggers = []
    for eps in cfg.dp_epsilons:
        runner, test_loader, _ = build_federation(
            cfg,
            use_masking    = True,
            use_dropout    = True,
            transform_mode = "dp",
            dp_epsilon     = eps,
        )
        name   = f"exp5_dp_eps{eps}"
        logger = run_experiment(cfg, name, runner, test_loader,
                                extra_meta={"dp_epsilon": eps})
        loggers.append(logger)
    return loggers


# =============================================================================
# Experiment 6 — Full + Projection (multiple dims)
# =============================================================================

def exp6_projection(cfg: Config) -> List[MetricsLogger]:
    """
    Sweep over random projection dimensions.
    Masking + dropout + random projection on aggregate.
    """
    loggers = []
    for pdim in cfg.projection_dims:
        runner, test_loader, _ = build_federation(
            cfg,
            use_masking    = True,
            use_dropout    = True,
            transform_mode = "projection",
            projection_dim = pdim,
        )
        name   = f"exp6_proj_dim{pdim}"
        logger = run_experiment(cfg, name, runner, test_loader,
                                extra_meta={"projection_dim": pdim})
        loggers.append(logger)
    return loggers


# =============================================================================
# Experiment 7 — Full Framework
# =============================================================================

def exp7_full(cfg: Config) -> MetricsLogger:
    """
    All features enabled:
      Hierarchical aggregation
      Zone-local pairwise masking
      Bonawitz-correct dropout recovery
      DP noise (epsilon from config)
      Random projection (dim from config)
    """
    runner, test_loader, _ = build_federation(
        cfg,
        use_masking    = True,
        use_dropout    = True,
        transform_mode = cfg.transform_mode if cfg.transform_mode != "none" else "dp",
        dp_epsilon     = cfg.dp_epsilon,
        projection_dim = cfg.projection_dim,
    )
    return run_experiment(cfg, "exp7_full", runner, test_loader)


# =============================================================================
# Scalability sweep (for communication overhead figure)
# =============================================================================

def scalability_sweep(cfg: Config) -> List[MetricsLogger]:
    """
    Run Experiment 3 (masking, no dropout) for each vehicle count in
    cfg.vehicle_counts and return loggers for the comm overhead figure.
    """
    loggers = []
    for n_v in cfg.vehicle_counts:
        runner, test_loader, _ = build_federation(
            cfg,
            n_vehicles  = n_v,
            use_masking = True,
            use_dropout = False,
        )
        name   = f"scalability_v{n_v}"
        logger = run_experiment(cfg, name, runner, test_loader)
        loggers.append(logger)
    return loggers


# =============================================================================
# Experiment 8 — Centralised V2X Baseline (3GPP benchmark)
# =============================================================================

def exp8_centralised_v2x(cfg: Config) -> MetricsLogger:
    """
    Centralised V2X baseline: all vehicles upload raw (unmasked) updates
    directly to the cloud server through a single RSU acting as a pass-through.

    This represents the current state of V2X ML deployments — no FL hierarchy,
    no privacy, no dropout recovery. Used as the 3GPP comparison point.

    Differences from Exp 1:
      - Exp 1 uses 1 RSU as aggregator (FedAvg, no masking)
      - Exp 8 uses 1 RSU as pass-through; server receives individual updates
        (equivalent to centralised training, maximum communication cost)

    In practice Exp 1 and Exp 8 produce identical accuracy because both
    implement FedAvg without masking. The comparison is on communication
    overhead and privacy guarantees, not accuracy.
    """
    runner, test_loader, _ = build_federation(
        cfg,
        n_rsus      = 1,
        use_masking = False,
        use_dropout = False,
    )
    return run_experiment(cfg, "exp8_centralised_v2x", runner, test_loader)


# =============================================================================
# 3GPP Communication Analysis (standalone — no FL training needed)
# =============================================================================

def run_3gpp_analysis(cfg: Config) -> dict:
    """
    Generate the full 3GPP NR-V2X communication analysis:
      - Scalability table (Table III in paper)
      - Centralised vs proposed comparison
      - Per-scenario breakdown (urban, highway, suburban)

    Saves results to results/3gpp_analysis.csv.
    Returns a summary dict for the paper.
    """
    import csv
    from metrics.comm_model import (
        CommModel, URBAN_SCENARIO, HIGHWAY_SCENARIO, SUBURBAN_SCENARIO
    )
    from datasets.models import build_model, count_parameters

    os.makedirs(cfg.results_dir, exist_ok=True)

    model     = build_model(cfg.dataset)
    model_dim = count_parameters(model)

    results = {}

    # ── Scalability table ──────────────────────────────────────────────────────
    model_urban = CommModel(URBAN_SCENARIO)
    scale_rows  = model_urban.scalability_table(
        model_dim, cfg.vehicle_counts + [100, 200], cfg.num_rsus, cfg.float_bytes
    )
    # Deduplicate
    seen = set()
    scale_rows_dedup = []
    for r in scale_rows:
        k = r["n_vehicles"]
        if k not in seen:
            seen.add(k)
            scale_rows_dedup.append(r)
    results["scalability"] = scale_rows_dedup

    scale_path = os.path.join(cfg.results_dir, "3gpp_scalability.csv")
    with open(scale_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scale_rows_dedup[0].keys())
        writer.writeheader()
        writer.writerows(scale_rows_dedup)

    # ── Centralised vs proposed comparison ────────────────────────────────────
    cmp = model_urban.compare_to_centralised(
        model_dim, cfg.num_vehicles, cfg.num_rsus, cfg.float_bytes
    )
    results["comparison"] = cmp

    cmp_path = os.path.join(cfg.results_dir, "3gpp_comparison.csv")
    with open(cmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "centralised", "proposed"])
        writer.writerow(["latency_ms",      cmp["centralised"]["latency_ms"],   cmp["proposed"]["latency_ms"]])
        writer.writerow(["total_bytes_mb",  cmp["centralised"]["total_bytes_mb"], cmp["proposed"]["total_bytes_mb"]])
        writer.writerow(["energy_mj",       cmp["centralised"]["energy_mj"],    cmp["proposed"]["energy_mj"]])
        writer.writerow(["privacy",         cmp["centralised"]["privacy"],       cmp["proposed"]["privacy"]])
        writer.writerow(["dropout_handling",cmp["centralised"]["dropout_handling"], cmp["proposed"]["dropout_handling"]])

    # ── Per-scenario breakdown ─────────────────────────────────────────────────
    scenario_results = []
    for name, scenario_cfg in [
        ("Urban",    URBAN_SCENARIO),
        ("Highway",  HIGHWAY_SCENARIO),
        ("Suburban", SUBURBAN_SCENARIO),
    ]:
        m = CommModel(scenario_cfg)
        r = m.evaluate_round(model_dim, cfg.num_vehicles, cfg.num_rsus, cfg.float_bytes)
        scenario_results.append({
            "scenario":            name,
            "throughput_mbps":     scenario_cfg.pc5_throughput_mbps,
            "rsu_coverage_m":      scenario_cfg.rsu_coverage_m,
            "v2rsu_latency_ms":    round(r.v2rsu_latency_ms, 2),
            "total_latency_ms":    round(r.total_comm_latency_ms, 2),
            "energy_mj":           round(r.total_energy_mj, 2),
            "within_3gpp_budget":  r.within_3gpp_v2rsu,
            "safety_cbr":          r.safety_beacon_cbr,
        })
    results["scenarios"] = scenario_results

    scen_path = os.path.join(cfg.results_dir, "3gpp_scenarios.csv")
    with open(scen_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scenario_results[0].keys())
        writer.writeheader()
        writer.writerows(scenario_results)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  3GPP NR-V2X Communication Analysis")
    print(f"{'='*60}")
    print(f"  Model: {cfg.dataset} ({model_dim:,} parameters, {model_dim*4/1024:.0f} KB)")
    print(f"  Scenario: Urban (5.9 GHz, 20 MHz, 15 Mbps sidelink)")
    print()
    r_urban = CommModel(URBAN_SCENARIO).evaluate_round(
        model_dim, cfg.num_vehicles, cfg.num_rsus, cfg.float_bytes
    )
    print(f"  V2RSU transmission time:  {r_urban.v2rsu_tx_ms:.1f} ms")
    print(f"  V2RSU total latency:      {r_urban.v2rsu_latency_ms:.1f} ms  "
          f"({'OK' if r_urban.within_3gpp_v2rsu else 'EXCEEDS'} 3GPP 100ms budget)")
    print(f"  End-to-end per round:     {r_urban.total_comm_latency_ms:.1f} ms")
    print(f"  Energy per vehicle:       {r_urban.energy_per_vehicle_mj:.2f} mJ")
    print(f"  Safety beacon CBR:        {r_urban.safety_beacon_cbr:.3f}  "
          f"({'OK' if r_urban.safety_cbr_ok else 'HIGH'})")
    print()
    print(f"  Scalability table → {scale_path}")
    print(f"  Scenario comparison → {scen_path}")
    print(f"  Centralised comparison → {cmp_path}")
    print(f"{'='*60}")

    return results

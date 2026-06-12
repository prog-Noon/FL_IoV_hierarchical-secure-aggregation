"""
protocol/__init__.py

FLRoundRunner: orchestrates one complete FL round across all six phases.

Phase 1 — Round Initialization
Phase 2 — Mask Setup
Phase 3 — Local Training and Masking
Phase 4 — Dropout Detection and Recovery
Phase 5 — Hierarchical Aggregation
Phase 6 — Aggregate Transformation and Global Update
"""

from __future__ import annotations
import hashlib
import hmac as _hmac
import os
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from config import Config
from crypto  import MaskEngine
from entities import Vehicle, RSU, Coordinator, CloudServer
from metrics  import RoundMetrics


class FLRoundRunner:
    """
    Executes one federated learning round following the six-phase protocol.

    Instantiate once per experiment; call run_round() repeatedly.
    """

    def __init__(
        self,
        cfg:         Config,
        vehicles:    Dict[str, Vehicle],
        rsus:        Dict[str, RSU],
        coordinator: Coordinator,
        server:      CloudServer,
        mask_engine: MaskEngine,
        topology:    Dict[str, str],    # vehicle_id → rsu_id
        rng:         random.Random,
        *,
        use_masking:   bool = True,
        use_dropout:   bool = True,
        dropout_rate:  float = 0.0,
    ):
        self.cfg          = cfg
        self.vehicles     = vehicles
        self.rsus         = rsus
        self.coordinator  = coordinator
        self.server       = server
        self.mask_engine  = mask_engine
        self.topology     = topology
        self.rng          = rng

        self.use_masking  = use_masking
        self.use_dropout  = use_dropout
        self.dropout_rate = dropout_rate

        self._server_key = os.urandom(32)   # HMAC key for RID generation

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def phase1_initialize(self, t: int) -> Tuple[List[str], bytes, Dict[str, str]]:
        """
        Select participants, generate RID, assign to RSUs.
        Returns (selected_vehicles, rid, vehicle→rsu topology).
        """
        n_select = max(2, int(len(self.vehicles) * self.cfg.participation_fraction))
        selected = self.rng.sample(list(self.vehicles.keys()), n_select)

        # RID = HMAC-SHA256(server_key, round_index ∥ nonce)
        nonce = os.urandom(16)
        msg   = t.to_bytes(8, "big") + nonce
        rid   = _hmac.new(self._server_key, msg, hashlib.sha256).digest()

        # Assign each selected vehicle to its pre-configured RSU
        topo = {v: self.topology[v] for v in selected}

        # Reset RSU zones for this round
        for rsu in self.rsus.values():
            rsu.zone_vehicles.clear()
            rsu.set_round_id(rid)
        for vid, rsu_id in topo.items():
            self.rsus[rsu_id].zone_vehicles.add(vid)

        return selected, rid, topo

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def phase2_mask_setup(
        self, selected: List[str], rid: bytes, topo: Dict[str, str]
    ) -> Dict[str, dict]:
        """
        Generate zone-local pairwise masks and self-masks.
        Distribute recovery shares to RSUs.
        Returns {vehicle_id: mask_bundle} for Phase 3.
        """
        # Group selected vehicles by RSU zone (Fix 3: O(m²) per zone)
        zones: Dict[str, List[str]] = {}
        for vid, rsu_id in topo.items():
            zones.setdefault(rsu_id, []).append(vid)

        # Generate pairwise masks (zone-local only)
        all_pairwise: Dict[str, Dict[str, Tuple[bytes, np.ndarray]]] = {
            v: {} for v in selected
        }
        for rsu_id, zone_vids in zones.items():
            zone_masks = self.mask_engine.generate_zone_masks(zone_vids, rid)
            for vid, peers in zone_masks.items():
                all_pairwise[vid] = peers

        # Generate self-masks
        all_self = self.mask_engine.generate_self_masks(selected, rid)

        # Pre-split self_seeds once (both bundle and RSU use same shares)
        self_shares_map: Dict[str, List[Tuple[int, bytes]]] = {
            v: self.mask_engine.shamir.split(all_self[v][0])
            for v in selected
        }

        # Create per-vehicle bundles and deposit shares with RSUs
        bundles: Dict[str, dict] = {}
        for vid in selected:
            rsu_id              = topo[vid]
            self_seed, bi       = all_self[vid]
            pairwise_masks      = {vj: mask for vj, (_, mask) in all_pairwise[vid].items()}
            self_shares         = self_shares_map[vid]

            # Recovery shares (pairwise seeds split via Shamir)
            pairwise_shares: Dict[str, List[Tuple[int, bytes]]] = {
                vj: self.mask_engine.shamir.split(seed)
                for vj, (seed, _) in all_pairwise[vid].items()
            }

            bundle = {
                "vehicle_id":      vid,
                "self_mask":       bi,
                "self_seed":       self_seed,
                "self_seed_shares": self_shares,
                "pairwise_masks":  pairwise_masks,
            }
            bundles[vid] = bundle

            # Deposit to RSU
            rsu_bundle = {
                "vehicle_id":      vid,
                "self_shares":     self_shares,
                "pairwise_shares": pairwise_shares,
            }
            self.rsus[rsu_id].deposit_shares(rsu_bundle)

        return bundles

    # ── Phase 3 ───────────────────────────────────────────────────────────────

    def phase3_local_train(
        self,
        selected:  List[str],
        bundles:   Dict[str, dict],
        dropped:   Set[str],
    ) -> Dict[str, float]:
        """
        Each active vehicle trains locally and uploads a masked update.
        Returns {vehicle_id: train_time}.
        """
        global_model   = self.server.model
        train_times    = {}

        for vid in selected:
            if vid in dropped:
                continue   # simulate dropout: vehicle never uploads

            vehicle = self.vehicles[vid]
            vehicle.prepare_round(global_model, bundles[vid] if self.use_masking else {
                "self_mask": np.zeros(self.mask_engine._dim),
                "pairwise_masks": {},
                "self_seed_shares": [],
                "self_seed": b"",
            })

            # Train
            update = vehicle.train(self.cfg.learning_rate, self.cfg.local_epochs)

            if self.use_masking:
                # Mask and upload
                masked_update, _ = vehicle.mask_and_upload(update)
            else:
                # No masking — transmit raw update
                masked_update = update.cpu().numpy()
                vehicle._upload_done = True

            rsu_id = self.topology[vid]
            self.rsus[rsu_id].receive_masked_update(vid, masked_update)
            train_times[vid] = vehicle.last_train_time

        return train_times

    # ── Phase 4 ───────────────────────────────────────────────────────────────

    def phase4_dropout_recovery(
        self, selected: List[str], dropped: Set[str]
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Each RSU detects dropouts and produces a corrected local aggregate.
        Returns ({rsu_id: corrected_aggregate}, total_active_count).
        """
        aggregates:   Dict[str, np.ndarray] = {}
        total_active: int = 0

        for rsu_id, rsu in self.rsus.items():
            if not rsu.zone_vehicles:
                continue

            if self.use_masking:
                lr = rsu.resolve_dropouts(self.vehicles, self.mask_engine)
            else:
                # No masking: simple average of raw updates
                at, dt = rsu.detect_dropouts()
                if not at:
                    lr = np.zeros(self.mask_engine._dim)
                else:
                    lr = np.zeros(self.mask_engine._dim)
                    for vid in at:
                        lr += rsu._masked_updates.get(vid, np.zeros(self.mask_engine._dim))
                rsu._corrected_agg = lr

            if rsu._corrected_agg is not None:
                aggregates[rsu_id] = rsu._corrected_agg
                total_active += len(rsu._active_set or set())

        return aggregates, total_active

    # ── Phase 5 ───────────────────────────────────────────────────────────────

    def phase5_hierarchical_aggregation(
        self, aggregates: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Coordinator receives corrected RSU aggregates and sums them.
        G = Σ_r Lr  (Fix 2: no second dropout correction).
        """
        self.coordinator.reset_round()
        for rsu_id, lr in aggregates.items():
            self.coordinator.receive(rsu_id, lr)
        return self.coordinator.combine()

    # ── Phase 6 ───────────────────────────────────────────────────────────────

    def phase6_update_and_cleanup(
        self, G: np.ndarray, n_active: int, t: int
    ) -> None:
        """
        Apply optional transformation, update global model, cleanup state.
        """
        self.server.update_model(G, max(n_active, 1))

        # Cleanup RSU state
        for rsu in self.rsus.values():
            rsu.reset_round()
        # Reset vehicle state
        for v in self.vehicles.values():
            v.reset_round()
        self.server.cleanup()

    # ── Top-level round execution ──────────────────────────────────────────────

    def run_round(
        self, t: int, test_loader
    ) -> RoundMetrics:
        """
        Execute one complete FL round and return all metrics.
        """
        round_start = time.perf_counter()
        m           = RoundMetrics(round_idx=t)

        # Phase 1
        t1 = time.perf_counter()
        selected, rid, topo = self.phase1_initialize(t)
        m.phase1_time = time.perf_counter() - t1
        m.num_selected = len(selected)

        # Simulate dropouts
        dropped: Set[str] = set()
        if self.use_dropout and self.dropout_rate > 0:
            n_drop  = int(len(selected) * self.dropout_rate)
            dropped = set(self.rng.sample(selected, n_drop))
        m.num_dropped = len(dropped)

        # Phase 2
        t2 = time.perf_counter()
        if self.use_masking:
            bundles = self.phase2_mask_setup(selected, rid, topo)
        else:
            bundles = {v: {} for v in selected}
        m.phase2_time = time.perf_counter() - t2

        # Phase 3
        t3 = time.perf_counter()
        train_times = self.phase3_local_train(selected, bundles, dropped)
        m.phase3_time    = time.perf_counter() - t3
        m.avg_train_time = (sum(train_times.values()) / len(train_times)
                            if train_times else 0.0)

        # Phase 4
        t4 = time.perf_counter()
        aggregates, n_active = self.phase4_dropout_recovery(selected, dropped)
        m.phase4_time = time.perf_counter() - t4
        m.n_active    = n_active

        # Phase 5
        t5 = time.perf_counter()
        G  = self.phase5_hierarchical_aggregation(aggregates)
        m.phase5_time = time.perf_counter() - t5
        m.agg_norm    = float(np.linalg.norm(G))

        # Phase 6
        t6 = time.perf_counter()
        self.phase6_update_and_cleanup(G, n_active, t)
        m.phase6_time = time.perf_counter() - t6

        # Evaluate
        acc, loss = self.server.evaluate(test_loader)
        m.test_accuracy = acc
        m.test_loss     = loss

        # Communication overhead
        model_dim  = self.mask_engine._dim
        float_size = self.cfg.float_bytes
        m.vehicle_to_rsu_bytes  = (len(selected) - len(dropped)) * model_dim * float_size
        m.rsu_to_coord_bytes    = len(aggregates) * model_dim * float_size
        m.coord_to_server_bytes = model_dim * float_size
        m.total_comm_bytes      = (m.vehicle_to_rsu_bytes
                                   + m.rsu_to_coord_bytes
                                   + m.coord_to_server_bytes)

        m.round_time = time.perf_counter() - round_start
        m.dropout_recovery_success = (n_active > 0)

        return m

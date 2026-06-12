"""
entities/__init__.py

Four entity classes corresponding to the four tiers of the IoV FL architecture.

Vehicle      — FL client; trains locally and uploads masked updates
RSU          — Intermediate aggregator; handles dropout correction
Coordinator  — Combines RSU aggregates into federation-wide G
CloudServer  — Updates global model, applies transformation, broadcasts
"""

from __future__ import annotations
import copy
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.models import (
    get_flat_params, set_flat_params, compute_update, apply_update
)


# =============================================================================
# Vehicle (Algorithm 3 actor)
# =============================================================================

class Vehicle:
    """
    FL client.  Runs local SGD and uploads only masked updates.

    Phase 3 logic lives here.  No other class calls train().
    """

    def __init__(
        self,
        vid:        str,
        dataloader: DataLoader,
        device:     torch.device,
    ):
        self.vid        = vid
        self.dataloader = dataloader
        self.device     = device

        # Round-scoped state
        self._model:       Optional[nn.Module] = None
        self._mask_bundle: Optional[dict]      = None
        self._upload_done: bool                = False

        # Timing
        self.last_train_time: float = 0.0
        self.last_mask_time:  float = 0.0

    def prepare_round(self, global_model: nn.Module, mask_bundle: dict) -> None:
        """Install fresh model copy and mask bundle for this round."""
        self._model       = copy.deepcopy(global_model).to(self.device)
        self._mask_bundle = mask_bundle
        self._upload_done = False

    def train(self, lr: float, epochs: int) -> torch.Tensor:
        """
        Local SGD for `epochs` epochs.
        Returns unmasked update wi(t) = new_params - old_params.
        wi is NEVER stored on self after this call returns.
        """
        assert self._model is not None
        t0            = time.perf_counter()
        old_params    = get_flat_params(self._model)
        optimizer     = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=0.9)
        criterion     = nn.CrossEntropyLoss()
        self._model.train()

        for _ in range(epochs):
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._model(X), y)
                loss.backward()
                # Gradient clipping for bounded sensitivity (DP requirement)
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

        new_params           = get_flat_params(self._model)
        self.last_train_time = time.perf_counter() - t0
        return compute_update(old_params, new_params)

    def mask_and_upload(
        self, update: torch.Tensor
    ) -> Tuple[np.ndarray, float]:
        """
        Compute total mask Mi = bi + Σ m_ij (zone peers), apply to update.
        Returns (masked_update_numpy, mask_time_seconds).
        wi is discarded after masking — never stored.
        """
        assert self._mask_bundle is not None
        t0 = time.perf_counter()

        bi = self._mask_bundle["self_mask"]           # np.ndarray
        total_mask = bi.copy()
        for mij in self._mask_bundle["pairwise_masks"].values():
            total_mask += mij

        masked = update.cpu().numpy() + total_mask
        self._upload_done    = True
        self.last_mask_time  = time.perf_counter() - t0
        return masked, self.last_mask_time

    def provide_self_mask_removal_info(self) -> Tuple[int, bytes]:
        """
        Bonawitz-correct Fix 1: return share[0] of self_seed, not bi itself.
        RSU combines with its deposited share[1] to reconstruct bi.
        """
        assert self._mask_bundle is not None
        assert self._upload_done
        return self._mask_bundle["self_seed_shares"][0]

    def reset_round(self) -> None:
        self._model       = None
        self._mask_bundle = None
        self._upload_done = False

    @property
    def num_samples(self) -> int:
        return len(self.dataloader.dataset)


# =============================================================================
# RSU (Algorithm 4 actor)
# =============================================================================

class RSU:
    """
    Intermediate aggregator.

    Responsibilities (Phase 4):
      - Buffer masked updates from assigned vehicles
      - Detect dropouts
      - Reconstruct dropped pairwise masks
      - Collect self-mask removal info from active vehicles
      - Produce fully corrected local aggregate Lr
    """

    def __init__(self, rsu_id: str, model_dim: int):
        self.rsu_id    = rsu_id
        self._dim      = model_dim

        # Zone membership (set each round by runner)
        self.zone_vehicles: Set[str] = set()

        # Round-scoped buffers
        self._masked_updates:  Dict[str, np.ndarray]  = {}
        self._share_store:     Dict[str, dict]         = {}   # {vid: share_bundle}
        self._active_set:      Optional[Set[str]]      = None
        self._dropped_set:     Optional[Set[str]]      = None
        self._corrected_agg:   Optional[np.ndarray]    = None
        self._round_id:        Optional[bytes]         = None

        # Timing
        self.last_aggregation_time:  float = 0.0
        self.last_recovery_time:     float = 0.0

        # Communication tracking
        self.bytes_received: int = 0
        self.bytes_sent:     int = 0

    # ── Phase 2 interface ──────────────────────────────────────────────────────

    def deposit_shares(self, bundle: dict) -> None:
        """Store recovery shares from Phase 2."""
        self._share_store[bundle["vehicle_id"]] = bundle

    def set_round_id(self, rid: bytes) -> None:
        self._round_id = rid

    # ── Phase 3 interface ──────────────────────────────────────────────────────

    def receive_masked_update(self, vid: str, update: np.ndarray) -> None:
        self._masked_updates[vid] = update.copy()
        self.bytes_received += update.nbytes

    # ── Phase 4: dropout detection ─────────────────────────────────────────────

    def detect_dropouts(self) -> Tuple[Set[str], Set[str]]:
        at = set(self._masked_updates.keys()) & self.zone_vehicles
        dt = self.zone_vehicles - at
        self._active_set  = at
        self._dropped_set = dt
        return at, dt

    # ── Phase 4: self-mask removal info from active vehicles ──────────────────

    def collect_self_mask_removal_info(
        self,
        active: Set[str],
        vehicle_refs: Dict[str, Vehicle],
        mask_engine,
    ) -> Dict[str, np.ndarray]:
        """
        Fix 1 — Bonawitz-correct:
        Requests one Shamir share from each active vehicle.
        Combines with RSU's own deposited share to reconstruct bi.
        """
        self_masks: Dict[str, np.ndarray] = {}
        for vid in active:
            if vid not in self._share_store:
                continue
            vehicle_share = vehicle_refs[vid].provide_self_mask_removal_info()
            rsu_share     = self._share_store[vid]["self_shares"][1]   # index 1
            bi = mask_engine.reconstruct_self_mask(
                vehicle_share, rsu_share, self._round_id
            )
            self_masks[vid] = bi
        return self_masks

    # ── Phase 4: pairwise mask reconstruction for dropped vehicles ────────────

    def reconstruct_dropped_masks(
        self,
        dropped: Set[str],
        active:  Set[str],
        mask_engine,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """For each Vk ∈ Dt reconstruct m_kj for each Vj ∈ At."""
        dropped_masks: Dict[str, Dict[str, np.ndarray]] = {}
        for vk in dropped:
            if vk not in self._share_store:
                continue
            dropped_masks[vk] = mask_engine.reconstruct_dropped_masks(
                vk, list(active),
                self._share_store[vk],
                self._round_id,
            )
        return dropped_masks

    # ── Phase 4: corrected local aggregate (Fix 2: all correction here) ───────

    def compute_corrected_aggregate(
        self,
        self_masks:    Dict[str, np.ndarray],
        dropped_masks: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Lr = Σ_{i ∈ At} ŵi − Σ bi − Σ m_kj (dropped)

        Fix 2: all correction is done here; Phase 5 just sums Lr values.
        """
        t0  = time.perf_counter()
        agg = np.zeros(self._dim, dtype=np.float64)

        for vid in self._active_set:
            agg += self._masked_updates[vid]
        for bi in self_masks.values():
            agg -= bi
        for peer_masks in dropped_masks.values():
            for mkj in peer_masks.values():
                agg -= mkj

        self._corrected_agg         = agg
        self.last_aggregation_time  = time.perf_counter() - t0
        self.bytes_sent += agg.nbytes
        return agg

    def resolve_dropouts(
        self,
        vehicle_refs: Dict[str, Vehicle],
        mask_engine,
    ) -> np.ndarray:
        """Run full Phase 4 and return corrected Lr."""
        t0                 = time.perf_counter()
        at, dt             = self.detect_dropouts()
        self_masks         = self.collect_self_mask_removal_info(at, vehicle_refs, mask_engine)
        dropped_masks      = self.reconstruct_dropped_masks(dt, at, mask_engine)
        agg                = self.compute_corrected_aggregate(self_masks, dropped_masks)
        self.last_recovery_time = time.perf_counter() - t0
        return agg

    def reset_round(self) -> None:
        self._masked_updates.clear()
        self._share_store.clear()
        self._active_set    = None
        self._dropped_set   = None
        self._corrected_agg = None
        self._round_id      = None
        self.bytes_received = 0
        self.bytes_sent     = 0
        self.zone_vehicles.clear()


# =============================================================================
# Coordinator (Algorithm 5 actor)
# =============================================================================

class Coordinator:
    """
    Phase 5: combines RSU aggregates into federation-wide G.

    Fix 2: receives fully corrected Lr values — no second correction pass.
    G = Σ_r Lr  (plain sum).
    """

    def __init__(self, model_dim: int):
        self._dim = model_dim
        self._local_aggregates: Dict[str, np.ndarray] = {}
        self._combined:         Optional[np.ndarray]  = None
        self.last_G:            Optional[np.ndarray]  = None

        self.bytes_received: int   = 0
        self.bytes_sent:     int   = 0
        self.last_agg_time:  float = 0.0

    def receive(self, rsu_id: str, lr: np.ndarray) -> None:
        self._local_aggregates[rsu_id] = lr.copy()
        self.bytes_received += lr.nbytes

    def combine(self) -> np.ndarray:
        """G = Σ Lr — plain sum, no correction (Fix 2)."""
        t0 = time.perf_counter()
        G  = np.zeros(self._dim, dtype=np.float64)
        for lr in self._local_aggregates.values():
            G += lr
        self._combined     = G
        self.last_G        = G.copy()
        self.last_agg_time = time.perf_counter() - t0
        self.bytes_sent   += G.nbytes
        return G

    def reset_round(self) -> None:
        self._local_aggregates.clear()
        self._combined      = None
        self.bytes_received = 0
        self.bytes_sent     = 0


# =============================================================================
# CloudServer (Algorithm 6 actor)
# =============================================================================

class CloudServer:
    """
    Phase 6: optional aggregate transformation + FedAvg model update.

    Transformation modes: none | dp | projection | dp+projection
    """

    def __init__(
        self,
        model:          nn.Module,
        model_dim:      int,
        device:         torch.device,
        transform_mode: str   = "none",
        dp_epsilon:     float = 1.0,
        dp_sensitivity: float = 1.0,
        projection_dim: int   = 64,
        seed:           int   = 42,
    ):
        self.model          = model.to(device)
        self._dim           = model_dim
        self.device         = device
        self.transform_mode = transform_mode
        self.dp_epsilon     = dp_epsilon
        self.dp_sensitivity = dp_sensitivity
        self.projection_dim = projection_dim

        # Lazy-initialised random projection matrix (fixed across rounds)
        self._proj_matrix: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(seed)

        self.round_stats: List[dict] = []
        self.bytes_received: int = 0

    def get_flat_model(self) -> torch.Tensor:
        return get_flat_params(self.model)

    def _random_projection(self, G: np.ndarray) -> np.ndarray:
        src = len(G)
        if self.projection_dim >= src:
            return G
        if self._proj_matrix is None or self._proj_matrix.shape != (self.projection_dim, src):
            P = self._rng.standard_normal((self.projection_dim, src))
            self._proj_matrix = P / np.sqrt(self.projection_dim)
        return self._proj_matrix @ G

    def _add_dp_noise(self, G: np.ndarray) -> np.ndarray:
        scale = self.dp_sensitivity / self.dp_epsilon
        return G + self._rng.laplace(0, scale, size=G.shape)

    def apply_transform(self, G: np.ndarray) -> np.ndarray:
        """Apply configured aggregate transformation before model update."""
        mode = self.transform_mode
        if mode in ("projection", "dp+projection"):
            G = self._random_projection(G)
        if mode in ("dp", "dp+projection"):
            G = self._add_dp_noise(G)
        return G

    def update_model(self, G: np.ndarray, n_active: int) -> None:
        """
        FedAvg: W(t+1) = W(t) + (1/n_active) * G_transformed.
        Skips model update if projection changed dimension.
        """
        self.bytes_received += G.nbytes
        G_t = self.apply_transform(G)
        if self.transform_mode in ("projection", "dp+projection"):
            # Cannot apply projected vector back to model directly in this simulation;
            # record the aggregate norm for research tracking.
            return
        update = torch.tensor(G_t / n_active, dtype=torch.float32).to(self.device)
        apply_update(self.model, update)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Return (accuracy, avg_loss) on test set."""
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X, y in test_loader:
                X, y    = X.to(self.device), y.to(self.device)
                logits  = self.model(X)
                loss    = criterion(logits, y).item()
                preds   = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
                total_loss += loss * y.size(0)
        return correct / total, total_loss / total

    def cleanup(self) -> None:
        self.bytes_received = 0

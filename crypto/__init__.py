"""
crypto/__init__.py

All cryptographic primitives for the secure aggregation protocol.

Modules:
  ECDHKeyExchange      — Simulated ECDH for pairwise seed derivation
  PRG                  — Counter-mode HMAC-SHA256 mask expansion
  ShamirSecretSharing  — GF(256) threshold secret sharing (g=3)
  MaskEngine           — High-level mask generation orchestrator

Implementation note:
  This is a research simulation.  The ECDH is deterministic (reproducible)
  and the GF arithmetic is pure Python.  For deployment, use libsodium or
  a constant-time C extension.
"""

from __future__ import annotations
import hashlib
import hmac as _hmac
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# ECDH Key Exchange (Simulated)
# =============================================================================

class ECDHKeyPair:
    """Deterministic key pair for (vehicle_id, round_id) tuple."""

    def __init__(self, vehicle_id: str, round_id: bytes):
        self._private = hashlib.sha256(
            vehicle_id.encode() + b":" + round_id
        ).digest()
        self.public = hashlib.sha256(b"pub:" + self._private).digest()

    @property
    def private(self) -> bytes:
        return self._private


class ECDHKeyExchange:
    """
    Simulated ECDH: commutative shared secret from private + peer public.
    shared_seed(sk_i, pk_j) == shared_seed(sk_j, pk_i) by XOR commutativity.
    """

    def shared_seed(self, private_key: bytes, peer_public: bytes) -> bytes:
        h1 = _hmac.new(private_key,   peer_public,  hashlib.sha256).digest()
        h2 = _hmac.new(peer_public,   private_key,  hashlib.sha256).digest()
        return bytes(a ^ b for a, b in zip(h1, h2))

    def generate_keypair(self, vehicle_id: str, round_id: bytes) -> ECDHKeyPair:
        return ECDHKeyPair(vehicle_id, round_id)


# =============================================================================
# Pseudorandom Generator (Counter-mode HMAC-SHA256)
# =============================================================================

class PRG:
    """
    Expands a seed into a float64 mask vector of arbitrary dimension.
    Includes round_id in key derivation for forward secrecy across rounds.
    """

    def expand(self, seed: bytes, dim: int, round_id: bytes = b"") -> np.ndarray:
        out = bytearray()
        counter = 0
        key = seed + round_id
        while len(out) < dim * 8:
            out += hashlib.sha256(key + struct.pack(">Q", counter)).digest()
            counter += 1
        raw = np.frombuffer(bytes(out[:dim * 8]), dtype=np.int64).astype(np.float64)
        return raw / (2 ** 63)   # normalise to (-1, 1)

    def pairwise_mask(
        self, seed: bytes, rid: bytes, dim: int, *, negate: bool = False
    ) -> np.ndarray:
        mask = self.expand(seed, dim, rid)
        return -mask if negate else mask

    def self_mask(self, seed: bytes, rid: bytes, dim: int) -> np.ndarray:
        return self.expand(seed, dim, rid)


# =============================================================================
# GF(256) Shamir Secret Sharing  (primitive element g=3, polynomial 0x11b)
# =============================================================================

def _gf_mul_naive(a: int, b: int, poly: int = 0x11b) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= poly
        b >>= 1
    return r


# Build log/exp tables with g=3 (generates all 255 non-zero GF(256) elements)
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x]  = _i
    _x = _gf_mul_naive(_x, 3)
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[(_LOG[a] + _LOG[b]) % 255]


def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]


def _poly_eval(coeffs: List[int], x: int) -> int:
    result = 0
    for c in reversed(coeffs):
        result = _gf_mul(result, x) ^ c
    return result


def _lagrange_at_zero(points: List[Tuple[int, int]]) -> int:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    result = 0
    for i in range(len(xs)):
        num, den = 1, 1
        for j in range(len(xs)):
            if i == j:
                continue
            num = _gf_mul(num, xs[j])
            den = _gf_mul(den, xs[i] ^ xs[j])
        result ^= _gf_mul(ys[i], _gf_div(num, den))
    return result


class ShamirSecretSharing:
    """
    (threshold, n_shares) Shamir secret sharing over GF(256).
    Secret is a byte string; each byte is shared independently.
    """

    def __init__(self, threshold: int = 2, n_shares: int = 3):
        if threshold < 2 or n_shares < threshold:
            raise ValueError("Need 2 ≤ threshold ≤ n_shares")
        self.threshold = threshold
        self.n_shares  = n_shares

    def split(self, secret: bytes) -> List[Tuple[int, bytes]]:
        per_byte: List[List[int]] = []
        for byte_val in secret:
            coeffs = [byte_val] + [
                int.from_bytes(os.urandom(1), "big")
                for _ in range(self.threshold - 1)
            ]
            per_byte.append([_poly_eval(coeffs, x) for x in range(1, self.n_shares + 1)])
        return [
            (idx + 1, bytes(per_byte[b][idx] for b in range(len(secret))))
            for idx in range(self.n_shares)
        ]

    def reconstruct(self, shares: List[Tuple[int, bytes]]) -> bytes:
        if len(shares) < self.threshold:
            raise ValueError(f"Need {self.threshold} shares, got {len(shares)}")
        chosen  = shares[:self.threshold]
        n_bytes = len(chosen[0][1])
        return bytes(
            _lagrange_at_zero([(idx, share[b]) for idx, share in chosen])
            for b in range(n_bytes)
        )


# =============================================================================
# MaskEngine — high-level orchestrator for Phase 2
# =============================================================================

class MaskEngine:
    """
    Generates all pairwise and self masks for one FL round.

    Fix 3 applied: pairwise masking is zone-local only.
    Complexity: O(m²) per RSU zone, not O(N²) federation-wide.
    """

    def __init__(
        self,
        model_dim:        int,
        shamir_threshold: int = 2,
        shamir_n_shares:  int = 3,
    ):
        self._dim    = model_dim
        self._kex    = ECDHKeyExchange()
        self._prg    = PRG()
        self._shamir = ShamirSecretSharing(shamir_threshold, shamir_n_shares)

    # ── Pairwise mask generation (zone-local) ──────────────────────────────────

    def generate_zone_masks(
        self,
        zone_vehicles: List[str],
        rid: bytes,
    ) -> Dict[str, Dict[str, Tuple[bytes, np.ndarray]]]:
        """
        For every ordered pair (vi, vj) in the same RSU zone, derive:
          - shared seed s_ij
          - pairwise mask m_ij   (m_ij = -m_ji enforced by canonical sign rule)

        Returns: {vi: {vj: (seed_ij, mask_ij)}}
        """
        keypairs = {v: self._kex.generate_keypair(v, rid) for v in zone_vehicles}
        result: Dict[str, Dict[str, Tuple[bytes, np.ndarray]]] = {
            v: {} for v in zone_vehicles
        }
        for vi in zone_vehicles:
            for vj in zone_vehicles:
                if vi == vj:
                    continue
                seed   = self._kex.shared_seed(keypairs[vi].private, keypairs[vj].public)
                negate = vi > vj
                mask   = self._prg.pairwise_mask(seed, rid, self._dim, negate=negate)
                result[vi][vj] = (seed, mask)
        return result

    # ── Self-mask generation ───────────────────────────────────────────────────

    def generate_self_masks(
        self, vehicles: List[str], rid: bytes
    ) -> Dict[str, Tuple[bytes, np.ndarray]]:
        """Returns {vi: (self_seed, self_mask_bi)}"""
        result: Dict[str, Tuple[bytes, np.ndarray]] = {}
        for v in vehicles:
            seed = os.urandom(32)
            mask = self._prg.self_mask(seed, rid, self._dim)
            result[v] = (seed, mask)
        return result

    # ── Recovery share creation ────────────────────────────────────────────────

    def create_recovery_shares(
        self,
        vehicle_id:    str,
        self_seed:     bytes,
        pairwise_data: Dict[str, Tuple[bytes, np.ndarray]],
    ) -> dict:
        """
        Shamir-split the self_seed and each pairwise seed.
        Returns a dict suitable for depositing with the RSU.
        """
        self_shares = self._shamir.split(self_seed)
        pairwise_shares = {
            vj: self._shamir.split(seed)
            for vj, (seed, _) in pairwise_data.items()
        }
        return {
            "vehicle_id":      vehicle_id,
            "self_shares":     self_shares,
            "pairwise_shares": pairwise_shares,
        }

    # ── Mask reconstruction for dropped vehicles ───────────────────────────────

    def reconstruct_dropped_masks(
        self,
        dropped_id:    str,
        active_ids:    List[str],
        share_bundle:  dict,
        rid:           bytes,
    ) -> Dict[str, np.ndarray]:
        """
        For Vk ∈ Dt and each Vj ∈ At, reconstruct seed_kj → mask_kj.
        """
        masks: Dict[str, np.ndarray] = {}
        for vj in active_ids:
            if vj not in share_bundle["pairwise_shares"]:
                continue
            shares  = share_bundle["pairwise_shares"][vj]
            seed_kj = self._shamir.reconstruct(shares)
            negate  = dropped_id > vj
            masks[vj] = self._prg.pairwise_mask(seed_kj, rid, self._dim, negate=negate)
        return masks

    # ── Self-mask reconstruction from shares ───────────────────────────────────

    def reconstruct_self_mask(
        self,
        vehicle_share:   Tuple[int, bytes],
        rsu_share:       Tuple[int, bytes],
        rid:             bytes,
    ) -> np.ndarray:
        """Reconstruct bi from two shares of self_seed (Fix 1: Bonawitz-correct)."""
        seed = self._shamir.reconstruct([vehicle_share, rsu_share])
        return self._prg.self_mask(seed, rid, self._dim)

    @property
    def prg(self) -> PRG:
        return self._prg

    @property
    def shamir(self) -> ShamirSecretSharing:
        return self._shamir

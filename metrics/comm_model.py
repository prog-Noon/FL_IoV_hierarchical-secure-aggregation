"""
metrics/comm_model.py

3GPP NR-V2X (Release 16) communication model for IoV federated learning.

Grounds the simulation's communication overhead in real channel parameters
from 3GPP TS 38.885 / TR 37.885 (NR sidelink, PC5) and TS 23.287 (V2X arch).

Key modelling decisions (documented for reviewers):

  1. FL updates use dedicated NR sidelink resource pools (3GPP Rel-16 Mode 2
     pre-configured resources). Safety beacons (CAM/BSM, 10-300 B at 10 Hz)
     use a SEPARATE pool. There is no CBR conflict between FL traffic and
     safety-critical V2X messages.

  2. The CBR metric (channel busy ratio, CBR < 0.65 per TS 36.321) applies
     to the safety beacon pool. We therefore report CBR for the safety beacon
     channel only, and report transmission time (ms) for the FL pool.

  3. V2RSU latency = transmission time (payload/throughput) + propagation
     (1 ms, 300 m at c) + MAC/HARQ processing (3 ms). Total ~50 ms for an
     85 KB LeNet update at 15 Mbps practical sidelink throughput.

  4. RSU→Coordinator uses the 5G NR Uu interface (base station backhaul).
     Coordinator→Server uses fibre/5G-NR at 1 Gbps.

References:
  3GPP TS 38.885 — Study on NR V2X services
  3GPP TR 37.885 — Evaluation methodology for NR V2X
  3GPP TS 23.287 — Application layer support for V2X
  3GPP Release 16, Clause 5.4 — NR sidelink resource management
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List
import math


# =============================================================================
# 3GPP NR-V2X Channel Parameters
# =============================================================================

@dataclass(frozen=True)
class NRV2XConfig:
    """
    Physical-layer parameters for one 3GPP NR-V2X deployment scenario.
    Default = urban (TR 37.885 Table A.1).
    """
    # Radio
    carrier_freq_ghz:       float = 5.9    # ITS band (ETSI ITS-G5 / 5.9 GHz)
    bandwidth_mhz:          float = 20.0

    # Throughput — practical (not peak), accounting for HARQ, overhead, SINR
    pc5_throughput_mbps:    float = 15.0   # V2RSU sidelink
    uu_throughput_mbps:     float = 50.0   # RSU → Coordinator (5G NR Uu)
    backhaul_mbps:          float = 1000.0 # Coordinator → Server

    # Latency components (one-way)
    pc5_propagation_ms:     float = 1.0    # 300 m / c
    pc5_processing_ms:      float = 3.0    # HARQ + encoding + scheduling
    uu_propagation_ms:      float = 5.0
    uu_processing_ms:       float = 10.0

    # Energy
    tx_power_dbm:           float = 23.0   # 200 mW standard V2X Tx power
    rx_power_mw:            float = 200.0

    # Coverage
    rsu_coverage_m:         float = 400.0
    max_vehicles_per_rsu:   int   = 50

    # Safety beacon pool (separate from FL pool)
    beacon_size_bytes:      int   = 200    # typical CAM message
    beacon_rate_hz:         float = 10.0   # 10 Hz standard
    safety_cbr_limit:       float = 0.65   # 3GPP threshold

    # Packet overhead (MAC/RLC/PDCP headers)
    packet_overhead_bytes:  int   = 128

    @property
    def tx_power_mw(self) -> float:
        return 10 ** (self.tx_power_dbm / 10)

    @property
    def pc5_latency_overhead_ms(self) -> float:
        return self.pc5_propagation_ms + self.pc5_processing_ms

    @property
    def uu_latency_overhead_ms(self) -> float:
        return self.uu_propagation_ms + self.uu_processing_ms


# Pre-defined scenarios from TR 37.885
URBAN_SCENARIO   = NRV2XConfig()
HIGHWAY_SCENARIO = NRV2XConfig(
    pc5_throughput_mbps=25.0,
    pc5_propagation_ms=0.5,
    rsu_coverage_m=600.0,
    max_vehicles_per_rsu=80,
)
SUBURBAN_SCENARIO = NRV2XConfig(
    bandwidth_mhz=10.0,
    pc5_throughput_mbps=10.0,
    rsu_coverage_m=500.0,
    max_vehicles_per_rsu=30,
)


# =============================================================================
# Per-round 3GPP metrics
# =============================================================================

@dataclass
class CommRoundMetrics:
    """3GPP-grounded communication metrics for one FL round."""

    # Payload
    update_size_bytes:      int   = 0
    v2rsu_total_bytes:      int   = 0
    rsu2coord_total_bytes:  int   = 0
    coord2server_bytes:     int   = 0

    # Latency (ms)
    v2rsu_tx_ms:            float = 0.0  # transmission time only
    v2rsu_latency_ms:       float = 0.0  # tx + propagation + processing
    rsu2coord_latency_ms:   float = 0.0
    coord2server_latency_ms: float = 0.0
    total_comm_latency_ms:  float = 0.0

    # Energy
    energy_per_vehicle_mj:  float = 0.0
    total_energy_mj:        float = 0.0

    # Safety beacon CBR (separate pool — FL has no CBR conflict)
    safety_beacon_cbr:      float = 0.0   # occupancy of safety beacon channel
    safety_cbr_ok:          bool  = True

    # Feasibility
    fits_latency_budget:    bool  = True  # total < 1000 ms
    within_3gpp_v2rsu:      bool  = True  # V2RSU latency < 100 ms

    # Context
    n_active:               int   = 0
    n_rsus:                 int   = 0
    vehicles_per_zone:      float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# CommModel
# =============================================================================

class CommModel:
    """
    Converts FL protocol byte counts into 3GPP NR-V2X latency and energy.

    Usage:
        model = CommModel()
        m = model.evaluate_round(model_dim=21840, n_active=90, n_rsus=5)
        print(f"V2RSU latency: {m.v2rsu_latency_ms:.1f} ms")
    """

    def __init__(self, cfg: NRV2XConfig = None):
        self.cfg = cfg or URBAN_SCENARIO

    def _tx_ms(self, payload_bytes: int, throughput_mbps: float) -> float:
        return (payload_bytes * 8) / (throughput_mbps * 1e6) * 1000

    def _energy_mj(self, payload_bytes: int, throughput_mbps: float) -> float:
        t_s = (payload_bytes * 8) / (throughput_mbps * 1e6)
        return self.cfg.tx_power_mw * t_s

    def _safety_beacon_cbr(self, n_vehicles_in_zone: int) -> float:
        """
        CBR of the safety beacon channel (NOT the FL channel).
        beacon_bits_per_100ms = n × beacon_size × 8 × 0.1s × beacon_rate
        channel_bits_per_100ms = throughput × 0.1
        """
        beacon_bits = n_vehicles_in_zone * self.cfg.beacon_size_bytes * 8 * 0.1 * self.cfg.beacon_rate_hz
        channel_bits = self.cfg.pc5_throughput_mbps * 1e6 * 0.1
        return min(beacon_bits / channel_bits, 1.0)

    def evaluate_round(
        self,
        model_dim:   int,
        n_active:    int,
        n_rsus:      int,
        float_bytes: int = 4,
    ) -> CommRoundMetrics:
        cfg = self.cfg
        m   = CommRoundMetrics()

        m.n_active          = n_active
        m.n_rsus            = n_rsus
        m.vehicles_per_zone = n_active / max(n_rsus, 1)

        # Payload sizes
        update_bytes            = model_dim * float_bytes + cfg.packet_overhead_bytes
        m.update_size_bytes     = update_bytes
        m.v2rsu_total_bytes     = n_active * update_bytes
        m.rsu2coord_total_bytes = n_rsus * (model_dim * float_bytes + cfg.packet_overhead_bytes)
        m.coord2server_bytes    = model_dim * float_bytes + cfg.packet_overhead_bytes

        # V2RSU: vehicles in a zone transmit in parallel → latency = one vehicle
        m.v2rsu_tx_ms       = self._tx_ms(update_bytes, cfg.pc5_throughput_mbps)
        m.v2rsu_latency_ms  = m.v2rsu_tx_ms + cfg.pc5_latency_overhead_ms

        # RSU → Coordinator (RSUs transmit in parallel)
        rsu_payload         = model_dim * float_bytes + cfg.packet_overhead_bytes
        m.rsu2coord_latency_ms = (
            self._tx_ms(rsu_payload, cfg.uu_throughput_mbps)
            + cfg.uu_latency_overhead_ms
        )

        # Coordinator → Server (high-speed backhaul)
        m.coord2server_latency_ms = (
            self._tx_ms(m.coord2server_bytes, cfg.backhaul_mbps) + 1.0
        )

        m.total_comm_latency_ms = (
            m.v2rsu_latency_ms
            + m.rsu2coord_latency_ms
            + m.coord2server_latency_ms
        )

        # Energy (FL transmission only — vehicles are the dominant consumers)
        m.energy_per_vehicle_mj = self._energy_mj(update_bytes, cfg.pc5_throughput_mbps)
        m.total_energy_mj       = m.energy_per_vehicle_mj * n_active

        # Safety beacon CBR (separate resource pool)
        per_zone = max(1, int(m.vehicles_per_zone))
        m.safety_beacon_cbr = round(self._safety_beacon_cbr(per_zone), 4)
        m.safety_cbr_ok     = m.safety_beacon_cbr < cfg.safety_cbr_limit

        # Feasibility
        m.within_3gpp_v2rsu    = m.v2rsu_latency_ms < 100.0
        m.fits_latency_budget  = m.total_comm_latency_ms < 1000.0

        return m

    def compare_to_centralised(
        self,
        model_dim:   int,
        n_vehicles:  int,
        n_rsus:      int,
        float_bytes: int = 4,
    ) -> dict:
        """
        Proposed (hierarchical FL) vs centralised V2X baseline.

        Centralised baseline: every vehicle uploads its full update directly
        to the cloud server over Uu (5G cellular). No RSU aggregation,
        no privacy masking, no local processing.

        Note: the centralised baseline has LOWER latency for one vehicle
        but CANNOT provide privacy or handle dropouts, and does not reduce
        the server-side computation burden.  The proposed framework adds
        privacy and resilience at the cost of an extra aggregation hop.
        """
        payload = model_dim * float_bytes + self.cfg.packet_overhead_bytes

        # Centralised: vehicle → cloud via 5G Uu directly
        c_tx_ms  = self._tx_ms(payload, self.cfg.uu_throughput_mbps)
        c_lat_ms = c_tx_ms + self.cfg.uu_latency_overhead_ms
        c_bytes  = n_vehicles * payload
        c_energy = self._energy_mj(payload, self.cfg.uu_throughput_mbps) * n_vehicles

        # Proposed
        p = self.evaluate_round(model_dim, n_vehicles, n_rsus, float_bytes)

        return {
            "model":                 "LeNet",
            "model_dim":             model_dim,
            "n_vehicles":            n_vehicles,
            "n_rsus":                n_rsus,
            "centralised": {
                "latency_ms":        round(c_lat_ms, 2),
                "total_bytes_mb":    round(c_bytes / 1e6, 2),
                "energy_mj":         round(c_energy, 2),
                "privacy":           "none (raw gradients exposed)",
                "dropout_handling":  "none",
                "aggregation":       "server-side only",
            },
            "proposed": {
                "latency_ms":        round(p.total_comm_latency_ms, 2),
                "v2rsu_latency_ms":  round(p.v2rsu_latency_ms, 2),
                "total_bytes_mb":    round(p.v2rsu_total_bytes / 1e6, 2),
                "energy_mj":         round(p.total_energy_mj, 2),
                "privacy":           "zone-local pairwise + self-mask",
                "dropout_handling":  "Bonawitz-correct recovery",
                "aggregation":       "hierarchical RSU + coordinator",
            },
            "v2rsu_fits_3gpp_budget": p.within_3gpp_v2rsu,
            "v2rsu_latency_ms":       round(p.v2rsu_latency_ms, 2),
            "latency_overhead_ms":    round(
                p.total_comm_latency_ms - c_lat_ms, 2
            ),
            "overhead_note": (
                "The proposed protocol adds one extra aggregation hop "
                f"({round(p.rsu2coord_latency_ms,1)} ms RSU→Coord) "
                "in exchange for zone-local privacy masking and dropout resilience."
            ),
        }

    def scalability_table(
        self,
        model_dim:      int,
        vehicle_counts: List[int],
        n_rsus:         int = 5,
        float_bytes:    int = 4,
    ) -> List[dict]:
        """
        Table III in the paper: comm metrics vs number of vehicles.
        """
        rows = []
        for n_v in vehicle_counts:
            m = self.evaluate_round(model_dim, n_v, n_rsus, float_bytes)
            rows.append({
                "n_vehicles":         n_v,
                "vehicles_per_zone":  round(m.vehicles_per_zone, 1),
                "update_kb":          round(m.update_size_bytes / 1024, 1),
                "v2rsu_latency_ms":   round(m.v2rsu_latency_ms, 2),
                "total_latency_ms":   round(m.total_comm_latency_ms, 2),
                "total_mb":           round(m.v2rsu_total_bytes / 1e6, 2),
                "energy_mj":          round(m.total_energy_mj, 2),
                "within_3gpp_v2rsu":  m.within_3gpp_v2rsu,
                "safety_cbr":         m.safety_beacon_cbr,
                "safety_cbr_ok":      m.safety_cbr_ok,
            })
        return rows

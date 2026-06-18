#!/usr/bin/env python3
"""
Feature extractor for dual-stage LSTM-AE runtime.

Runtime stores union of 19 features needed by all supported variants. Active
model ordering still comes from metadata-selected features.

Argus does not expose CICFlowMeter bulk/subflow semantics directly, so live
pipeline uses stateful approximations based on repeated flow updates:
- bulk features estimated from packet/byte deltas grouped by short gaps
- subflow forward bytes estimated from forward-byte deltas separated by idle gaps
- Flow IAT Mean approximated from duration / (total packets - 1)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from feature_schema import FEATURE_KEYS, FEATURE_SET_VERSION, ordered_feature_values

logger = logging.getLogger("feature_extractor")

ARGUS_FIELDS = [
    "stime",
    "ltime",
    "saddr",
    "daddr",
    "sport",
    "dport",
    "proto",
    "dur",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "flgs",
    "state",
    "sttl",
    "dttl",
    "smeansz",
    "dmeansz",
    "sminsz",
    "dminsz",
    "smaxsz",
    "dmaxsz",
    "sintpkt",
    "dintpkt",
    "sjit",
    "djit",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sappbytes",
    "dappbytes",
    "swin",
    "dwin",
]


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        if val is None or str(val).strip() in ("", "-", "*"):
            return default
        return float(str(val).strip())
    except (TypeError, ValueError):
        return default


def safe_int(val: str, default: int = 0) -> int:
    try:
        if val is None or str(val).strip() in ("", "-", "*"):
            return default
        return int(float(str(val).strip()))
    except (TypeError, ValueError):
        return default


def parse_protocol(proto_str: str) -> str:
    if not proto_str:
        return "unknown"

    proto = proto_str.strip().lower()
    if proto in ("tcp", "6"):
        return "tcp"
    if proto in ("udp", "17"):
        return "udp"
    if proto in ("icmp", "1"):
        return "icmp"
    return proto


@dataclass
class BulkAccumulator:
    bulk_count: int = 0
    total_packets: int = 0
    total_bytes: float = 0.0
    total_duration: float = 0.0
    candidate_packets: int = 0
    candidate_bytes: float = 0.0
    candidate_start: float = 0.0
    candidate_end: float = 0.0
    candidate_updates: int = 0


@dataclass
class FlowState:
    last_stime: float
    last_ltime: float
    last_spkts: int
    last_dpkts: int
    last_sbytes: float
    last_dbytes: float
    last_sappbytes: float
    last_dappbytes: float
    initial_fwd_window: float
    subflow_count: int = 1
    archived_subflow_fwd_bytes: float = 0.0
    current_subflow_fwd_bytes: float = 0.0
    fwd_bulk: BulkAccumulator = field(default_factory=BulkAccumulator)
    bwd_bulk: BulkAccumulator = field(default_factory=BulkAccumulator)


class FlowTracker:
    """Track flow-local state for bulk and subflow approximations."""

    def __init__(self):
        self.flow_states: Dict[str, FlowState] = {}
        self.bulk_gap_threshold = 1.0
        self.bulk_packet_threshold = 4
        self.subflow_gap_threshold = 1.0
        self.bulk_duration_floor = 1.0
        self.max_bulk_packets_per_average = 256.0
        self.max_bulk_bytes_per_average = 65535.0
        self.max_bulk_rate_average = 65535.0

    def get_flow_key(self, row: Dict) -> str:
        return "-".join(
            [
                row.get("saddr", ""),
                row.get("daddr", ""),
                row.get("sport", ""),
                row.get("dport", ""),
                row.get("proto", ""),
            ]
        )

    def _new_state(
        self,
        stime: float,
        ltime: float,
        spkts: int,
        dpkts: int,
        sbytes: float,
        dbytes: float,
        sappbytes: float,
        dappbytes: float,
        swin: float,
    ) -> FlowState:
        return FlowState(
            last_stime=stime,
            last_ltime=ltime,
            last_spkts=spkts,
            last_dpkts=dpkts,
            last_sbytes=sbytes,
            last_dbytes=dbytes,
            last_sappbytes=sappbytes,
            last_dappbytes=dappbytes,
            initial_fwd_window=self._normalize_tcp_window(swin),
            current_subflow_fwd_bytes=max(sbytes, 0.0),
        )

    def _normalize_tcp_window(self, window_value: float) -> float:
        return min(max(window_value, 0.0), 65535.0)

    def _looks_like_new_flow(
        self,
        state: FlowState,
        stime: float,
        ltime: float,
        spkts: int,
        dpkts: int,
        sbytes: float,
        dbytes: float,
    ) -> bool:
        if stime > state.last_ltime + 1e-6:
            return True

        return any(
            [
                ltime < state.last_ltime,
                spkts < state.last_spkts,
                dpkts < state.last_dpkts,
                sbytes < state.last_sbytes,
                dbytes < state.last_dbytes,
            ]
        )

    def _flush_bulk_candidate(self, accumulator: BulkAccumulator):
        if (
            accumulator.candidate_packets >= self.bulk_packet_threshold
            and accumulator.candidate_updates >= 2
            and accumulator.candidate_bytes > 0
        ):
            duration = max(accumulator.candidate_end - accumulator.candidate_start, self.bulk_duration_floor)
            accumulator.bulk_count += 1
            accumulator.total_packets += accumulator.candidate_packets
            accumulator.total_bytes += accumulator.candidate_bytes
            accumulator.total_duration += duration

        accumulator.candidate_packets = 0
        accumulator.candidate_bytes = 0.0
        accumulator.candidate_start = 0.0
        accumulator.candidate_end = 0.0
        accumulator.candidate_updates = 0

    def _update_bulk(
        self,
        accumulator: BulkAccumulator,
        delta_packets: int,
        delta_payload_bytes: float,
        gap: float,
        previous_time: float,
        current_time: float,
    ):
        if accumulator.candidate_packets > 0 and gap > self.bulk_gap_threshold:
            self._flush_bulk_candidate(accumulator)

        if delta_packets <= 0 or delta_payload_bytes <= 0:
            return

        if accumulator.candidate_packets == 0:
            accumulator.candidate_start = previous_time if previous_time > 0 else current_time
            accumulator.candidate_end = current_time
        else:
            accumulator.candidate_end = current_time

        accumulator.candidate_packets += delta_packets
        accumulator.candidate_bytes += delta_payload_bytes
        accumulator.candidate_updates += 1

    def update(
        self,
        flow_key: str,
        stime: float,
        ltime: float,
        spkts: int,
        dpkts: int,
        sbytes: float,
        dbytes: float,
        sappbytes: float,
        dappbytes: float,
        swin: float,
        proto: str,
    ) -> FlowState:
        state = self.flow_states.get(flow_key)
        if state is None:
            state = self._new_state(stime, ltime, spkts, dpkts, sbytes, dbytes, sappbytes, dappbytes, swin)
            self.flow_states[flow_key] = state
            return state

        if self._looks_like_new_flow(state, stime, ltime, spkts, dpkts, sbytes, dbytes):
            self._flush_bulk_candidate(state.fwd_bulk)
            self._flush_bulk_candidate(state.bwd_bulk)
            state = self._new_state(stime, ltime, spkts, dpkts, sbytes, dbytes, sappbytes, dappbytes, swin)
            self.flow_states[flow_key] = state
            return state

        delta_spkts = max(0, spkts - state.last_spkts)
        delta_dpkts = max(0, dpkts - state.last_dpkts)
        delta_sbytes = max(0.0, sbytes - state.last_sbytes)
        delta_dbytes = max(0.0, dbytes - state.last_dbytes)
        delta_sappbytes = max(0.0, sappbytes - state.last_sappbytes)
        delta_dappbytes = max(0.0, dappbytes - state.last_dappbytes)
        gap = max(0.0, ltime - state.last_ltime)

        if swin > 0 and state.initial_fwd_window <= 0:
            state.initial_fwd_window = self._normalize_tcp_window(swin)

        if gap > self.subflow_gap_threshold:
            state.archived_subflow_fwd_bytes += state.current_subflow_fwd_bytes
            state.current_subflow_fwd_bytes = delta_sbytes
            if delta_spkts > 0 or delta_dpkts > 0 or delta_sbytes > 0 or delta_dbytes > 0:
                state.subflow_count += 1
        else:
            state.current_subflow_fwd_bytes += delta_sbytes

        if proto == "tcp":
            self._update_bulk(state.fwd_bulk, delta_spkts, delta_sappbytes, gap, state.last_ltime, ltime)
            self._update_bulk(state.bwd_bulk, delta_dpkts, delta_dappbytes, gap, state.last_ltime, ltime)

        state.last_stime = stime
        state.last_ltime = ltime
        state.last_spkts = spkts
        state.last_dpkts = dpkts
        state.last_sbytes = sbytes
        state.last_dbytes = dbytes
        state.last_sappbytes = sappbytes
        state.last_dappbytes = dappbytes
        return state

    def _bulk_snapshot(
        self,
        accumulator: BulkAccumulator,
        _fallback_packets: int,
        _fallback_bytes: float,
        _fallback_duration: float,
    ) -> Dict[str, float]:
        bulk_count = accumulator.bulk_count
        total_packets = accumulator.total_packets
        total_bytes = accumulator.total_bytes
        total_duration = accumulator.total_duration

        if (
            accumulator.candidate_packets >= self.bulk_packet_threshold
            and accumulator.candidate_updates >= 2
            and accumulator.candidate_bytes > 0
        ):
            bulk_count += 1
            total_packets += accumulator.candidate_packets
            total_bytes += accumulator.candidate_bytes
            total_duration += max(accumulator.candidate_end - accumulator.candidate_start, self.bulk_duration_floor)

        if bulk_count == 0:
            return {
                "bulk_count": 0.0,
                "pkts_per_bulk_avg": 0.0,
                "byts_per_bulk_avg": 0.0,
                "blk_rate_avg": 0.0,
            }

        pkts_per_bulk_avg = min(float(total_packets) / bulk_count, self.max_bulk_packets_per_average)
        byts_per_bulk_avg = min(float(total_bytes) / bulk_count, self.max_bulk_bytes_per_average)
        blk_rate_avg = min(float(total_bytes) / max(total_duration, self.bulk_duration_floor), self.max_bulk_rate_average)

        return {
            "bulk_count": float(bulk_count),
            "pkts_per_bulk_avg": pkts_per_bulk_avg,
            "byts_per_bulk_avg": byts_per_bulk_avg,
            "blk_rate_avg": blk_rate_avg,
        }

    def get_subflow_fwd_byts(self, state: Optional[FlowState], fallback_sbytes: float) -> float:
        if state is None:
            return fallback_sbytes

        total_subflow_bytes = state.archived_subflow_fwd_bytes + state.current_subflow_fwd_bytes
        subflow_count = max(1, state.subflow_count)
        return total_subflow_bytes / subflow_count if total_subflow_bytes > 0 else 0.0

    def get_forward_bulk_metrics(self, state: Optional[FlowState], spkts: int, sbytes: float, duration: float) -> Dict[str, float]:
        if state is None:
            return self._bulk_snapshot(BulkAccumulator(), spkts, sbytes, duration)
        return self._bulk_snapshot(state.fwd_bulk, spkts, sbytes, duration)

    def get_backward_bulk_metrics(self, state: Optional[FlowState], dpkts: int, dbytes: float, duration: float) -> Dict[str, float]:
        if state is None:
            return self._bulk_snapshot(BulkAccumulator(), dpkts, dbytes, duration)
        return self._bulk_snapshot(state.bwd_bulk, dpkts, dbytes, duration)

    def cleanup_old(self, max_age: float = 300.0):
        now = time.time()
        expired_keys = []
        for key, state in self.flow_states.items():
            if (now - state.last_ltime) > max_age:
                self._flush_bulk_candidate(state.fwd_bulk)
                self._flush_bulk_candidate(state.bwd_bulk)
                expired_keys.append(key)

        for key in expired_keys:
            del self.flow_states[key]


def extract_features_from_row(row: Dict, flow_tracker: FlowTracker) -> Optional[Dict]:
    try:
        stime = safe_float(row.get("stime", "0"))
        ltime = safe_float(row.get("ltime", "0"))
        duration = safe_float(row.get("dur", "0"))

        src_ip = row.get("saddr", "").strip()
        dst_ip = row.get("daddr", "").strip()
        sport = safe_int(row.get("sport", "0"))
        dport = safe_int(row.get("dport", "0"))
        proto = parse_protocol(row.get("proto", ""))
        if proto == "man":
            return None

        sbytes = safe_float(row.get("sbytes", "0"))
        dbytes = safe_float(row.get("dbytes", "0"))
        spkts = safe_int(row.get("spkts", "0"))
        dpkts = safe_int(row.get("dpkts", "0"))
        total_pkts = spkts + dpkts

        sappbytes = safe_float(row.get("sappbytes", "0"))
        dappbytes = safe_float(row.get("dappbytes", "0"))
        swin = safe_float(row.get("swin", "0"))
        dwin = safe_float(row.get("dwin", "0"))

        flow_key = flow_tracker.get_flow_key(row)
        state = flow_tracker.update(flow_key, stime, ltime, spkts, dpkts, sbytes, dbytes, sappbytes, dappbytes, swin, proto)

        fwd_bulk_metrics = flow_tracker.get_forward_bulk_metrics(state, spkts, sbytes, duration)
        bwd_bulk_metrics = flow_tracker.get_backward_bulk_metrics(state, dpkts, dbytes, duration)

        flow_iat_mean = duration / (total_pkts - 1) if total_pkts > 1 and duration > 0 else 0.0
        fwd_header_len = max(0.0, sbytes - max(sappbytes, 0.0))
        total_bytes = sbytes + dbytes
        pkt_len_mean = total_bytes / total_pkts if total_pkts > 0 else 0.0

        features = {
            "dst_port_feat": float(dport),
            "flow_duration": duration,
            "flow_byts_per_s": (total_bytes / duration) if duration > 0 else 0.0,
            "fwd_header_len": fwd_header_len,
            "fwd_pkt_len_mean": (sbytes / spkts) if spkts > 0 else 0.0,
            "fwd_pkts_per_s": (spkts / duration) if duration > 0 else 0.0,
            "fwd_seg_size_min": safe_float(row.get("sminsz", "0")),
            "bwd_pkts_b_avg": bwd_bulk_metrics["pkts_per_bulk_avg"],
            "init_fwd_win_byts": max(state.initial_fwd_window if state else swin, 0.0),
            "init_bwd_win_byts": max(min(dwin, 65535.0), 0.0),
            "flow_iat_mean": flow_iat_mean,
            "flow_iat_max": max(safe_float(row.get("sintpkt", "0")), safe_float(row.get("dintpkt", "0")), flow_iat_mean),
            "bwd_iat_std": safe_float(row.get("djit", "0")),
            "bwd_blk_rate_avg": bwd_bulk_metrics["blk_rate_avg"],
            "bwd_byts_b_avg": bwd_bulk_metrics["byts_per_bulk_avg"],
            "bwd_pkts_per_s": (dpkts / duration) if duration > 0 else 0.0,
            "pkt_len_mean": pkt_len_mean,
            "subflow_fwd_byts": flow_tracker.get_subflow_fwd_byts(state, sbytes),
            "fwd_byts_b_avg": fwd_bulk_metrics["byts_per_bulk_avg"],
        }

        metadata = {
            "timestamp": stime,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": sport,
            "dst_port_raw": dport,
            "protocol": proto.upper(),
            "duration": duration,
            "total_bytes": sbytes + dbytes,
            "total_packets": total_pkts,
            "fwd_bytes": sbytes,
            "bwd_bytes": dbytes,
            "fwd_packets": spkts,
            "bwd_packets": dpkts,
            "state": row.get("state", "").strip(),
            "feature_set_version": FEATURE_SET_VERSION,
        }

        return {"features": features, "metadata": metadata}
    except Exception as e:
        logger.warning("Feature extraction error: %s", e)
        return None


def process_csv_stream(input_stream, callback=None):
    flow_tracker = FlowTracker()
    record_count = 0
    cleanup_interval = 100

    reader = csv.DictReader(input_stream, fieldnames=ARGUS_FIELDS)
    for row in reader:
        if not row or not row.get("stime"):
            continue
        if row.get("stime", "").strip().startswith("StartTime"):
            continue

        result = extract_features_from_row(row, flow_tracker)
        if result:
            record_count += 1
            if callback:
                callback(result)
            else:
                yield result

            if record_count % cleanup_interval == 0:
                flow_tracker.cleanup_old()


def process_csv_line(line: str, flow_tracker: FlowTracker) -> Optional[Dict]:
    if not line or not line.strip():
        return None

    reader = csv.DictReader(StringIO(line), fieldnames=ARGUS_FIELDS)
    for row in reader:
        if not row or not row.get("stime"):
            return None
        if row.get("stime", "").strip().startswith("StartTime"):
            return None
        return extract_features_from_row(row, flow_tracker)
    return None


def get_feature_vector(features_dict: Dict, selected_features=None) -> List[float]:
    return ordered_feature_values(features_dict, selected_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dual-stage live features from Argus flows")
    parser.add_argument("--input", "-i", default="-", help="Input CSV file or pipe (default: stdin)")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    input_stream = sys.stdin if args.input == "-" else open(args.input, "r")
    output_stream = sys.stdout if args.output == "-" else open(args.output, "w")

    try:
        meta_fields = [
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_port",
            "protocol",
            "duration",
            "total_bytes",
            "total_packets",
            "feature_set_version",
        ]

        if args.format == "csv":
            header = ",".join(meta_fields + FEATURE_KEYS)
            output_stream.write(header + "\n")
            output_stream.flush()

        count = 0
        for result in process_csv_stream(input_stream):
            features = result["features"]
            metadata = result["metadata"]

            if args.format == "csv":
                meta_vals = [
                    str(metadata["timestamp"]),
                    metadata["src_ip"],
                    metadata["dst_ip"],
                    str(metadata["src_port"]),
                    metadata["protocol"],
                    f"{metadata['duration']:.6f}",
                    f"{metadata['total_bytes']:.0f}",
                    str(metadata["total_packets"]),
                    metadata["feature_set_version"],
                ]
                feat_vals = [f"{features[key]:.6f}" for key in FEATURE_KEYS]
                output_stream.write(",".join(meta_vals + feat_vals) + "\n")
            else:
                output_stream.write(json.dumps({**metadata, **features}) + "\n")

            output_stream.flush()
            count += 1
            if count % 100 == 0:
                logger.info("Processed %s flows", count)
    except KeyboardInterrupt:
        logger.info("Stopped. Total flows processed: %s", count)
    finally:
        if args.input != "-":
            input_stream.close()
        if args.output != "-":
            output_stream.close()

#!/usr/bin/env python3
"""
CICFlowMeter CSV adapter for dual-stage LSTM-AE runtime.

CICFlowMeter already emits completed bidirectional flow rows, matching the
CSE-CICIDS2018 training source. Runtime only maps those columns into the shared
19-feature contract used by all model variants.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from feature_schema import FEATURE_KEYS, FEATURE_SET_VERSION, ordered_feature_values

logger = logging.getLogger("feature_extractor")


def safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or str(val).strip() in ("", "-", "*"):
            return default
        value = float(str(val).strip())
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def safe_int(val, default: int = 0) -> int:
    try:
        if val is None or str(val).strip() in ("", "-", "*"):
            return default
        return int(float(str(val).strip()))
    except (TypeError, ValueError):
        return default


def normalize_column(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def normalized_row(row: Dict) -> Dict[str, str]:
    return {normalize_column(key): value for key, value in row.items()}


def first_value(row: Dict[str, str], aliases: List[str], default=""):
    normalized = normalized_row(row)
    for alias in aliases:
        value = normalized.get(normalize_column(alias))
        if value not in (None, ""):
            return value
    return default


FEATURE_COLUMN_ALIASES = {
    "dst_port_feat": ["Dst Port", "Destination Port", "dst_port"],
    "flow_duration": ["Flow Duration", "flow_duration"],
    "flow_byts_per_s": ["Flow Byts/s", "Flow Bytes/s", "flow_byts_s"],
    "fwd_header_len": ["Fwd Header Len", "Fwd Header Length", "fwd_header_len"],
    "fwd_pkt_len_mean": ["Fwd Pkt Len Mean", "Fwd Packet Length Mean", "fwd_pkt_len_mean"],
    "fwd_pkts_per_s": ["Fwd Pkts/s", "Fwd Packets/s", "fwd_pkts_s"],
    "fwd_seg_size_min": ["Fwd Seg Size Min", "Min Seg Size Forward", "fwd_seg_size_min"],
    "bwd_pkts_b_avg": ["Bwd Pkts/b Avg", "Bwd Packets/Bulk Avg", "Bwd Avg Packets/Bulk", "bwd_pkts_b_avg"],
    "init_fwd_win_byts": ["Init Fwd Win Byts", "Init_Win_bytes_forward", "init_fwd_win_byts"],
    "init_bwd_win_byts": ["Init Bwd Win Byts", "Init_Win_bytes_backward", "init_bwd_win_byts"],
    "flow_iat_mean": ["Flow IAT Mean", "flow_iat_mean"],
    "flow_iat_max": ["Flow IAT Max", "flow_iat_max"],
    "bwd_iat_std": ["Bwd IAT Std", "bwd_iat_std"],
    "bwd_blk_rate_avg": ["Bwd Blk Rate Avg", "Bwd Bulk Rate Avg", "Bwd Avg Bulk Rate", "bwd_blk_rate_avg"],
    "bwd_byts_b_avg": ["Bwd Byts/b Avg", "Bwd Bytes/Bulk Avg", "Bwd Avg Bytes/Bulk", "bwd_byts_b_avg"],
    "bwd_pkts_per_s": ["Bwd Pkts/s", "Bwd Packets/s", "bwd_pkts_s"],
    "pkt_len_mean": ["Pkt Len Mean", "Packet Length Mean", "pkt_len_mean"],
    "subflow_fwd_byts": ["Subflow Fwd Byts", "Subflow Fwd Bytes", "subflow_fwd_byts"],
    "fwd_byts_b_avg": ["Fwd Byts/b Avg", "Fwd Bytes/Bulk Avg", "Fwd Avg Bytes/Bulk", "fwd_byts_b_avg"],
}

META_ALIASES = {
    "timestamp": ["Timestamp", "StartTime", "Start Time", "timestamp"],
    "src_ip": ["Src IP", "Source IP", "src_ip"],
    "dst_ip": ["Dst IP", "Destination IP", "dst_ip"],
    "src_port": ["Src Port", "Source Port", "src_port"],
    "dst_port": ["Dst Port", "Destination Port", "dst_port"],
    "protocol": ["Protocol", "protocol"],
    "fwd_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets", "totlen_fwd_pkts"],
    "bwd_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets", "totlen_bwd_pkts"],
    "fwd_packets": ["Tot Fwd Pkts", "Total Fwd Packets", "tot_fwd_pkts"],
    "bwd_packets": ["Tot Bwd Pkts", "Total Backward Packets", "tot_bwd_pkts"],
}

TIME_FEATURE_KEYS = {"flow_duration", "flow_iat_mean", "flow_iat_max", "bwd_iat_std"}

PYTHON_CICFLOWMETER_FIELDS = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "timestamp",
    "flow_duration",
    "flow_byts_s",
    "fwd_header_len",
    "fwd_pkt_len_mean",
    "fwd_pkts_s",
    "fwd_seg_size_min",
    "bwd_pkts_b_avg",
    "init_fwd_win_byts",
    "init_bwd_win_byts",
    "flow_iat_mean",
    "flow_iat_max",
    "bwd_iat_std",
    "bwd_blk_rate_avg",
    "bwd_byts_b_avg",
    "bwd_pkts_s",
    "pkt_len_mean",
    "subflow_fwd_byts",
    "fwd_byts_b_avg",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
]


def is_python_cicflowmeter_row(row: Dict) -> bool:
    return any(str(key).strip().lower() == "flow_duration" for key in row)


def parse_timestamp(value) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0

    numeric = safe_float(text, None)
    if numeric is not None:
        return numeric

    for fmt in (
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    return 0.0


def parse_protocol(value) -> str:
    proto = str(value or "").strip().lower()
    proto_num = safe_int(proto, None)
    if proto in ("tcp",) or proto_num == 6:
        return "TCP"
    if proto in ("udp",) or proto_num == 17:
        return "UDP"
    if proto in ("icmp",) or proto_num == 1:
        return "ICMP"
    return proto.upper() if proto else "UNKNOWN"


def extract_features_from_row(row: Dict) -> Optional[Dict]:
    try:
        features = {
            key: safe_float(first_value(row, aliases))
            for key, aliases in FEATURE_COLUMN_ALIASES.items()
        }
        python_cicflowmeter = is_python_cicflowmeter_row(row)
        if python_cicflowmeter:
            for key in TIME_FEATURE_KEYS:
                features[key] *= 1_000_000.0
        for key in FEATURE_KEYS:
            features.setdefault(key, 0.0)

        flow_duration = features["flow_duration"]
        fwd_bytes = safe_float(first_value(row, META_ALIASES["fwd_bytes"]))
        bwd_bytes = safe_float(first_value(row, META_ALIASES["bwd_bytes"]))
        fwd_packets = safe_int(first_value(row, META_ALIASES["fwd_packets"]))
        bwd_packets = safe_int(first_value(row, META_ALIASES["bwd_packets"]))

        metadata = {
            "timestamp": parse_timestamp(first_value(row, META_ALIASES["timestamp"])),
            "src_ip": str(first_value(row, META_ALIASES["src_ip"])),
            "dst_ip": str(first_value(row, META_ALIASES["dst_ip"])),
            "src_port": safe_int(first_value(row, META_ALIASES["src_port"])),
            "dst_port_raw": safe_int(first_value(row, META_ALIASES["dst_port"], features["dst_port_feat"])),
            "protocol": parse_protocol(first_value(row, META_ALIASES["protocol"])),
            "duration": flow_duration / 1_000_000.0,
            "total_bytes": fwd_bytes + bwd_bytes,
            "total_packets": fwd_packets + bwd_packets,
            "fwd_bytes": fwd_bytes,
            "bwd_bytes": bwd_bytes,
            "fwd_packets": fwd_packets,
            "bwd_packets": bwd_packets,
            "state": "CICFlowMeter",
            "feature_set_version": FEATURE_SET_VERSION,
        }

        return {"features": features, "metadata": metadata}
    except Exception as e:
        logger.warning("CICFlowMeter feature extraction error: %s", e)
        return None


def read_new_cicflowmeter_rows(
    path: str,
    offset: int = 0,
    header: Optional[List[str]] = None,
) -> Tuple[List[Dict], int, Optional[List[str]]]:
    if not os.path.exists(path):
        return [], offset, header

    rows = []
    with open(path, "r", newline="") as f:
        f.seek(offset)
        while True:
            before = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith("\n"):
                f.seek(before)
                break

            stripped = line.strip()
            if not stripped:
                continue

            values = next(csv.reader([line]))
            if header is None:
                header = values
                continue
            if [normalize_column(value) for value in values] == [normalize_column(value) for value in header]:
                continue

            rows.append(dict(zip(header, values)))
        offset = f.tell()

    return rows, offset, header


def process_csv_stream(input_stream, callback=None):
    reader = csv.DictReader(input_stream)
    for row in reader:
        result = extract_features_from_row(row)
        if not result:
            continue
        if callback:
            callback(result)
        else:
            yield result


def get_feature_vector(features_dict: Dict, selected_features=None) -> List[float]:
    return ordered_feature_values(features_dict, selected_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dual-stage live features from CICFlowMeter CSV")
    parser.add_argument("--input", "-i", default="-", help="Input CSV file (default: stdin)")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")
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
            output_stream.write(",".join(meta_fields + FEATURE_KEYS) + "\n")

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
                logger.info("Processed %s CICFlowMeter flows", count)
    finally:
        if args.input != "-":
            input_stream.close()
        if args.output != "-":
            output_stream.close()

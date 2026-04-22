#!/usr/bin/env python3
"""
Feature Extractor - Argus Flow Records to CSE-CICIDS-2018 Features

Reads argus CSV flow records and extracts 10 features compatible with
the CSE-CICIDS-2018 dataset format for LSTM-AE anomaly detection.

10 Features:
  1. Dst Port           - Destination port number
  2. Fwd Pkt Len Min    - Minimum forward packet length
  3. Flow Pkts/s        - Total packets per second in the flow
  4. Bwd Pkts/s         - Backward packets per second
  5. Fwd IAT Min        - Minimum forward inter-arrival time
  6. ECE Flag Cnt       - ECE flag count from TCP flags
  7. ACK Flag Cnt       - ACK flag count from TCP flags
  8. Fwd Seg Size Min   - Minimum forward segment size
  9. Fwd Act Data Pkts  - Forward packets with payload > 0
  10. Idle Std          - Std deviation of idle times
"""

import sys
import csv
import argparse
import re
import time
import json
import logging
from io import StringIO
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# =================================================================
# Argus CSV Header Mapping
# =================================================================
ARGUS_FIELDS = [
    "stime", "ltime", "saddr", "daddr", "sport", "dport", "proto",
    "dur", "sbytes", "dbytes", "spkts", "dpkts", "flgs", "state",
    "sttl", "dttl", "smeansz", "dmeansz", "sminsz", "dminsz",
    "smaxsz", "dmaxsz", "sintpkt", "dintpkt", "sjit", "djit",
    "sload", "dload", "sloss", "dloss"
]

logger = logging.getLogger("feature_extractor")


def safe_float(val: str, default: float = 0.0) -> float:
    """Safely convert string to float."""
    try:
        if val is None or val.strip() in ("", "-", "*"):
            return default
        return float(val.strip())
    except (ValueError, AttributeError):
        return default


def safe_int(val: str, default: int = 0) -> int:
    """Safely convert string to int."""
    try:
        if val is None or val.strip() in ("", "-", "*"):
            return default
        return int(float(val.strip()))
    except (ValueError, AttributeError):
        return default


def parse_tcp_flags(flags_str: str) -> Dict[str, int]:
    """
    Parse Argus TCP flags string to individual flag counts.
    
    Argus flags format examples:
      'e*sA' = ECE + SYN + ACK (from source/dest directions)
      ' rR'  = RST
      's'    = SYN
      'A'    = ACK
      
    CSE-CICIDS-2018 flag mapping:
      - ECE Flag Cnt: count of 'e' or 'E' (ECE/ECN-Echo)
      - ACK Flag Cnt: count of 'A' or 'a' (ACK)
    """
    flags = {
        "ece": 0,
        "ack": 0,
        "syn": 0,
        "fin": 0,
        "rst": 0,
        "psh": 0,
        "urg": 0,
    }
    if not flags_str or flags_str.strip() in ("", "-", "*"):
        return flags

    f = flags_str.strip()
    # Argus uses: e=ECE, s/S=SYN, a/A=ACK, f/F=FIN, r/R=RST, p/P=PSH, u/U=URG
    flags["ece"] = sum(1 for c in f if c in ("e", "E"))
    flags["ack"] = sum(1 for c in f if c in ("a", "A"))
    flags["syn"] = sum(1 for c in f if c in ("s", "S"))
    flags["fin"] = sum(1 for c in f if c in ("f", "F"))
    flags["rst"] = sum(1 for c in f if c in ("r", "R"))
    flags["psh"] = sum(1 for c in f if c in ("p", "P"))
    flags["urg"] = sum(1 for c in f if c in ("u", "U"))

    return flags


def parse_protocol(proto_str: str) -> str:
    """Parse protocol string (tcp/udp/icmp/etc)."""
    if not proto_str:
        return "unknown"
    p = proto_str.strip().lower()
    if p in ("tcp", "6"):
        return "tcp"
    elif p in ("udp", "17"):
        return "udp"
    elif p in ("icmp", "1"):
        return "icmp"
    return p


class FlowTracker:
    """
    Track flow state untuk menghitung inter-arrival times dan idle times.
    
    Setiap flow diidentifikasi oleh (src_ip, dst_ip, src_port, dst_port, proto).
    Menyimpan timestamp packet terakhir untuk menghitung IAT dan idle time.
    """

    def __init__(self):
        self.flows: Dict[str, List[float]] = defaultdict(list)
        self.idle_times: Dict[str, List[float]] = defaultdict(list)
        self.fwd_timestamps: Dict[str, List[float]] = defaultdict(list)
        self.idle_threshold = 1.0  # seconds - threshold for idle detection

    def get_flow_key(self, row: Dict) -> str:
        """Generate unique flow key."""
        return f"{row.get('saddr','')}-{row.get('daddr','')}-{row.get('sport','')}-{row.get('dport','')}-{row.get('proto','')}"

    def update(self, flow_key: str, stime: float, ltime: float, duration: float):
        """Update flow tracker with new flow record."""
        self.flows[flow_key].append(stime)

        # Track forward inter-arrival times
        if len(self.flows[flow_key]) > 1:
            iat = stime - self.flows[flow_key][-2]
            self.fwd_timestamps[flow_key].append(iat)

            # Track idle times (IAT > threshold)
            if iat > self.idle_threshold:
                self.idle_times[flow_key].append(iat)

    def get_fwd_iat_min(self, flow_key: str) -> float:
        """Get minimum forward inter-arrival time."""
        iats = self.fwd_timestamps.get(flow_key, [])
        return min(iats) if iats else 0.0

    def get_idle_std(self, flow_key: str) -> float:
        """Get standard deviation of idle times."""
        idles = self.idle_times.get(flow_key, [])
        if len(idles) < 2:
            return 0.0
        return float(np.std(idles))

    def cleanup_old(self, max_age: float = 300.0):
        """Remove flows older than max_age seconds."""
        now = time.time()
        to_remove = []
        for key, timestamps in self.flows.items():
            if timestamps and (now - timestamps[-1]) > max_age:
                to_remove.append(key)
        for key in to_remove:
            del self.flows[key]
            self.fwd_timestamps.pop(key, None)
            self.idle_times.pop(key, None)


def extract_features_from_row(row: Dict, flow_tracker: FlowTracker) -> Optional[Dict]:
    """
    Extract 10 CSE-CICIDS-2018 features from a single argus flow record.
    
    Returns dict with:
        - 10 numeric features for LSTM-AE
        - metadata (IPs, ports, protocol, timestamp) for dashboard
    """
    try:
        # --- Parse raw values ---
        stime = safe_float(row.get("stime", "0"))
        ltime = safe_float(row.get("ltime", "0"))
        duration = safe_float(row.get("dur", "0"))
        
        src_ip = row.get("saddr", "").strip()
        dst_ip = row.get("daddr", "").strip()
        sport = safe_int(row.get("sport", "0"))
        dport = safe_int(row.get("dport", "0"))
        proto = parse_protocol(row.get("proto", ""))

        sbytes = safe_float(row.get("sbytes", "0"))
        dbytes = safe_float(row.get("dbytes", "0"))
        spkts = safe_int(row.get("spkts", "0"))
        dpkts = safe_int(row.get("dpkts", "0"))

        sminsz = safe_float(row.get("sminsz", "0"))
        dminsz = safe_float(row.get("dminsz", "0"))
        smeansz = safe_float(row.get("smeansz", "0"))

        sintpkt = safe_float(row.get("sintpkt", "0"))  # microseconds
        dintpkt = safe_float(row.get("dintpkt", "0"))

        flags_str = row.get("flgs", "")
        flags = parse_tcp_flags(flags_str)

        # --- Flow tracking for IAT and idle ---
        flow_key = flow_tracker.get_flow_key(row)
        flow_tracker.update(flow_key, stime, ltime, duration)

        # ===========================================================
        # CSE-CICIDS-2018 Feature Mapping
        # ===========================================================

        # 1. Dst Port - Destination port number
        dst_port = float(dport)

        # 2. Fwd Pkt Len Min - Minimum length of a packet in forward direction
        #    Mapped from argus sminsz (source min packet size)
        fwd_pkt_len_min = sminsz

        # 3. Flow Pkts/s - Number of packets per second
        #    Total packets / duration
        total_pkts = spkts + dpkts
        flow_pkts_per_s = total_pkts / duration if duration > 0 else 0.0

        # 4. Bwd Pkts/s - Backward packets per second
        #    dpkts / duration (destination = backward direction)
        bwd_pkts_per_s = dpkts / duration if duration > 0 else 0.0

        # 5. Fwd IAT Min - Minimum inter-arrival time (forward direction)
        #    From flow tracker history; fallback to argus sintpkt
        fwd_iat_min = flow_tracker.get_fwd_iat_min(flow_key)
        if fwd_iat_min == 0.0:
            # Fallback: use argus source inter-packet time (microseconds -> seconds)
            fwd_iat_min = sintpkt / 1e6 if sintpkt > 0 else 0.0

        # 6. ECE Flag Cnt - ECE flag count
        ece_flag_cnt = float(flags["ece"])

        # 7. ACK Flag Cnt - ACK flag count
        ack_flag_cnt = float(flags["ack"])

        # 8. Fwd Seg Size Min - Minimum segment size (forward)
        #    Similar to Fwd Pkt Len Min but at segment level
        #    In CICIDS this is typically 20 (TCP header) or actual min
        fwd_seg_size_min = sminsz if sminsz > 0 else (20.0 if proto == "tcp" else 8.0)

        # 9. Fwd Act Data Pkts - Forward packets with payload > 0
        #    Estimated: if mean size > header size, most packets have data
        #    spkts - estimated header-only packets
        if spkts > 0 and smeansz > 0:
            # Estimate: packets with data = total - (packets that are just headers)
            min_header = 20.0 if proto == "tcp" else 8.0
            if sminsz > min_header:
                fwd_act_data_pkts = float(spkts)  # All packets have data
            else:
                # Ratio of data packets based on mean vs min
                data_ratio = min(1.0, smeansz / max(sminsz + 1, 1))
                fwd_act_data_pkts = float(int(spkts * data_ratio))
        else:
            fwd_act_data_pkts = 0.0

        # 10. Idle Std - Standard deviation of idle times
        idle_std = flow_tracker.get_idle_std(flow_key)

        # --- Build feature vector ---
        features = {
            "dst_port": dst_port,
            "fwd_pkt_len_min": fwd_pkt_len_min,
            "flow_pkts_per_s": flow_pkts_per_s,
            "bwd_pkts_per_s": bwd_pkts_per_s,
            "fwd_iat_min": fwd_iat_min,
            "ece_flag_cnt": ece_flag_cnt,
            "ack_flag_cnt": ack_flag_cnt,
            "fwd_seg_size_min": fwd_seg_size_min,
            "fwd_act_data_pkts": fwd_act_data_pkts,
            "idle_std": idle_std,
        }

        # --- Metadata for dashboard ---
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
            "flags": flags_str.strip(),
            "state": row.get("state", "").strip(),
        }

        return {"features": features, "metadata": metadata}

    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
        return None


def process_csv_stream(input_stream, callback=None):
    """
    Process streaming CSV from argus (ra output).
    Yields feature dicts for each flow record.
    """
    flow_tracker = FlowTracker()
    record_count = 0
    cleanup_interval = 100  # Cleanup every N records

    reader = csv.DictReader(input_stream, fieldnames=ARGUS_FIELDS)

    for row in reader:
        # Skip empty or header rows
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

            # Periodic cleanup of old flows
            if record_count % cleanup_interval == 0:
                flow_tracker.cleanup_old()


def process_csv_line(line: str, flow_tracker: FlowTracker) -> Optional[Dict]:
    """Process a single CSV line. Used by pipeline.py for real-time processing."""
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


def get_feature_vector(features_dict: Dict) -> List[float]:
    """Convert features dict to ordered numpy-compatible list."""
    feature_order = [
        "dst_port", "fwd_pkt_len_min", "flow_pkts_per_s", "bwd_pkts_per_s",
        "fwd_iat_min", "ece_flag_cnt", "ack_flag_cnt", "fwd_seg_size_min",
        "fwd_act_data_pkts", "idle_std"
    ]
    return [features_dict[f] for f in feature_order]


# =================================================================
# CLI Mode - Read from file/pipe directly
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CSE-CICIDS-2018 features from Argus flows")
    parser.add_argument("--input", "-i", default="-", help="Input CSV file or pipe (default: stdin)")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Open input
    if args.input == "-":
        input_stream = sys.stdin
    else:
        input_stream = open(args.input, "r")

    # Open output
    if args.output == "-":
        output_stream = sys.stdout
    else:
        output_stream = open(args.output, "w")

    try:
        feature_names = [
            "dst_port", "fwd_pkt_len_min", "flow_pkts_per_s", "bwd_pkts_per_s",
            "fwd_iat_min", "ece_flag_cnt", "ack_flag_cnt", "fwd_seg_size_min",
            "fwd_act_data_pkts", "idle_std"
        ]

        if args.format == "csv":
            # Print CSV header
            meta_fields = ["timestamp", "src_ip", "dst_ip", "src_port", "protocol", "duration", "total_bytes", "total_packets"]
            header = ",".join(meta_fields + feature_names)
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
                    f"{metadata['duration']:.4f}",
                    f"{metadata['total_bytes']:.0f}",
                    str(metadata["total_packets"]),
                ]
                feat_vals = [f"{features[f]:.6f}" for f in feature_names]
                line = ",".join(meta_vals + feat_vals)
                output_stream.write(line + "\n")
            else:
                output_stream.write(json.dumps({**metadata, **features}) + "\n")

            output_stream.flush()
            count += 1

            if count % 100 == 0:
                logger.info(f"Processed {count} flows")

    except KeyboardInterrupt:
        logger.info(f"Stopped. Total flows processed: {count}")
    finally:
        if args.input != "-":
            input_stream.close()
        if args.output != "-":
            output_stream.close()
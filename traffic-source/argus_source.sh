#!/usr/bin/env bash
set -euo pipefail

INTERFACE="${CAPTURE_INTERFACE:-eth0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

echo "[ARGUS] Starting capture on $INTERFACE"

cleanup() {
    echo "[ARGUS] Stopping..."
    kill "$PIPE_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

FIELDS="stime,ltime,saddr,daddr,sport,dport,proto,dur,sbytes,dbytes,spkts,dpkts,flgs,state,sttl,dttl,smeansz,dmeansz,sminsz,dminsz,smaxsz,dmaxsz,sintpkt,dintpkt,sjit,djit,sload,dload,sloss,dloss"

# Stream langsung ke stdout untuk dibaca main.py
sudo argus -i "$INTERFACE" -w - | ra -n -c ',' -s "$FIELDS" -u
PIPE_PID=$!

wait "$PIPE_PID"
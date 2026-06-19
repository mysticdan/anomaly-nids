#!/usr/bin/env python3
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT_DIR = os.path.dirname(__file__)
LSTM_AE_DIR = os.path.join(ROOT_DIR, "lstm-ae")
TRAFFIC_SRC_DIR = os.path.join(ROOT_DIR, "traffic-source")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if LSTM_AE_DIR not in sys.path:
    sys.path.insert(0, LSTM_AE_DIR)
if TRAFFIC_SRC_DIR not in sys.path:
    sys.path.insert(0, TRAFFIC_SRC_DIR)

from feature_schema import VARIANT_FEATURES, feature_keys_for_names, normalize_feature_names
from model import DualStageAutoencoder, LSTMAEService, dual_stage_errors
from extract_feature import extract_features_from_row, get_feature_vector, read_new_cicflowmeter_rows


def config_for_variant(variant):
    artifact_dir = os.path.join(LSTM_AE_DIR, "dual-stage-ae", "artifacts", variant)
    return {
        "model": {
            "device": os.getenv("MODEL_DEVICE", "cpu"),
            "model_path": os.path.join(artifact_dir, "model.pt"),
            "scaler_path": os.path.join(artifact_dir, "scaler.pkl"),
            "metadata_path": os.path.join(artifact_dir, "metadata.json"),
            "dropout": 0.2,
        },
        "detection": {
            "score_multiplier": 100,
            "default_threshold": 1.0,
            "threshold_percentile": 95,
        },
    }


def check_dual_stage_shapes():
    model = DualStageAutoencoder(10, 10)
    x = torch.randn(2, 10, 10)
    recon1, recon_residual = model(x)
    error, feature_errors = dual_stage_errors(model, x)
    assert recon1.shape == x.shape
    assert recon_residual.shape == x.shape
    assert error.shape == (2,)
    assert feature_errors.shape == (2, 10)


def check_variant_metadata():
    for variant in VARIANT_FEATURES:
        metadata_path = config_for_variant(variant)["model"]["metadata_path"]
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        selected_features = normalize_feature_names(metadata["selected_features"])
        assert selected_features == normalize_feature_names(VARIANT_FEATURES[variant])
        assert len(feature_keys_for_names(selected_features)) == len(selected_features)


def check_service_smoke():
    for variant in VARIANT_FEATURES:
        service = LSTMAEService(config_for_variant(variant))
        dummy = np.zeros((service.seq_len, service.input_dim), dtype=np.float32)
        scaled = service.scale_sequence(dummy)
        reconstruction_error, anomaly_score, feature_errors = service.predict(scaled)
        assert np.isfinite(reconstruction_error)
        assert np.isfinite(anomaly_score)
        assert feature_errors.shape == (service.input_dim,)


def check_cicflowmeter_mapping():
    row = {
        "Timestamp": "01/03/2018 08:00:00",
        "Src IP": "10.0.0.1",
        "Dst IP": "10.0.0.2",
        "Src Port": "1234",
        "Dst Port": "80",
        "Protocol": "6",
        "Flow Duration": "2000000",
        "TotLen Fwd Pkts": "120",
        "TotLen Bwd Pkts": "60",
        "Tot Fwd Pkts": "2",
        "Tot Bwd Pkts": "1",
        "Flow Byts/s": "90",
        "Fwd Header Len": "40",
        "Fwd Pkt Len Mean": "60",
        "Fwd Pkts/s": "1",
        "Fwd Seg Size Min": "20",
        "Bwd Pkts/b Avg": "1",
        "Init Fwd Win Byts": "65535",
        "Init Bwd Win Byts": "65535",
        "Flow IAT Mean": "1000000",
        "Flow IAT Max": "1500000",
        "Bwd IAT Std": "0",
        "Bwd Blk Rate Avg": "0",
        "Bwd Byts/b Avg": "60",
        "Bwd Pkts/s": "0.5",
        "Pkt Len Mean": "60",
        "Subflow Fwd Byts": "120",
        "Fwd Byts/b Avg": "120",
    }
    result = extract_features_from_row(row)
    assert result is not None
    assert result["features"]["flow_duration"] == 2000000
    assert result["metadata"]["duration"] == 2.0
    assert result["metadata"]["protocol"] == "TCP"
    assert len(get_feature_vector(result["features"], VARIANT_FEATURES["mrmr"])) == 10

    py_row = {
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": "1234",
        "dst_port": "80",
        "protocol": "6",
        "timestamp": "2026-06-19 13:00:00",
        "flow_duration": "2",
        "flow_iat_mean": "1",
        "flow_iat_max": "1.5",
        "bwd_iat_std": "0.25",
        "totlen_fwd_pkts": "120",
        "totlen_bwd_pkts": "60",
        "tot_fwd_pkts": "2",
        "tot_bwd_pkts": "1",
    }
    py_result = extract_features_from_row(py_row)
    assert py_result["features"]["flow_duration"] == 2000000
    assert py_result["features"]["flow_iat_mean"] == 1000000
    assert py_result["metadata"]["duration"] == 2.0

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        path = f.name
        f.write(",".join(row.keys()) + "\n")
        f.write(",".join(row.values()) + "\n")
    try:
        rows, offset, header = read_new_cicflowmeter_rows(path)
        assert len(rows) == 1
        rows, _, _ = read_new_cicflowmeter_rows(path, offset, header)
        assert rows == []
    finally:
        os.remove(path)


def main():
    check_dual_stage_shapes()
    check_variant_metadata()
    check_service_smoke()
    check_cicflowmeter_mapping()
    print("dual-stage smoke ok")


if __name__ == "__main__":
    main()

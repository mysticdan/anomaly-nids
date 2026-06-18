#!/usr/bin/env python3
import json
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(__file__)
LSTM_AE_DIR = os.path.join(ROOT_DIR, "lstm-ae")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if LSTM_AE_DIR not in sys.path:
    sys.path.insert(0, LSTM_AE_DIR)

from feature_schema import VARIANT_FEATURES, feature_keys_for_names, normalize_feature_names
from model import DualStageAutoencoder, LSTMAEService, dual_stage_errors


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


def main():
    check_dual_stage_shapes()
    check_variant_metadata()
    check_service_smoke()
    print("dual-stage smoke ok")


if __name__ == "__main__":
    main()

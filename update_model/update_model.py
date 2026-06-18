import json
import logging
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import database as db
from state import state

logger = logging.getLogger("model_updater")


def build_sequences(flow_vectors, seq_len: int) -> np.ndarray:
    if len(flow_vectors) < seq_len:
        return np.empty((0, seq_len, 0), dtype=np.float32)

    sequences = []
    for i in range(len(flow_vectors) - seq_len + 1):
        sequences.append(flow_vectors[i : i + seq_len])
    return np.asarray(sequences, dtype=np.float32)


def update_model_once(lstm_service, config: Optional[Dict] = None) -> Dict:
    config = config or {}
    min_normal_flows = int(config.get("min_normal_flows", 100))
    batch_limit = int(config.get("batch_limit", 1000))
    epochs = int(config.get("epochs", 1))
    learning_rate = float(config.get("learning_rate", 1e-4))
    adapt_scaler = bool(config.get("adapt_scaler", True))
    threshold_percentile = float(config.get("threshold_percentile", lstm_service.threshold_percentile))

    normal_flows = db.get_normal_flows(limit=batch_limit, selected_features=lstm_service.selected_features)
    if len(normal_flows) < max(min_normal_flows, lstm_service.seq_len):
        raise RuntimeError(
            "Not enough normal flows for update_model: "
            f"need at least {max(min_normal_flows, lstm_service.seq_len)}, got {len(normal_flows)}"
        )

    raw_features = np.asarray(normal_flows, dtype=np.float32)
    if adapt_scaler:
        lstm_service.adapt_scaler(raw_features)

    sequences = build_sequences(raw_features, lstm_service.seq_len)
    if len(sequences) == 0:
        raise RuntimeError("No training sequences generated for update_model")

    scaled_sequences = lstm_service.scale_sequence_batch(sequences)
    train_loss = float(lstm_service.train_incremental(scaled_sequences, epochs=epochs, lr=learning_rate))
    new_threshold = float(lstm_service.recalibrate_threshold(scaled_sequences, threshold_percentile))
    errors = lstm_service.compute_reconstruction_errors(scaled_sequences)

    summary = {
        "normal_flows_used": int(len(normal_flows)),
        "training_sequences": int(len(sequences)),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "adapt_scaler": adapt_scaler,
        "threshold_percentile": threshold_percentile,
        "avg_train_loss": train_loss,
        "new_threshold": new_threshold,
        "avg_reconstruction_error": float(np.mean(errors)),
        "max_reconstruction_error": float(np.max(errors)),
    }
    logger.info("One-shot update_model complete: %s", summary)
    return summary


def update_model_worker(config=None):
    config = config or {}
    if not config.get("enabled", False):
        logger.info("Update-model worker is disabled")
        return

    interval_seconds = config.get("interval_seconds", 600)

    while True:
        try:
            if state.lstm_service is None:
                time.sleep(10)
                continue

            time.sleep(interval_seconds)
            logger.info("Triggering incremental learning for dual-stage model...")
            update_model_once(state.lstm_service, config)
        except Exception as e:
            logger.error("Update-model worker error: %s", e)


if __name__ == "__main__":
    from main import CONFIG

    sys.path.insert(0, os.path.join(ROOT_DIR, "lstm-ae"))
    from model import LSTMAEService

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    service = LSTMAEService(CONFIG)
    summary = update_model_once(service, CONFIG.get("update_model", {}))
    print(json.dumps(summary, indent=2))

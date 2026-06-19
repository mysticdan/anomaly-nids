import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

import database as db
from feature_schema import FEATURE_SET_VERSION, VARIANT_FEATURES
from state import state
from update_model.update_model import update_model_worker

TRAFFIC_SRC_DIR = os.path.join(os.path.dirname(__file__), "traffic-source")
sys.path.insert(0, TRAFFIC_SRC_DIR)
from extract_feature import (
    PYTHON_CICFLOWMETER_FIELDS,
    extract_features_from_row,
    get_feature_vector,
    read_new_cicflowmeter_rows,
)

LSTM_AE_DIR = os.path.join(os.path.dirname(__file__), "lstm-ae")
sys.path.insert(0, LSTM_AE_DIR)
from model import LSTMAEService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")
SAFE_INTERFACE_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

MODEL_VARIANT = os.getenv("MODEL_VARIANT", "mrmr")
if MODEL_VARIANT not in VARIANT_FEATURES:
    raise RuntimeError(f"Invalid MODEL_VARIANT={MODEL_VARIANT!r}. Choose one of: {', '.join(VARIANT_FEATURES)}")

MODEL_ARTIFACT_DIR = os.path.join(LSTM_AE_DIR, "dual-stage-ae", "artifacts", MODEL_VARIANT)

CONFIG = {
    "model": {
        "device": os.getenv("MODEL_DEVICE", "cpu"),
        "model_path": os.path.join(MODEL_ARTIFACT_DIR, "model.pt"),
        "scaler_path": os.path.join(MODEL_ARTIFACT_DIR, "scaler.pkl"),
        "metadata_path": os.path.join(MODEL_ARTIFACT_DIR, "metadata.json"),
        "dropout": 0.2,
    },
    "detection": {
        "score_multiplier": 100,
    },
    "update_model": {
        "enabled": os.getenv("ENABLE_UPDATE_MODEL", "1") == "1",
        "interval_seconds": int(os.getenv("UPDATE_MODEL_INTERVAL_SECONDS", "600")),
        "min_normal_flows": int(os.getenv("UPDATE_MODEL_MIN_NORMAL_FLOWS", "100")),
        "batch_limit": int(os.getenv("UPDATE_MODEL_BATCH_LIMIT", "1000")),
        "epochs": int(os.getenv("UPDATE_MODEL_EPOCHS", "1")),
        "learning_rate": float(os.getenv("UPDATE_MODEL_LEARNING_RATE", "0.0001")),
        "adapt_scaler": os.getenv("UPDATE_MODEL_ADAPT_SCALER", "1") == "1",
        "threshold_percentile": float(os.getenv("UPDATE_MODEL_THRESHOLD_PERCENTILE", "95")),
    },
    "learning_mode": {
        "enabled": os.getenv("ENABLE_LEARNING_MODE", "1") == "1",
        "duration_minutes": int(os.getenv("LEARNING_MODE_DURATION_MINUTES", "10")),
    },
}

running = True


class CaptureProcess:
    def __init__(self, proc):
        self.proc = proc

    def poll(self):
        return self.proc.poll()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def stderr_text(self):
        if not self.proc.stderr:
            return ""
        text = self.proc.stderr.read()
        return text.strip() if text else ""


def signal_handler(sig, frame):
    global running
    running = False
    logger.info("Shutting down...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def start_cicflowmeter_capture(interface="eth0", output_file="/tmp/anomaly-nids-cicflowmeter/flows.csv"):
    if not SAFE_INTERFACE_RE.fullmatch(interface):
        raise RuntimeError(f"Invalid CAPTURE_INTERFACE={interface!r}")

    cmd_template = os.getenv("CICFLOWMETER_CMD", "").strip()
    if cmd_template:
        cmd = shlex.split(cmd_template.format(interface=interface, output_file=output_file))
    else:
        cicflowmeter_runner = os.path.join(TRAFFIC_SRC_DIR, "run_cicflowmeter.py")
        if os.path.exists(cicflowmeter_runner):
            cmd = [
                sys.executable,
                cicflowmeter_runner,
                "-i",
                interface,
                "-o",
                output_file,
                "--fields",
                ",".join(PYTHON_CICFLOWMETER_FIELDS),
            ]
        else:
            cmd = []
    if not cmd:
        raise RuntimeError(
            "CICFlowMeter CLI not found. Install it in .venv with "
            "'.venv/bin/python -m pip install cicflowmeter' or set CICFLOWMETER_CMD."
        )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info("CICFlowMeter capture started on %s, output=%s", interface, output_file)
    return CaptureProcess(proc)


def run_pipeline():
    global running

    db.init_db()
    logger.info("Database ready")

    lstm_service = LSTMAEService(CONFIG)
    logger.info(
        "Dual-stage LSTM-AE ready: variant=%s feature_set=%s seq_len=%s threshold=%.6f features=%s",
        MODEL_VARIANT,
        FEATURE_SET_VERSION,
        lstm_service.seq_len,
        lstm_service.threshold,
        ", ".join(lstm_service.get_feature_names()),
    )
    state.lstm_service = lstm_service

    learning_mode_cfg = CONFIG["learning_mode"]
    if learning_mode_cfg["enabled"]:
        state.enable_learning_mode(learning_mode_cfg["duration_minutes"])
        logger.info(
            "Learning mode enabled at startup for %s minutes",
            learning_mode_cfg["duration_minutes"],
        )
    else:
        state.disable_learning_mode()
        logger.info("Learning mode startup is disabled")

    if CONFIG["update_model"]["enabled"]:
        worker_thread = threading.Thread(
            target=update_model_worker,
            kwargs={"config": CONFIG["update_model"]},
            daemon=True,
        )
        worker_thread.start()
        logger.info("Update-model worker enabled")
    else:
        logger.info("Update-model worker is disabled")

    interface = os.getenv("CAPTURE_INTERFACE", "eth0")
    output_file = os.getenv("CICFLOWMETER_OUTPUT_FILE", "/tmp/anomaly-nids-cicflowmeter/flows.csv")
    poll_seconds = float(os.getenv("CICFLOWMETER_POLL_SECONDS", "1"))
    proc = start_cicflowmeter_capture(interface, output_file)
    flow_count = 0
    csv_offset = 0
    csv_header = None

    time.sleep(2)
    if proc.poll() is not None:
        proc.stop()
        err = proc.stderr_text()
        logger.error("CICFlowMeter exited immediately. stderr: %s", err)
        return

    try:
        def store_flow(result):
            nonlocal flow_count
            features = result["features"]
            metadata = result["metadata"]
            feature_vector = get_feature_vector(features, lstm_service.selected_features)

            sequence = lstm_service.add_to_buffer(feature_vector)

            anomaly_score = 0.0
            is_anomaly = False
            if sequence is not None:
                reconstruction_error, anomaly_score, _ = lstm_service.predict(sequence)
                is_anomaly = lstm_service.is_anomaly(reconstruction_error)

            if state.learning_mode:
                if time.time() < state.learning_mode_until:
                    is_anomaly = False
                else:
                    state.disable_learning_mode()
                    logger.info("Learning mode ended.")

            db.insert_flow(metadata, features, anomaly_score, is_anomaly)
            flow_count += 1

            if flow_count % 50 == 0:
                logger.info("Stored %s CICFlowMeter flows", flow_count)

        while running:
            rows, csv_offset, csv_header = read_new_cicflowmeter_rows(output_file, csv_offset, csv_header)
            for row in rows:
                result = extract_features_from_row(row)
                if result:
                    store_flow(result)

            if proc.poll() is not None:
                break

            time.sleep(poll_seconds)
    except Exception as e:
        logger.error("Pipeline error: %s", e)
    finally:
        proc.stop()
        if flow_count == 0:
            err = proc.stderr_text()
            logger.error("No flows captured. CICFlowMeter stderr: %s", err)
        logger.info("Stopped. Total CICFlowMeter flows: %s", flow_count)


def start_dashboard():
    from dashboard.app import app

    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    logger.info("Dashboard started on http://localhost:5000")

    run_pipeline()

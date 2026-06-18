import logging
import os
import re
import signal
import shutil
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
from extract_feature import ARGUS_FIELDS, FlowTracker, get_feature_vector, process_csv_line

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
    def __init__(self, argus_proc, ra_proc):
        self.argus_proc = argus_proc
        self.ra_proc = ra_proc
        self.stdout = ra_proc.stdout

    def poll(self):
        for proc in (self.argus_proc, self.ra_proc):
            code = proc.poll()
            if code is not None:
                return code
        return None

    def stop(self):
        for proc in (self.ra_proc, self.argus_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (self.ra_proc, self.argus_proc):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    def stderr_text(self):
        parts = []
        for name, proc in (("argus", self.argus_proc), ("ra", self.ra_proc)):
            if proc.stderr:
                text = proc.stderr.read()
                if text:
                    parts.append(f"{name}: {text.strip()}")
        return "\n".join(parts)


def signal_handler(sig, frame):
    global running
    running = False
    logger.info("Shutting down...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def start_argus_capture(interface="eth0"):
    if not SAFE_INTERFACE_RE.fullmatch(interface):
        raise RuntimeError(f"Invalid CAPTURE_INTERFACE={interface!r}")

    binary_paths = {binary: shutil.which(binary) for binary in ("argus", "ra")}
    missing_bins = [binary for binary, path in binary_paths.items() if path is None]
    if missing_bins:
        raise RuntimeError(f"Missing required capture binaries: {', '.join(missing_bins)}")

    argus_path = binary_paths["argus"]
    if os.geteuid() != 0:
        getcap_path = shutil.which("getcap")
        cap_output = ""
        if getcap_path:
            try:
                result = subprocess.run(
                    [getcap_path, argus_path],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                cap_output = (result.stdout or "").strip()
            except Exception:
                cap_output = ""

        has_capture_caps = "cap_net_raw" in cap_output or "cap_net_admin" in cap_output
        if not has_capture_caps:
            raise RuntimeError(
                "Argus capture privileges are missing. Run the pipeline with sudo or grant "
                f"capture capabilities to {argus_path} (cap_net_raw, cap_net_admin)."
            )

    argus_proc = subprocess.Popen(
        [argus_path, "-X", "-A", "-S", "1", "-P", "0", "-i", interface, "-w", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if argus_proc.stdout is None:
        argus_proc.kill()
        raise RuntimeError("Argus stdout pipe unavailable")

    fields = ",".join(ARGUS_FIELDS)
    ra_proc = subprocess.Popen(
        [binary_paths["ra"], "-M", "noman", "-n", "-c", ",", "-s", fields, "-u"],
        stdin=argus_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    argus_proc.stdout.close()
    logger.info("Argus capture started on %s", interface)
    return CaptureProcess(argus_proc, ra_proc)


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
    proc = start_argus_capture(interface)
    flow_tracker = FlowTracker()
    flow_count = 0

    time.sleep(2)
    if proc.poll() is not None:
        proc.stop()
        err = proc.stderr_text()
        logger.error("Argus exited immediately. stderr: %s", err)
        return

    try:
        for line in proc.stdout:
            if not running:
                break

            line = line.strip()
            if not line:
                continue

            result = process_csv_line(line, flow_tracker)
            if not result:
                continue

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
                logger.info("Processed %s flows", flow_count)
    except Exception as e:
        logger.error("Pipeline error: %s", e)
    finally:
        proc.stop()
        if flow_count == 0:
            err = proc.stderr_text()
            logger.error("No flows captured. Argus stderr: %s", err)
        logger.info("Stopped. Total: %s flows", flow_count)


def start_dashboard():
    from dashboard.app import app

    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    logger.info("Dashboard started on http://localhost:5000")

    run_pipeline()

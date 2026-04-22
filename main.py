import sys
import os
import time
import logging
import subprocess
import threading
import signal

sys.path.insert(0, os.path.dirname(__file__))

TRAFFIC_SRC_DIR = os.path.join(os.path.dirname(__file__), "traffic-source")
sys.path.insert(0, TRAFFIC_SRC_DIR)
from extract_feature import process_csv_line, FlowTracker, get_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")

LSTM_AE_DIR = os.path.join(os.path.dirname(__file__), "lstm-ae")
sys.path.insert(0, LSTM_AE_DIR)
from model import LSTMAEService

import database as db

CONFIG = {
    "model": {
        "input_dim": 10,
        "hidden_dim": 64,
        "latent_dim": 32,
        "num_layers": 2,
        "dropout": 0.1,
        "device": "cpu",
        "model_path": os.path.join(LSTM_AE_DIR, "best_lstm_autoencoder.pth"),
        "scaler_path": os.path.join(LSTM_AE_DIR, "scaler.pkl"),
        "threshold_path": os.path.join(LSTM_AE_DIR, "threshold.json"),
    },
    "features": {"sequence_length": 50},
    "detection": {
        "default_threshold": 0.5,
        "score_multiplier": 100,
    },
}

running = True


def signal_handler(sig, frame):
    global running
    running = False
    logger.info("Shutting down...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def start_argus_capture(interface="eth0"):
    fields = "stime,ltime,saddr,daddr,sport,dport,proto,dur,sbytes,dbytes,spkts,dpkts,flgs,state,sttl,dttl,smeansz,dmeansz,sminsz,dminsz,smaxsz,dmaxsz,sintpkt,dintpkt,sjit,djit,sload,dload,sloss,dloss"
    cmd = f"argus -P 0 -i {interface} -w - | ra -n -c ',' -s {fields} -u"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    logger.info(f"Argus capture started on {interface}")
    return proc


def run_pipeline():
    global running

    db.init_db()
    logger.info("Database ready")

    lstm_service = LSTMAEService(CONFIG)
    logger.info("LSTM-AE service ready")

    interface = os.getenv("CAPTURE_INTERFACE", "eth0")
    proc = start_argus_capture(interface)
    flow_tracker = FlowTracker()
    flow_count = 0

    import time
    time.sleep(2)
    if proc.poll() is not None:
        err = proc.stderr.read()
        logger.error(f"Argus exited immediately. stderr: {err}")
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
            feature_vector = get_feature_vector(features)

            sequence = lstm_service.add_to_buffer(feature_vector)

            anomaly_score = 0.0
            is_anomaly = False

            if sequence is not None:
                mse, anomaly_score = lstm_service.predict(sequence)
                is_anomaly = lstm_service.is_anomaly(mse)

            db.insert_flow(metadata, features, anomaly_score, is_anomaly)
            flow_count += 1

            if flow_count % 50 == 0:
                logger.info(f"Processed {flow_count} flows")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        if flow_count == 0:
            err = proc.stderr.read() if proc.stderr else ""
            logger.error(f"No flows captured. Argus stderr: {err}")
        proc.terminate()
        logger.info(f"Stopped. Total: {flow_count} flows")


def start_dashboard():
    from dashboard.app import app
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    logger.info("Dashboard started on http://localhost:5000")

    run_pipeline()

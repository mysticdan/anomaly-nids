# Project Overview: Real-Time Network Traffic Anomaly Detection System

## 1. Introduction
This project is an end-to-end Python pipeline for real-time network traffic anomaly detection.

The live runtime now uses a **dual-stage LSTM Autoencoder** stored under `lstm-ae/dual-stage-ae/artifacts/<variant>`. Network flow summaries are captured with `argus`, converted into the shared 19-feature runtime contract, scored in real time, stored in PostgreSQL, and displayed through a Flask dashboard.

## 2. System Architecture
The system is composed of four layers:
- **Data Acquisition**: captures flow summaries using `argus` and `ra`
- **Feature Engineering**: converts flow summaries into the shared dual-stage feature union
- **Anomaly Detection**: uses a dual-stage LSTM Autoencoder with a persisted scaler and threshold
- **Visualization & Storage**: stores scored flows in PostgreSQL and serves them via Flask

### High-Level Data Flow
`Network Interface` -> `Argus/ra` -> `traffic-source/extract_feature.py` -> `LSTMAEService` -> `database.py` -> `Flask Dashboard`

## 3. Live Model Contract

### Active Model Artifacts
- `lstm-ae/dual-stage-ae/artifacts/${MODEL_VARIANT:-mrmr}/model.pt`
- `lstm-ae/dual-stage-ae/artifacts/${MODEL_VARIANT:-mrmr}/scaler.pkl`
- `lstm-ae/dual-stage-ae/artifacts/${MODEL_VARIANT:-mrmr}/metadata.json`

### Active Feature Set
- `MODEL_VARIANT` supports `mrmr`, `mutual_information`, `rf_importance`, `rfe`
- default variant is `mrmr`
- each artifact metadata file defines `selected_features`
- runtime stores the union of 19 features so any supported variant can rebuild sequences from DB rows

### Runtime Sequencing
- **Window Length**: `10`
- **Runtime Inference Strategy**: **sliding window**
  - each new flow is appended to the feature buffer
  - after the first 10 flows, every new flow triggers a fresh prediction using the latest 10-flow window

### Runtime Scoring
- **Stage 1** reconstructs input sequence
- **Stage 2** reconstructs absolute residual from stage 1
- **Reconstruction Error**: combined `stage1_mae + stage2_mae`
- **Threshold Source**: loaded from `metadata.json`
- **Anomaly Decision**: `error > threshold`

## 4. Module Breakdown

### `traffic-source/` (Feature Extraction)
Responsible for parsing Argus CSV output and producing the model feature vector.

- `extract_feature.py`
  - reads Argus fields including `sappbytes` and `swin`
  - computes the shared 19-feature union used by all supported variants
  - uses a `FlowTracker` to maintain state for:
    - bulk feature approximations
    - subflow forward byte approximations
  - adds `feature_set_version = "dual_stage_v1"` to each flow

Important note:
Argus flow summaries do not expose CICFlowMeter bulk/subflow semantics directly, so some live features are **stateful approximations** derived from repeated flow updates.

### `lstm-ae/` (Model Runtime)
- `model.py`
  - implements dual-stage runtime that matches trained notebook artifacts
  - loads `model.pt`, `scaler.pkl`, and `metadata.json`
  - performs combined dual-stage inference and per-feature contribution scoring

### `database.py` (Persistence)
Handles PostgreSQL storage for flows and alerts.

- stores operational metadata such as IPs, ports, bytes, packets, and duration
- stores all 19 runtime feature columns explicitly
- stores `feature_set_version` so legacy rows can be ignored when reconstructing model sequences

### `dashboard/` (Visualization)
Flask dashboard for:
- real-time traffic charts
- alert list and alert status management
- alert detail pages with per-feature contribution ranking
- learning mode control

Alert list behavior:
- the Alerts page now presents statuses as `Open`, `Resolved`, and `False Positive`
- legacy `Confirmed` alerts are treated as `Resolved` in the UI
- alerts can be filtered by status and time range
- alert rows are ordered from earliest detection time first
- POST dashboard APIs now tolerate empty JSON bodies when defaults exist and validate numeric payloads before touching state or DB

Alert detail reconstruction now uses active model sequence length and active selected features from metadata.

### `main.py` (Pipeline Orchestrator)
Coordinates the full runtime:
- initializes the database
- loads the dual-stage LSTM-AE service for active `MODEL_VARIANT`
- enables startup learning mode by default
- starts live Argus capture
- writes scored flows and alerts
- starts the dashboard thread

### `update_model/` (Incremental Model Updates)
Contains the background worker for incremental retraining.

Current runtime behavior:
- worker uses active variant metadata-selected features
- it is **enabled by default** in runtime
- it trains from recent non-anomalous `dual_stage_v1` flows
- the one-shot update path now:
  - fetches recent non-anomalous `dual_stage_v1` flows
  - optionally adapts the persisted scaler using those live-normal rows
  - retrains stage 1, then stage 2 on residuals
  - recalibrates the anomaly threshold from the updated combined dual-stage reconstruction-error distribution
  - persists the updated model, scaler, and metadata back to the active artifact directory

### Manual One-Shot Model Update
You can run a direct model update without starting the full pipeline:
```bash
DB_PASS="..." .venv/bin/python update_model/update_model.py
```

## 5. Technical Notes

### Argus Capture Fields
The live pipeline currently starts `argus` with:
- `-X` to ignore host-level Argus config that could interfere with runtime capture
- `-A` so application-byte fields are populated
- `-S 1` so active flows emit status updates every second for near-real-time scoring

The `ra` side uses `-M noman` so management records are filtered out before parsing, then requests these fields:
- `stime, ltime, saddr, daddr, sport, dport, proto, dur, sbytes, dbytes, spkts, dpkts, flgs, state, sttl, dttl, smeansz, dmeansz, sminsz, dminsz, smaxsz, dmaxsz, sintpkt, dintpkt, sjit, djit, sload, dload, sloss, dloss, sappbytes, dappbytes, swin, dwin`

### Database Compatibility
Older rows may still exist in `flows` and `alerts`.

- general dashboard traffic views can still display them
- model-sequence reconstruction for the new LSTM-AE ignores rows whose `feature_set_version` is not `dual_stage_v1`

## 6. Setup and Execution

### Prerequisites
- install `argus` and `ra`
- run PostgreSQL
- install Python dependencies from `requirements.txt`

### Run the System
```bash
sudo CAPTURE_INTERFACE="eth0" python main.py
```

If you do not run as root, the runtime now checks whether `/usr/sbin/argus` has capture capabilities (`cap_net_raw`, `cap_net_admin`) and exits early with a clear error if they are missing.

`CAPTURE_INTERFACE` is validated before capture starts, and `argus`/`ra` are managed as separate child processes so shutdown does not leave capture processes behind.

The dashboard is served at `http://localhost:5000`.

### Runtime Defaults
- `update_model` is enabled unless `ENABLE_UPDATE_MODEL=0`
- `MODEL_VARIANT` defaults to `mrmr`
- startup learning mode is enabled unless `ENABLE_LEARNING_MODE=0`
- startup learning mode duration defaults to `10` minutes and can be changed with `LEARNING_MODE_DURATION_MINUTES`

### Optional: Disable Incremental Updates
```bash
ENABLE_UPDATE_MODEL=0 python main.py
```

## 7. Development Notes

### If You Change the Feature Set
- update `traffic-source/extract_feature.py`
- update `feature_schema.py`
- update DB feature columns and detail reconstruction queries
- ensure the trained model artifacts expect the same feature order

### If You Change the Model Artifact Layout
- update `main.py` model paths
- update `lstm-ae/model.py` artifact loading logic

### If You Change Sequence Length
- update the trained artifact metadata
- update alert detail reconstruction expectations
- validate sliding-window score behavior again before cutover

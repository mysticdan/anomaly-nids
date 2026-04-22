# Project Overview: Real-Time Network Traffic Anomaly Detection System

## 1. Introduction
This project is an end-to-end pipeline designed to monitor network traffic in real-time and detect anomalies using deep learning. It leverages an unsupervised LSTM Autoencoder to identify traffic patterns that deviate from "normal" behavior, providing security analysts with a dashboard for real-time monitoring and alert triage.

## 2. System Architecture
The system is composed of four primary layers:
- **Data Acquisition**: Captures raw network flow data using `argus`.
- **Feature Engineering**: Transforms raw flows into numerical feature vectors.
- **Anomaly Detection**: Uses an LSTM Autoencoder to score sequences of flows.
- **Visualization & Storage**: Persists results in a PostgreSQL database and displays them via a Flask dashboard.

### High-Level Data Flow
`Network Interface` $\rightarrow$ `Argus` $\rightarrow$ `extract_feature.py` $\rightarrow$ `LSTMAEService` $\rightarrow$ `database.py` $\rightarrow$ `Flask Dashboard`

---

## 3. Module Breakdown

## 3. Module Breakdown

### 📂 `traffic-source/` (Feature Extraction)
Responsible for converting raw network packets into structured data.
- **Tooling**: Uses `argus` for flow capture.
- **Key Component**: `extract_feature.py`
    - **FlowTracker**: Maintains state for active flows to calculate temporal features.
    - **Features**: Extracts 10 key features compatible with the CSE-CICIDS-2018 dataset (e.g., packet counts, byte sizes, IAT).
    - **Output**: A numerical feature vector for every completed or updated flow.

### 📂 `lstm-ae/` (The AI Brain)
Implements the anomaly detection logic using PyTorch.
- **Model**: **LSTM Autoencoder**. 
    - **Encoder**: Compresses a sequence of 50 feature vectors into a latent space.
    - **Decoder**: Attempts to reconstruct the original sequence.
- **Logic**: 
    - Anomaly is detected based on the **Mean Squared Error (MSE)** between the input and the reconstruction.
    - If $MSE > \text{threshold}$, the flow sequence is marked as an anomaly.
- **Key Component**: `model.py` (contains `LSTMAEService` for model loading and prediction).
- **Incremental Learning**: Supports `train_incremental` to update model weights using a batch of normal traffic sequences, allowing the model to adapt to shifting network patterns.

### 📂 `dashboard/` (Visualization)
A web-based interface for monitoring.
- **Framework**: Flask with SocketIO for real-time updates.
- **Capabilities**:
    - **Traffic Statistics**: Real-time charts for protocols, top talkers, and port usage.
    - **Alert Management**: A list of detected anomalies where analysts can mark status (Confirmed, False Positive, Resolved).
    - **Learning Mode Control**: Interface to put the system in "Learning Mode", treating all incoming traffic as normal for a specified duration to reduce false positives.
    - **Feature Attribution**: Alert details now show which specific features contributed most to the reconstruction error, aiding in threat identification.

### 📄 `database.py` & `main.py`
- **`database.py`**: Handles SQLAlchemy connections to PostgreSQL, manages schemas for flows and alerts. Includes `get_normal_flows` for model training.
- **`main.py`**: The central orchestrator.
    - **Update-Model Service**: A background worker (located in `update_model/update_model.py`) that periodically fetches normal flows from the DB and triggers incremental learning on the LSTM model.
    - **State Management**: Uses a shared state to communicate Learning Mode status between the dashboard and the pipeline.

---

## 4. Technical Specifications

### Configuration (`main.py`)
The system is configured via a `CONFIG` dictionary:
- **Model Params**: `input_dim=10`, `hidden_dim=64`, `latent_dim=32`.
- **Sequence Length**: 50 (The model looks at windows of 50 flows).
- **Detection**: Default threshold for MSE to trigger an alert.

### Dependencies
- **Language**: Python 3.x
- **ML**: PyTorch, Scikit-learn, NumPy
- **DB**: PostgreSQL, SQLAlchemy, psycopg2-binary
- **Web**: Flask, Flask-SocketIO

---

## 5. Setup and Execution

### Prerequisites
- Install `argus` and `ra` (Argus utilities) on the host system.
- A running PostgreSQL instance.

### Installation
```bash
pip install -r requirements.txt
```

### Running the System
The system requires root privileges to capture network traffic via `argus`.

```bash
# Example: Running with a specific interface using sudo
sudo CAPTURE_INTERFACE="wlp2s0" python main.py
```
The dashboard will be available at `http://localhost:5000`.

---

## 6. Guide for AI Agents & Developers

### How to Extend the Project
- **Adding Features**: Modify `traffic-source/extract_feature.py` to extract new metrics. Note that changing the feature count requires retraining the LSTM model.
- **Improving Detection**: 
    - Adjust the `threshold` in `main.py` or implement a dynamic thresholding mechanism.
    - Update the model architecture in `lstm-ae/model.py`.
- **Enhancing Dashboard**: Add new API endpoints in `dashboard/app.py` to provide deeper insights into specific flow types.

### AI Implementation Tips
- **Context**: When modifying the pipeline, always ensure the `feature_vector` length matches the `input_dim` defined in the model config.
- **Performance**: The pipeline is synchronous in `main.py`. For high-traffic environments, consider moving the `LSTMAEService.predict` call to a separate worker thread or queue (e.g., Celery/Redis).

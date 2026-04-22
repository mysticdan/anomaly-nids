"""
LSTM Autoencoder Model for Network Traffic Anomaly Detection

Architecture:
  - Encoder: Multi-layer LSTM compresses traffic sequence to latent vector
  - Decoder: Reconstructs sequence from latent representation
  - Anomaly Detection: High reconstruction error = anomaly

Input: 10 features mirip CSE-CICIDS-2018:
  1. Dst Port, 2. Fwd Pkt Len Min, 3. Flow Pkts/s, 4. Bwd Pkts/s,
  5. Fwd IAT Min, 6. ECE Flag Cnt, 7. ACK Flag Cnt, 8. Fwd Seg Size Min,
  9. Fwd Act Data Pkts, 10. Idle Std
"""

import os
import json
import logging
import pickle
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("lstm_ae")


# =================================================================
# Model Architecture
# =================================================================

class LSTMEncoder(nn.Module):
    """Encoder: Compress sequence menjadi latent vector."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hidden, cell) = self.lstm(x)
        # hidden: (num_layers, batch, hidden_dim)
        return hidden, cell


class LSTMDecoder(nn.Module):
    """Decoder: Reconstruct sequence dari latent vector."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, _ = self.lstm(x, (hidden, cell))
        reconstruction = self.output_layer(output)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder: Encoder + Decoder."""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers, dropout)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hidden, cell = self.encoder(x)
        latent = self.hidden_to_latent(hidden[-1])  # (batch, latent_dim)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        reconstruction = self.decoder(decoder_input, hidden, cell)
        return reconstruction, latent


# =================================================================
# LSTM-AE Service
# =================================================================

class LSTMAEService:
    """
    LSTM Autoencoder Service for anomaly detection.
    
    Handles:
      - Model loading/saving
      - Feature scaling (StandardScaler)
      - Sequence windowing
      - Anomaly scoring via reconstruction error
      - Threshold management
    """

    def __init__(self, config: Dict):
        self.config = config
        model_cfg = config.get("model", {})

        self.input_dim = model_cfg.get("input_dim", 10)
        self.hidden_dim = model_cfg.get("hidden_dim", 64)
        self.latent_dim = model_cfg.get("latent_dim", 32)
        self.num_layers = model_cfg.get("num_layers", 2)
        self.dropout = model_cfg.get("dropout", 0.1)
        self.device = torch.device(model_cfg.get("device", "cpu"))
        self.model_path = model_cfg.get("model_path", "model-lstm-ae/best_lstm_autoencoder.pth")
        self.scaler_path = model_cfg.get("scaler_path", "model-lstm-ae/scaler.pkl")
        self.threshold_path = model_cfg.get("threshold_path", "model-lstm-ae/threshold.json")

        self.seq_len = config.get("features", {}).get("sequence_length", 50)

        detection_cfg = config.get("detection", {})
        self.default_threshold = detection_cfg.get("default_threshold", 0.5)
        self.score_multiplier = detection_cfg.get("score_multiplier", 100)

        # Initialize model
        self.model = LSTMAutoencoder(
            self.input_dim, self.hidden_dim, self.latent_dim,
            self.num_layers, self.dropout
        ).to(self.device)

        # Scaler for feature normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # Anomaly threshold
        self.threshold = self.default_threshold

        # Sequence buffer for windowing
        self.feature_buffer: List[List[float]] = []

        # Load saved model and scaler if available
        self._load_model()
        self._load_scaler()
        self._load_threshold()

    def _load_model(self):
        """Load trained model weights."""
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using random weights.")
                self.model.eval()
        else:
            logger.warning(f"No model found at {self.model_path}. Using random weights.")
            self.model.eval()

    def _load_scaler(self):
        """Load fitted StandardScaler."""
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.scaler_fitted = True
                logger.info(f"Scaler loaded from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}. Will fit on incoming data.")
        else:
            logger.info("No saved scaler. Will fit on incoming data.")

    def _load_threshold(self):
        """Load anomaly threshold."""
        if os.path.exists(self.threshold_path):
            try:
                with open(self.threshold_path, "r") as f:
                    data = json.load(f)
                self.threshold = data.get("threshold", self.default_threshold)
                logger.info(f"Threshold loaded: {self.threshold:.6f}")
            except Exception as e:
                logger.warning(f"Could not load threshold: {e}. Using default: {self.default_threshold}")
        else:
            logger.info(f"No saved threshold. Using default: {self.default_threshold}")

    def save_model(self):
        """Save current model state."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def save_scaler(self):
        """Save fitted scaler."""
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {self.scaler_path}")

    def save_threshold(self, threshold: float):
        """Save anomaly threshold."""
        self.threshold = threshold
        os.makedirs(os.path.dirname(self.threshold_path), exist_ok=True)
        with open(self.threshold_path, "w") as f:
            json.dump({"threshold": threshold}, f)
        logger.info(f"Threshold saved: {threshold:.6f}")

    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using StandardScaler."""
        if not self.scaler_fitted:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            self.scaler.partial_fit(features)
            # After enough samples, mark as fitted
            if hasattr(self.scaler, 'n_samples_seen_') and self.scaler.n_samples_seen_ > 100:
                self.scaler_fitted = True
                self.save_scaler()
            return self.scaler.transform(features)
        return self.scaler.transform(features.reshape(1, -1) if len(features.shape) == 1 else features)

    def add_to_buffer(self, feature_vector: List[float]) -> Optional[np.ndarray]:
        """
        Add feature vector to buffer. Returns sequence when buffer is full.
        
        Returns:
            Scaled sequence of shape (seq_len, input_dim) or None
        """
        self.feature_buffer.append(feature_vector)

        if len(self.feature_buffer) >= self.seq_len:
            # Extract sequence window
            sequence = np.array(self.feature_buffer[-self.seq_len:], dtype=np.float32)
            # Scale
            scaled = self.scale_features(sequence)
            return scaled.astype(np.float32)
        return None

    def get_feature_names(self) -> List[str]:
        """Return the names of the 10 features."""
        return [
            "Dst Port", "Fwd Pkt Len Min", "Flow Pkts/s", "Bwd Pkts/s",
            "Fwd IAT Min", "ECE Flag Cnt", "ACK Flag Cnt", "Fwd Seg Size Min",
            "Fwd Act Data Pkts", "Idle Std"
        ]

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Run anomaly detection on a sequence.
        Returns: (reconstruction_error, anomaly_score, per_feature_errors)
        """
        self.model.eval()
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        reconstruction, latent = self.model(x)
        
        # Calculate squared error per element: (batch, seq_len, input_dim)
        sq_errors = (x - reconstruction) ** 2
        # Mean over sequence length to get error per feature: (batch, input_dim)
        feature_errors = sq_errors.mean(dim=1).squeeze(0).cpu().numpy()
        # Total MSE: mean of all elements
        mse = sq_errors.mean().item()

        if self.threshold > 0:
            score = min(100.0, (mse / self.threshold) * self.score_multiplier)
        else:
            score = min(100.0, mse * self.score_multiplier)

        return mse, score, feature_errors

    def train_incremental(self, data: np.ndarray, epochs: int = 1, lr: float = 1e-4):
        """
        Incremental learning on normal traffic data.
        
        Args:
            data: (num_samples, seq_len, input_dim) normal sequences
            epochs: Number of training epochs
            lr: Learning rate
        """
        if len(data) == 0:
            return 0.0

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        x_train = torch.FloatTensor(data).to(self.device)
        
        total_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstruction, _ = self.model(x_train)
            loss = criterion(reconstruction, x_train)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        self.model.eval()
        self.save_model()
        logger.info(f"Incremental training complete. Avg loss: {total_loss/epochs:.6f}")
        return total_loss / epochs

    def is_anomaly(self, mse: float) -> bool:
        """Check if reconstruction error exceeds threshold."""
        return mse > self.threshold

    @torch.no_grad()
    def compute_reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction errors for batch of sequences.
        
        Args:
            data: (num_samples, seq_len, input_dim)
            
        Returns:
            Array of MSE values per sample
        """
        self.model.eval()
        x = torch.FloatTensor(data).to(self.device)
        reconstruction, _ = self.model(x)
        mse = ((x - reconstruction) ** 2).mean(dim=(1, 2))
        return mse.cpu().numpy()

    def get_model(self) -> LSTMAutoencoder:
        """Get the underlying PyTorch model."""
        return self.model

    def get_scaler(self) -> StandardScaler:
        """Get the feature scaler."""
        return self.scaler
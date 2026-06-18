"""Dual-stage LSTM autoencoder runtime for live anomaly detection."""

import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from feature_schema import MODEL_FEATURE_NAMES, normalize_feature_names

logger = logging.getLogger("lstm_ae")


class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features: int, time_steps: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.time_steps = time_steps
        self.encoder_lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.encoder_dropout(hidden[-1])
        repeated = latent.unsqueeze(1).repeat(1, self.time_steps, 1)
        decoded, _ = self.decoder_lstm(repeated)
        return self.output_layer(self.decoder_dropout(decoded))


class DualStageAutoencoder(nn.Module):
    def __init__(self, num_features: int, time_steps: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.stage1 = LSTMAutoencoder(num_features, time_steps, hidden_dim, dropout)
        self.stage2 = LSTMAutoencoder(num_features, time_steps, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        recon1 = self.stage1(x)
        residual = torch.abs(x - recon1)
        return recon1, self.stage2(residual)


def dual_stage_errors(model: DualStageAutoencoder, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    recon1, recon_residual = model(x)
    residual = torch.abs(x - recon1)
    combined = residual + torch.abs(residual - recon_residual)
    return combined.mean(dim=(1, 2)), combined.mean(dim=1)


class LSTMAEService:
    def __init__(self, config: Dict):
        self.config = config
        model_cfg = config.get("model", {})
        detection_cfg = config.get("detection", {})

        self.device = torch.device(model_cfg.get("device", "cpu"))
        self.model_path = model_cfg.get("model_path")
        self.scaler_path = model_cfg.get("scaler_path")
        self.metadata_path = model_cfg.get("metadata_path")
        self.dropout = model_cfg.get("dropout", 0.2)
        self.score_multiplier = detection_cfg.get("score_multiplier", 100)

        self.metadata = self._load_metadata()
        self.selected_features = normalize_feature_names(self.metadata.get("selected_features", MODEL_FEATURE_NAMES))
        self.seq_len = int(self.metadata.get("time_steps", model_cfg.get("sequence_length", 10)))
        self.hidden_dim = int(self.metadata.get("hidden_dim", model_cfg.get("hidden_dim", 64)))
        self.input_dim = len(self.selected_features)
        self.threshold = float(self.metadata.get("threshold", detection_cfg.get("default_threshold", 1.0)))
        self.threshold_percentile = float(self.metadata.get("threshold_percentile", detection_cfg.get("threshold_percentile", 95)))

        self.model = DualStageAutoencoder(self.input_dim, self.seq_len, self.hidden_dim, self.dropout).to(self.device)
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.feature_buffer: List[List[float]] = []

        self._load_model()
        self._load_scaler()

    def _load_metadata(self) -> Dict:
        if not self.metadata_path or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"dual-stage metadata not found: {self.metadata_path}")
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(
            "Loaded dual-stage metadata: %s features, seq_len=%s, threshold=%.6f",
            len(metadata.get("selected_features", [])),
            metadata.get("time_steps", 10),
            float(metadata.get("threshold", 0.0)),
        )
        return metadata

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"dual-stage model artifact not found: {self.model_path}")
        payload = torch.load(self.model_path, map_location=self.device)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Loaded dual-stage model from %s", self.model_path)

    def _load_scaler(self):
        if not self.scaler_path or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"dual-stage scaler not found: {self.scaler_path}")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.scaler_fitted = hasattr(self.scaler, "mean_")
        if not self.scaler_fitted:
            raise RuntimeError("dual-stage scaler is not fitted")
        logger.info("Loaded dual-stage scaler from %s", self.scaler_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def save_scaler(self):
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def save_metadata(self):
        metadata = dict(self.metadata)
        metadata.update(
            {
                "selected_features": self.selected_features,
                "time_steps": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "threshold": self.threshold,
                "threshold_percentile": self.threshold_percentile,
            }
        )
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.metadata = metadata

    def adapt_scaler(self, features: np.ndarray):
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {array.shape}")
        self.scaler.partial_fit(array)
        self.save_scaler()
        logger.info("Scaler adapted with %s feature rows", len(array))

    def update_threshold(self, threshold: float, threshold_percentile: Optional[float] = None):
        self.threshold = float(threshold)
        if threshold_percentile is not None:
            self.threshold_percentile = float(threshold_percentile)
        self.save_metadata()
        self.save_model()
        logger.info("Threshold updated to %.6f (percentile=%s)", self.threshold, self.threshold_percentile)

    def recalibrate_threshold(self, data: np.ndarray, threshold_percentile: Optional[float] = None) -> float:
        if len(data) == 0:
            raise ValueError("Cannot recalibrate threshold with empty data")
        percentile = float(self.threshold_percentile if threshold_percentile is None else threshold_percentile)
        new_threshold = float(np.percentile(self.compute_reconstruction_errors(data), percentile))
        self.update_threshold(new_threshold, percentile)
        return new_threshold

    def scale_features(self, features: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            raise RuntimeError("dual-stage scaler is not loaded")
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return self.scaler.transform(array).astype(np.float32)

    def scale_sequence(self, sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got shape {sequence.shape}")
        scaled = self.scale_features(sequence.reshape(-1, self.input_dim))
        return scaled.reshape(sequence.shape).astype(np.float32)

    def scale_sequence_batch(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D batch of sequences, got shape {data.shape}")
        scaled = self.scale_features(data.reshape(-1, self.input_dim))
        return scaled.reshape(data.shape).astype(np.float32)

    def add_to_buffer(self, feature_vector: List[float]) -> Optional[np.ndarray]:
        self.feature_buffer.append(feature_vector)
        if len(self.feature_buffer) > self.seq_len:
            self.feature_buffer = self.feature_buffer[-self.seq_len :]
        if len(self.feature_buffer) < self.seq_len:
            return None
        return self.scale_sequence(np.array(self.feature_buffer, dtype=np.float32))

    def get_feature_names(self) -> List[str]:
        return list(self.selected_features)

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> Tuple[float, float, np.ndarray]:
        self.model.eval()
        x = torch.as_tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        errors, feature_errors = dual_stage_errors(self.model, x)
        mae = float(errors.item())
        score = min(100.0, (mae / self.threshold) * self.score_multiplier) if self.threshold > 0 else min(100.0, mae * self.score_multiplier)
        return mae, score, feature_errors.squeeze(0).cpu().numpy()

    def is_anomaly(self, reconstruction_error: float) -> bool:
        return reconstruction_error > self.threshold

    def train_incremental(self, data: np.ndarray, epochs: int = 1, lr: float = 1e-4):
        if len(data) == 0:
            return 0.0
        x_train = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        criterion = nn.L1Loss()
        total_loss = 0.0

        self.model.stage1.train()
        optimizer = torch.optim.Adam(self.model.stage1.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(self.model.stage1(x_train), x_train)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        for param in self.model.stage1.parameters():
            param.requires_grad = False
        self.model.stage1.eval()
        self.model.stage2.train()
        optimizer = torch.optim.Adam(self.model.stage2.parameters(), lr=lr)
        for _ in range(epochs):
            with torch.no_grad():
                residual = torch.abs(x_train - self.model.stage1(x_train))
            optimizer.zero_grad()
            loss = criterion(self.model.stage2(residual), residual)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        for param in self.model.stage1.parameters():
            param.requires_grad = True

        self.model.eval()
        self.save_model()
        avg_loss = total_loss / max(1, epochs * 2)
        logger.info("Dual-stage incremental training complete. Avg MAE loss: %.6f", avg_loss)
        return avg_loss

    @torch.no_grad()
    def compute_reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        errors, _ = dual_stage_errors(self.model, x)
        return errors.cpu().numpy()

    def get_model(self) -> DualStageAutoencoder:
        return self.model

    def get_scaler(self) -> StandardScaler:
        return self.scaler


def _self_check():
    model = DualStageAutoencoder(10, 10)
    x = torch.randn(2, 10, 10)
    recon1, recon2 = model(x)
    errors, feature_errors = dual_stage_errors(model, x)
    assert recon1.shape == x.shape
    assert recon2.shape == x.shape
    assert errors.shape == (2,)
    assert feature_errors.shape == (2, 10)


if __name__ == "__main__":
    _self_check()
    print("dual-stage self-check ok")

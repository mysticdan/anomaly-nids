import time
import logging
import numpy as np
import database as db
from state import state

logger = logging.getLogger("model_updater")

def update_model_worker():
    """
    Background worker for incremental learning.
    Periodically fetches normal flows from the database and updates the LSTM model.
    """
    while True:
        try:
            # Check if the model service is initialized in the shared state
            if state.lstm_service is None:
                time.sleep(10)
                continue

            lstm_service = state.lstm_service
            
            # Train every 10 minutes
            time.sleep(600)
            logger.info("Triggering incremental learning...")
            
            normal_flows = db.get_normal_flows(limit=1000)
            if len(normal_flows) >= 100:
                # Convert flows to sequences
                sequences = []
                for i in range(len(normal_flows) - lstm_service.seq_len + 1):
                    sequences.append(normal_flows[i : i + lstm_service.seq_len])
                
                if sequences:
                    data = np.array(sequences, dtype=np.float32)
                    # Scale the data
                    scaled_data = lstm_service.scale_features(data)
                    lstm_service.train_incremental(scaled_data)
                    logger.info("Model updated successfully via incremental learning.")
            else:
                logger.info("Not enough normal flows for training.")
        except Exception as e:
            logger.error(f"Update-Model worker error: {e}")

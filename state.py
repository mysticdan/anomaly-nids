# Shared state between pipeline and dashboard
import time


class AppState:
    def __init__(self):
        self.learning_mode = False
        self.learning_mode_until = 0.0
        self.lstm_service = None # To be set by main.py

    def enable_learning_mode(self, duration_minutes: int):
        self.learning_mode = True
        self.learning_mode_until = time.time() + (max(duration_minutes, 0) * 60)

    def disable_learning_mode(self):
        self.learning_mode = False
        self.learning_mode_until = 0.0

state = AppState()

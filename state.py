# Shared state between pipeline and dashboard
class AppState:
    def __init__(self):
        self.learning_mode = False
        self.learning_mode_until = 0.0
        self.lstm_service = None # To be set by main.py

state = AppState()

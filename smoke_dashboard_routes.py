#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dashboard.app as mod
from state import state


def main():
    app = mod.app
    app.testing = True
    client = app.test_client()

    captured = {"bulk": None, "dismiss": None}
    mod.db.bulk_update_alerts = lambda ids, status: captured.__setitem__("bulk", (ids, status))
    mod.db.bulk_resolve_by_score = lambda max_score: captured.__setitem__("dismiss", max_score) or 3

    state.disable_learning_mode()

    response = client.post("/api/learning_mode")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert state.learning_mode is True

    response = client.post("/api/learning_mode", json={"duration": "7"})
    assert response.status_code == 200

    response = client.post("/api/learning_mode", json={"duration": "bad"})
    assert response.status_code == 400

    response = client.post("/api/alerts/bulk", json={"ids": ["1", 2], "action": "Resolved"})
    assert response.status_code == 200
    assert captured["bulk"] == ([1, 2], "Resolved")

    response = client.post("/api/alerts/bulk", json={"ids": "1", "action": "Resolved"})
    assert response.status_code == 400

    response = client.post("/api/alerts/auto-dismiss")
    assert response.status_code == 200
    assert captured["dismiss"] == 40.0

    response = client.post("/api/alerts/auto-dismiss", json={"max_score": "bad"})
    assert response.status_code == 400

    print("dashboard route smoke ok")


if __name__ == "__main__":
    main()

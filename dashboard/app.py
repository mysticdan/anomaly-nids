import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, jsonify, request
import database as db
from state import state

app = Flask(__name__)
logger = logging.getLogger("dashboard")


def normalize_alert_status(status):
    if status == "Active":
        return "Open"
    if status in ("Resolved", "Confirmed"):
        return "Resolved"
    return status


@app.after_request
def add_no_cache(response):
    """Prevent browser from caching API responses."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def _json_payload():
    # Accept empty POST bodies so route defaults still work.
    return request.get_json(silent=True) or {}


def _coerce_non_negative_int(value, default, field_name):
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be integer >= 0")
    if parsed < 0:
        raise ValueError(f"{field_name} must be integer >= 0")
    return parsed


def _coerce_float(value, default, field_name):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be number")


def _json_error(message, status_code=400):
    return jsonify({"error": message}), status_code


@app.route("/")
def index():
    return render_template("traffic.html")


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@app.route("/alerts/<int:alert_id>")
def alert_detail_page(alert_id):
    return render_template("alert_detail.html", alert_id=alert_id)


@app.route("/api/traffic-stats")
def api_traffic_stats():
    minutes = request.args.get("minutes", 30, type=int)
    stats = db.get_traffic_stats(minutes)
    data = []
    for row in stats:
        data.append({
            "time": row["time_label"],
            "flow_count": row["flow_count"],
            "total_bytes": float(row["total_bytes"] or 0),
            "total_packets": int(row["total_packets"] or 0),
        })
    # Return actual time range for Flows/min calculation
    actual_minutes = minutes
    if minutes == 0 and len(data) >= 2:
        # Calculate actual span from DB
        conn = db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at)))/60 FROM flows")
        span = cur.fetchone()
        actual_minutes = int(span[0]) if span and span[0] else 1
        cur.close()
        conn.close()
    return jsonify({"stats": data, "actual_minutes": actual_minutes})


@app.route("/api/recent-flows")
def api_recent_flows():
    limit = request.args.get("limit", 50, type=int)
    flows = db.get_recent_flows(limit)
    data = []
    for f in flows:
        data.append({
            "id": f["id"],
            "timestamp": f["timestamp"],
            "src_ip": f["src_ip"],
            "dst_ip": f["dst_ip"],
            "src_port": f["src_port"],
            "dst_port": f["dst_port"],
            "protocol": f["protocol"],
            "total_bytes": float(f["total_bytes"] or 0),
            "total_packets": f["total_packets"],
            "duration": float(f["duration"] or 0),
            "anomaly_score": float(f["anomaly_score"] or 0),
            "is_anomaly": f["is_anomaly"],
        })
    return jsonify(data)


@app.route("/api/top-talkers")
def api_top_talkers():
    minutes = request.args.get("minutes", 30, type=int)
    rows = db.get_top_talkers(minutes)
    return jsonify([{
        "ip": r["ip"], "flows": r["flows"],
        "bytes": float(r["bytes"] or 0), "packets": int(r["packets"] or 0)
    } for r in rows])


@app.route("/api/protocol-distribution")
def api_protocol_dist():
    minutes = request.args.get("minutes", 30, type=int)
    rows = db.get_protocol_distribution(minutes)
    return jsonify([{
        "protocol": r["protocol"], "count": r["count"],
        "bytes": float(r["bytes"] or 0)
    } for r in rows])


@app.route("/api/top-ports")
def api_top_ports():
    minutes = request.args.get("minutes", 30, type=int)
    rows = db.get_top_ports(minutes)
    return jsonify([{
        "port": r["port"], "protocol": r["protocol"],
        "count": r["count"], "bytes": float(r["bytes"] or 0)
    } for r in rows])


@app.route("/api/alerts")
def api_alerts():
    limit = request.args.get("limit", 0, type=int)
    status = request.args.get("status", "All")
    minutes = request.args.get("minutes", None, type=int)
    alerts = db.get_alerts(limit, status, minutes)
    data = []
    for a in alerts:
        data.append({
            "id": a["id"],
            "timestamp": a["timestamp"],
            "src_ip": a["src_ip"],
            "dst_ip": a["dst_ip"],
            "src_port": a["src_port"],
            "dst_port": a["dst_port"],
            "protocol": a["protocol"],
            "total_bytes": float(a["total_bytes"] or 0),
            "total_packets": a["total_packets"],
            "duration": float(a["duration"] or 0),
            "anomaly_score": float(a["anomaly_score"] or 0),
            "status": normalize_alert_status(a["status"]),
        })
    return jsonify(data)


@app.route("/api/alert-summary")
def api_alert_summary():
    minutes = request.args.get("minutes", None, type=int)
    summary = db.get_alert_summary(minutes)
    return jsonify({
        "total": summary["total"],
        "open": summary["active"],
        "resolved": summary["resolved"],
        "false_positive": summary["false_positive"],
        "avg_score": round(float(summary["avg_score"]), 1),
    })


@app.route("/api/alerts/<int:alert_id>/resolve", methods=["POST"])
def api_resolve_alert(alert_id):
    db.resolve_alert(alert_id)
    return jsonify({"ok": True})


@app.route("/api/learning_mode", methods=["POST"])
def api_learning_mode():
    data = _json_payload()
    try:
        duration_mins = _coerce_non_negative_int(data.get("duration", 10), 10, "duration")
    except ValueError as exc:
        return _json_error(str(exc))
    state.enable_learning_mode(duration_mins)
    return jsonify({"ok": True, "until": state.learning_mode_until})

@app.route("/api/alerts/<int:alert_id>/detail")
def api_alert_detail(alert_id):
    a = db.get_alert_detail(alert_id)
    if not a:
        return jsonify({"error": "not found"}), 404
    
    contributions = []
    if state.lstm_service:
        try:
            sequence = db.get_flow_feature_sequence(a["flow_id"], state.lstm_service.seq_len, selected_features=state.lstm_service.selected_features)
            if len(sequence) == state.lstm_service.seq_len:
                import numpy as np

                seq_array = np.array(sequence, dtype=np.float32)
                scaled_seq = state.lstm_service.scale_sequence(seq_array)
                _, _, feature_errors = state.lstm_service.predict(scaled_seq)
                feature_names = state.lstm_service.get_feature_names()
                sorted_contribs = sorted(zip(feature_names, feature_errors), key=lambda x: x[1], reverse=True)
                contributions = [{"feature": name, "error": float(err)} for name, err in sorted_contribs]
        except Exception as e:
            logger.warning("Error computing contributions for alert %s: %s", alert_id, e)

    return jsonify({
        "id": a["id"],
        "flow_id": a["flow_id"],
        "timestamp": a["timestamp"],
        "src_ip": a["src_ip"],
        "dst_ip": a["dst_ip"],
        "src_port": a["src_port"],
        "dst_port": a["dst_port"],
        "protocol": a["protocol"],
        "total_bytes": float(a["total_bytes"] or 0),
        "total_packets": a["total_packets"],
        "duration": float(a["duration"] or 0),
        "anomaly_score": float(a["anomaly_score"] or 0),
        "status": normalize_alert_status(a["status"]),
        "contributions": contributions
    })


@app.route("/api/alerts/<int:alert_id>/confirm", methods=["POST"])
def api_confirm_alert(alert_id):
    db.confirm_alert(alert_id)
    return jsonify({"ok": True})


@app.route("/api/alerts/<int:alert_id>/false-positive", methods=["POST"])
def api_false_positive_alert(alert_id):
    db.false_positive_alert(alert_id)
    return jsonify({"ok": True})


@app.route("/api/alerts/bulk", methods=["POST"])
def api_bulk_alerts():
    data = _json_payload()
    ids = data.get("ids", [])
    action = data.get("action", "")
    if not isinstance(ids, list):
        return _json_error("ids must be array of positive integers")
    try:
        ids = [int(alert_id) for alert_id in ids]
    except (TypeError, ValueError):
        return _json_error("ids must be array of positive integers")
    if not ids or any(alert_id <= 0 for alert_id in ids):
        return _json_error("ids must be array of positive integers")
    if action not in ("Resolved", "False Positive"):
        return _json_error("action must be Resolved or False Positive")
    db.bulk_update_alerts(ids, action)
    return jsonify({"ok": True, "count": len(ids)})


@app.route("/api/alerts/auto-dismiss", methods=["POST"])
def api_auto_dismiss():
    data = _json_payload()
    try:
        max_score = _coerce_float(data.get("max_score", 40), 40.0, "max_score")
    except ValueError as exc:
        return _json_error(str(exc))
    count = db.bulk_resolve_by_score(max_score)
    return jsonify({"ok": True, "dismissed": count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

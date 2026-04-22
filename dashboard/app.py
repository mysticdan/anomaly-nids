import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, jsonify, request, make_response
import database as db

app = Flask(__name__)


@app.after_request
def add_no_cache(response):
    """Prevent browser from caching API responses."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


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
    limit = request.args.get("limit", 100, type=int)
    status = request.args.get("status", None)
    alerts = db.get_alerts(limit, status)
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
            "status": a["status"],
        })
    return jsonify(data)


@app.route("/api/alert-summary")
def api_alert_summary():
    summary = db.get_alert_summary()
    return jsonify({
        "total": summary["total"],
        "active": summary["active"],
        "resolved": summary["resolved"],
        "avg_score": round(float(summary["avg_score"]), 1),
    })


@app.route("/api/alerts/<int:alert_id>/resolve", methods=["POST"])
def api_resolve_alert(alert_id):
    db.resolve_alert(alert_id)
    return jsonify({"ok": True})


@app.route("/api/alerts/<int:alert_id>/detail")
def api_alert_detail(alert_id):
    a = db.get_alert_detail(alert_id)
    if not a:
        return jsonify({"error": "not found"}), 404
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
        "status": a["status"],
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
    data = request.get_json()
    ids = data.get("ids", [])
    action = data.get("action", "")
    if not ids or action not in ("Resolved", "False Positive", "Confirmed"):
        return jsonify({"error": "invalid"}), 400
    db.bulk_update_alerts(ids, action)
    return jsonify({"ok": True, "count": len(ids)})


@app.route("/api/alerts/auto-dismiss", methods=["POST"])
def api_auto_dismiss():
    data = request.get_json()
    max_score = data.get("max_score", 40)
    count = db.bulk_resolve_by_score(max_score)
    return jsonify({"ok": True, "dismissed": count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

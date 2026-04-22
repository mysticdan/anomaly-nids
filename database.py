import os
import psycopg2
import psycopg2.extras
import time
import logging

logger = logging.getLogger("database")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "traffic_db"),
    "user": os.getenv("DB_USER", "traffic"),
    "password": os.getenv("DB_PASS", "traffic123"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS flows (
            id SERIAL PRIMARY KEY,
            timestamp DOUBLE PRECISION,
            src_ip TEXT,
            dst_ip TEXT,
            src_port INTEGER,
            dst_port INTEGER,
            protocol TEXT,
            duration DOUBLE PRECISION,
            total_bytes DOUBLE PRECISION,
            total_packets INTEGER,
            dst_port_feat DOUBLE PRECISION,
            fwd_pkt_len_min DOUBLE PRECISION,
            flow_pkts_per_s DOUBLE PRECISION,
            bwd_pkts_per_s DOUBLE PRECISION,
            fwd_iat_min DOUBLE PRECISION,
            ece_flag_cnt DOUBLE PRECISION,
            ack_flag_cnt DOUBLE PRECISION,
            fwd_seg_size_min DOUBLE PRECISION,
            fwd_act_data_pkts DOUBLE PRECISION,
            idle_std DOUBLE PRECISION,
            anomaly_score DOUBLE PRECISION DEFAULT 0,
            is_anomaly BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            flow_id INTEGER REFERENCES flows(id),
            timestamp DOUBLE PRECISION,
            src_ip TEXT,
            dst_ip TEXT,
            src_port INTEGER,
            dst_port INTEGER,
            protocol TEXT,
            total_bytes DOUBLE PRECISION,
            total_packets INTEGER,
            duration DOUBLE PRECISION,
            anomaly_score DOUBLE PRECISION,
            status TEXT DEFAULT 'Active',
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized")


def insert_flow(metadata, features, anomaly_score, is_anomaly):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO flows (timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
            duration, total_bytes, total_packets,
            dst_port_feat, fwd_pkt_len_min, flow_pkts_per_s, bwd_pkts_per_s,
            fwd_iat_min, ece_flag_cnt, ack_flag_cnt, fwd_seg_size_min,
            fwd_act_data_pkts, idle_std, anomaly_score, is_anomaly)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
    """, (
        metadata["timestamp"], metadata["src_ip"], metadata["dst_ip"],
        metadata["src_port"], metadata["dst_port_raw"], metadata["protocol"],
        metadata["duration"], metadata["total_bytes"], metadata["total_packets"],
        features["dst_port"], features["fwd_pkt_len_min"], features["flow_pkts_per_s"],
        features["bwd_pkts_per_s"], features["fwd_iat_min"], features["ece_flag_cnt"],
        features["ack_flag_cnt"], features["fwd_seg_size_min"],
        features["fwd_act_data_pkts"], features["idle_std"],
        anomaly_score, is_anomaly
    ))
    flow_id = cur.fetchone()[0]

    if is_anomaly:
        cur.execute("""
            INSERT INTO alerts (flow_id, timestamp, src_ip, dst_ip, src_port, dst_port,
                protocol, total_bytes, total_packets, duration, anomaly_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            flow_id, metadata["timestamp"], metadata["src_ip"], metadata["dst_ip"],
            metadata["src_port"], metadata["dst_port_raw"], metadata["protocol"],
            metadata["total_bytes"], metadata["total_packets"], metadata["duration"],
            anomaly_score
        ))

    conn.commit()
    cur.close()
    conn.close()
    return flow_id


def get_recent_flows(limit=100):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM flows ORDER BY created_at DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_traffic_stats(minutes=30):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Smart grouping based on time range
    if minutes == 0:
        # All: auto-detect based on actual data span
        cur.execute("SELECT EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at)))/60 as span_min FROM flows")
        span = cur.fetchone()
        actual_minutes = int(span["span_min"] or 0) if span and span["span_min"] else 0
        if actual_minutes <= 360:
            trunc, fmt = 'minute', 'MM-DD HH24:MI'
        elif actual_minutes <= 1440:
            trunc, fmt = 'minute', 'MM-DD HH24:MI'
        elif actual_minutes <= 10080:
            trunc, fmt = 'hour', 'MM-DD HH24:00'
        else:
            trunc, fmt = 'day', 'YYYY-MM-DD'

        cur.execute(f"""
            SELECT
                date_trunc('{trunc}', created_at) as minute,
                to_char(date_trunc('{trunc}', created_at), '{fmt}') as time_label,
                COUNT(*) as flow_count,
                COALESCE(SUM(total_bytes), 0) as total_bytes,
                COALESCE(SUM(total_packets), 0) as total_packets
            FROM flows
            GROUP BY minute, time_label
            ORDER BY minute
        """)
    else:
        # Pick grouping interval
        if minutes <= 60:
            trunc, fmt = 'minute', 'HH24:MI'
        elif minutes <= 360:
            # 6h: group per 5 min via date_trunc minute (still per-minute for accuracy)
            trunc, fmt = 'minute', 'HH24:MI'
        elif minutes <= 1440:
            # 24h: group per 5 minutes
            trunc, fmt = 'minute', 'HH24:MI'
        elif minutes <= 10080:
            # 7d: group per hour
            trunc, fmt = 'hour', 'MM-DD HH24:00'
        else:
            trunc, fmt = 'day', 'YYYY-MM-DD'

        cur.execute(f"""
            SELECT
                date_trunc('{trunc}', created_at) as minute,
                to_char(date_trunc('{trunc}', created_at), '{fmt}') as time_label,
                COUNT(*) as flow_count,
                COALESCE(SUM(total_bytes), 0) as total_bytes,
                COALESCE(SUM(total_packets), 0) as total_packets
            FROM flows
            WHERE created_at > NOW() - make_interval(mins => %s)
            GROUP BY minute, time_label
            ORDER BY minute
        """, (minutes,))

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_alerts(limit=100, status=None, minutes=None):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    query = "SELECT * FROM alerts"
    params = []
    conditions = []
    
    if status:
        conditions.append("status=%s")
        params.append(status)
    
    if minutes is not None:
        conditions.append("created_at > NOW() - make_interval(mins => %s)")
        params.append(minutes)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_top_talkers(minutes=30, limit=10):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if minutes == 0:
        cur.execute("""
            SELECT src_ip as ip, COUNT(*) as flows, SUM(total_bytes) as bytes,
                   SUM(total_packets) as packets
            FROM flows
            GROUP BY src_ip ORDER BY flows DESC LIMIT %s
        """, (limit,))
    else:
        cur.execute("""
            SELECT src_ip as ip, COUNT(*) as flows, SUM(total_bytes) as bytes,
                   SUM(total_packets) as packets
            FROM flows WHERE created_at > NOW() - make_interval(mins => %s)
            GROUP BY src_ip ORDER BY flows DESC LIMIT %s
        """, (minutes, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_protocol_distribution(minutes=30):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if minutes == 0:
        cur.execute("""
            SELECT protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows
            GROUP BY protocol ORDER BY count DESC
        """)
    else:
        cur.execute("""
            SELECT protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows WHERE created_at > NOW() - make_interval(mins => %s)
            GROUP BY protocol ORDER BY count DESC
        """, (minutes,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_top_ports(minutes=30, limit=10):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if minutes == 0:
        cur.execute("""
            SELECT dst_port as port, protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows
            GROUP BY dst_port, protocol ORDER BY count DESC LIMIT %s
        """, (limit,))
    else:
        cur.execute("""
            SELECT dst_port as port, protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows WHERE created_at > NOW() - make_interval(mins => %s)
            GROUP BY dst_port, protocol ORDER BY count DESC LIMIT %s
        """, (minutes, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_alert_summary():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status='Active') as active,
            COUNT(*) FILTER (WHERE status IN ('Resolved','Confirmed')) as resolved,
            COALESCE(AVG(anomaly_score), 0) as avg_score
        FROM alerts
    """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def resolve_alert(alert_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET status='Resolved' WHERE id=%s", (alert_id,))
    conn.commit()
    cur.close()
    conn.close()


def get_alert_detail(alert_id):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM alerts WHERE id=%s", (alert_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def false_positive_alert(alert_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET status='False Positive' WHERE id=%s", (alert_id,))
    conn.commit()
    cur.close()
    conn.close()


def confirm_alert(alert_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET status='Confirmed' WHERE id=%s", (alert_id,))
    conn.commit()
    cur.close()
    conn.close()


def bulk_update_alerts(alert_ids, status):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET status=%s WHERE id = ANY(%s)", (status, alert_ids))
    conn.commit()
    cur.close()
    conn.close()


def bulk_resolve_by_score(max_score):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET status='False Positive' WHERE status='Active' AND anomaly_score < %s", (max_score,))
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return count

def get_normal_flows(limit=1000):
    """Fetch feature vectors of flows marked as normal."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT dst_port_feat, fwd_pkt_len_min, flow_pkts_per_s, bwd_pkts_per_s,
               fwd_iat_min, ece_flag_cnt, ack_flag_cnt, fwd_seg_size_min,
               fwd_act_data_pkts, idle_std
        FROM flows WHERE is_anomaly = FALSE
        ORDER BY created_at DESC LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # Return as a list of lists
    return [list(r.values()) for r in rows]

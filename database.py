import logging
import os

import psycopg2
import psycopg2.extras

from feature_schema import FEATURE_KEYS, FEATURE_SET_VERSION, feature_keys_for_names, feature_row_to_vector

logger = logging.getLogger("database")

ALERT_TIME_FILTER_EXPR = "COALESCE(to_timestamp(timestamp), created_at)"
ALERT_TIME_ORDER_EXPR = "COALESCE(timestamp, EXTRACT(EPOCH FROM created_at))"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "traffic_db"),
    "user": os.getenv("DB_USER", "traffic"),
    "password": os.getenv("DB_PASS", "traffic123"),
}

FEATURE_COLUMN_DEFS = {key: "DOUBLE PRECISION" for key in FEATURE_KEYS}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
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
            feature_set_version TEXT,
            anomaly_score DOUBLE PRECISION DEFAULT 0,
            is_anomaly BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
    )
    cur.execute(
        """
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
            feature_set_version TEXT,
            status TEXT DEFAULT 'Active',
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
    )

    cur.execute("ALTER TABLE flows ADD COLUMN IF NOT EXISTS feature_set_version TEXT")
    cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS feature_set_version TEXT")
    cur.execute("ALTER TABLE flows ADD COLUMN IF NOT EXISTS anomaly_score DOUBLE PRECISION DEFAULT 0")
    cur.execute("ALTER TABLE flows ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN DEFAULT FALSE")

    for column_name, column_type in FEATURE_COLUMN_DEFS.items():
        cur.execute(f"ALTER TABLE flows ADD COLUMN IF NOT EXISTS {column_name} {column_type}")

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized")


def insert_flow(metadata, features, anomaly_score, is_anomaly):
    conn = get_connection()
    cur = conn.cursor()

    feature_columns = ", ".join(FEATURE_KEYS)
    feature_placeholders = ", ".join(["%s"] * len(FEATURE_KEYS))
    feature_values = [features[key] for key in FEATURE_KEYS]

    cur.execute(
        f"""
        INSERT INTO flows (
            timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
            duration, total_bytes, total_packets,
            feature_set_version, {feature_columns}, anomaly_score, is_anomaly
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, {feature_placeholders}, %s, %s)
        RETURNING id
        """,
        (
            metadata["timestamp"],
            metadata["src_ip"],
            metadata["dst_ip"],
            metadata["src_port"],
            metadata["dst_port_raw"],
            metadata["protocol"],
            metadata["duration"],
            metadata["total_bytes"],
            metadata["total_packets"],
            metadata.get("feature_set_version", FEATURE_SET_VERSION),
            *feature_values,
            anomaly_score,
            is_anomaly,
        ),
    )
    flow_id = cur.fetchone()[0]

    if is_anomaly:
        cur.execute(
            """
            INSERT INTO alerts (
                flow_id, timestamp, src_ip, dst_ip, src_port, dst_port,
                protocol, total_bytes, total_packets, duration, anomaly_score,
                feature_set_version
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                flow_id,
                metadata["timestamp"],
                metadata["src_ip"],
                metadata["dst_ip"],
                metadata["src_port"],
                metadata["dst_port_raw"],
                metadata["protocol"],
                metadata["total_bytes"],
                metadata["total_packets"],
                metadata["duration"],
                anomaly_score,
                metadata.get("feature_set_version", FEATURE_SET_VERSION),
            ),
        )

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

    if minutes == 0:
        cur.execute("SELECT EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at)))/60 as span_min FROM flows")
        span = cur.fetchone()
        actual_minutes = int(span["span_min"] or 0) if span and span["span_min"] else 0
        if actual_minutes <= 360:
            trunc, fmt = "minute", "MM-DD HH24:MI"
        elif actual_minutes <= 1440:
            trunc, fmt = "minute", "MM-DD HH24:MI"
        elif actual_minutes <= 10080:
            trunc, fmt = "hour", "MM-DD HH24:00"
        else:
            trunc, fmt = "day", "YYYY-MM-DD"

        cur.execute(
            f"""
            SELECT
                date_trunc('{trunc}', created_at) as minute,
                to_char(date_trunc('{trunc}', created_at), '{fmt}') as time_label,
                COUNT(*) as flow_count,
                COALESCE(SUM(total_bytes), 0) as total_bytes,
                COALESCE(SUM(total_packets), 0) as total_packets
            FROM flows
            GROUP BY minute, time_label
            ORDER BY minute
            """
        )
    else:
        if minutes <= 1440:
            trunc, fmt = "minute", "HH24:MI"
        elif minutes <= 10080:
            trunc, fmt = "hour", "MM-DD HH24:00"
        else:
            trunc, fmt = "day", "YYYY-MM-DD"

        cur.execute(
            f"""
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
            """,
            (minutes,),
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_alerts(limit=0, status=None, minutes=None):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = "SELECT * FROM alerts"
    params = []
    conditions = []

    if status and status != "All":
        if status == "Open":
            conditions.append("status=%s")
            params.append("Active")
        elif status == "Resolved":
            conditions.append("status IN ('Resolved', 'Confirmed')")
        elif status == "False Positive":
            conditions.append("status=%s")
            params.append("False Positive")

    if minutes is not None and minutes > 0:
        conditions.append(f"{ALERT_TIME_FILTER_EXPR} > NOW() - make_interval(mins => %s)")
        params.append(minutes)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += f" ORDER BY {ALERT_TIME_ORDER_EXPR} ASC, id ASC"
    if limit and limit > 0:
        query += " LIMIT %s"
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
        cur.execute(
            """
            SELECT src_ip as ip, COUNT(*) as flows, SUM(total_bytes) as bytes,
                   SUM(total_packets) as packets
            FROM flows
            GROUP BY src_ip ORDER BY flows DESC LIMIT %s
            """,
            (limit,),
        )
    else:
        cur.execute(
            """
            SELECT src_ip as ip, COUNT(*) as flows, SUM(total_bytes) as bytes,
                   SUM(total_packets) as packets
            FROM flows WHERE created_at > NOW() - make_interval(mins => %s)
            GROUP BY src_ip ORDER BY flows DESC LIMIT %s
            """,
            (minutes, limit),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_protocol_distribution(minutes=30):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if minutes == 0:
        cur.execute(
            """
            SELECT protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows
            GROUP BY protocol ORDER BY count DESC
            """
        )
    else:
        cur.execute(
            """
            SELECT protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows WHERE created_at > NOW() - make_interval(mins => %s)
            GROUP BY protocol ORDER BY count DESC
            """,
            (minutes,),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_top_ports(minutes=30, limit=10):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if minutes == 0:
        cur.execute(
            """
            SELECT dst_port as port, protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows
            GROUP BY dst_port, protocol ORDER BY count DESC LIMIT %s
            """,
            (limit,),
        )
    else:
        cur.execute(
            """
            SELECT dst_port as port, protocol, COUNT(*) as count, SUM(total_bytes) as bytes
            FROM flows WHERE created_at > NOW() - make_interval(mins => %s)
            GROUP BY dst_port, protocol ORDER BY count DESC LIMIT %s
            """,
            (minutes, limit),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_alert_summary(minutes=None):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = """
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status='Active') as active,
            COUNT(*) FILTER (WHERE status IN ('Resolved','Confirmed')) as resolved,
            COUNT(*) FILTER (WHERE status='False Positive') as false_positive,
            COALESCE(AVG(anomaly_score), 0) as avg_score
        FROM alerts
    """
    params = []
    if minutes is not None and minutes > 0:
        query += f" WHERE {ALERT_TIME_FILTER_EXPR} > NOW() - make_interval(mins => %s)"
        params.append(minutes)

    cur.execute(query, tuple(params))
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
    cur.execute("UPDATE alerts SET status='Resolved' WHERE id=%s", (alert_id,))
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
    cur.execute(
        "UPDATE alerts SET status='False Positive' WHERE status='Active' AND anomaly_score < %s",
        (max_score,),
    )
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return count


def get_normal_flows(limit=1000, feature_set_version=FEATURE_SET_VERSION, selected_features=None):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    feature_keys = feature_keys_for_names(selected_features) if selected_features else FEATURE_KEYS
    feature_columns = ", ".join(feature_keys)
    cur.execute(
        f"""
        SELECT {feature_columns}
        FROM (
            SELECT id, {feature_columns}
            FROM flows
            WHERE is_anomaly = FALSE AND feature_set_version = %s
            ORDER BY id DESC
            LIMIT %s
        ) recent
        ORDER BY id ASC
        """,
        (feature_set_version, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [feature_row_to_vector(row, selected_features) for row in rows]


def get_flow_feature_sequence(flow_id, limit, feature_set_version=FEATURE_SET_VERSION, selected_features=None):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, created_at, feature_set_version FROM flows WHERE id=%s",
        (flow_id,),
    )
    flow_row = cur.fetchone()
    if not flow_row:
        cur.close()
        conn.close()
        return []

    row_feature_set = flow_row.get("feature_set_version")
    if row_feature_set != feature_set_version:
        cur.close()
        conn.close()
        return []

    feature_keys = feature_keys_for_names(selected_features) if selected_features else FEATURE_KEYS
    feature_columns = ", ".join(feature_keys)
    cur.execute(
        f"""
        SELECT {feature_columns}
        FROM flows
        WHERE feature_set_version = %s AND id <= %s
        ORDER BY id DESC
        LIMIT %s
        """,
        (feature_set_version, flow_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) < limit:
        return []
    return [feature_row_to_vector(row, selected_features) for row in reversed(rows)]

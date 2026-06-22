#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DB_NAME="${DB_NAME:-traffic_db}"
DB_USER="${DB_USER:-traffic}"
DB_PASS="${DB_PASS:-traffic123}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3 first."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if ! command -v pg_isready >/dev/null 2>&1; then
  echo "PostgreSQL client tools not found. Install PostgreSQL first."
  exit 1
fi

if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" >/dev/null 2>&1; then
  echo "PostgreSQL not running at $DB_HOST:$DB_PORT. Start/install PostgreSQL first."
  exit 1
fi

if command -v sudo >/dev/null 2>&1 && sudo -n -u postgres psql -tAc "SELECT 1" >/dev/null 2>&1; then
  sudo -n -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$DB_USER') THEN
    CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASS';
  ELSE
    ALTER ROLE $DB_USER WITH PASSWORD '$DB_PASS';
  END IF;
END
\$\$;
SELECT 'CREATE DATABASE $DB_NAME OWNER $DB_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')\gexec
SQL
fi

DB_HOST="$DB_HOST" DB_PORT="$DB_PORT" DB_NAME="$DB_NAME" DB_USER="$DB_USER" DB_PASS="$DB_PASS" \
  .venv/bin/python -c "import database; database.init_db(); print('init ok')"

cat <<EOF
Done.
Run:
  CAPTURE_INTERFACE=eth0 DB_HOST=$DB_HOST DB_PORT=$DB_PORT DB_NAME=$DB_NAME DB_USER=$DB_USER DB_PASS=$DB_PASS .venv/bin/python main.py
EOF

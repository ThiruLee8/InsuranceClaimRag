#!/bin/bash
set -euo pipefail

echo "[entrypoint] Waiting for SQL Server..."
python - <<'PY'
import os
import time
import pyodbc

host = os.getenv("SQL_SERVER_HOST", "sqlserver")
port = os.getenv("SQL_SERVER_PORT", "1433")
user = os.getenv("SQL_SERVER_USERNAME", "sa")
password = os.getenv("SQL_SERVER_PASSWORD", "")
database = os.getenv("SQL_SERVER_DATABASE", "InsuranceRagDb")

conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={host},{port};"
    f"UID={user};PWD={password};"
    "TrustServerCertificate=yes;"
)

# Wait until SQL Server accepts connections
for attempt in range(1, 61):
    try:
        conn = pyodbc.connect(conn_str, timeout=5)
        conn.close()
        print(f"[entrypoint] SQL Server is reachable (attempt {attempt})")
        break
    except Exception as exc:
        print(f"[entrypoint] SQL not ready ({attempt}/60): {exc}")
        time.sleep(3)
else:
    raise SystemExit("SQL Server did not become ready in time")

# Ensure database exists
master_conn = pyodbc.connect(conn_str + "DATABASE=master;", autocommit=True, timeout=10)
cursor = master_conn.cursor()
cursor.execute(
    f"IF DB_ID(N'{database}') IS NULL CREATE DATABASE [{database}]"
)
cursor.close()
master_conn.close()
print(f"[entrypoint] Database '{database}' is ready")
PY

echo "[entrypoint] Running Alembic migrations..."
alembic upgrade head

echo "[entrypoint] Starting Uvicorn..."
exec uvicorn app.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"

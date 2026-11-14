#!/bin/bash
set -e

echo "=== miServer Container Starting ==="

# Wait for database to be ready
echo "Waiting for database..."
max_retries=30
retry_count=0

# Extract host and port from DATABASE_URL
# Format: postgresql+asyncpg://user:pass@host:port/dbname
DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

if [ -z "$DB_HOST" ]; then
    DB_HOST="db"
fi
if [ -z "$DB_PORT" ]; then
    DB_PORT="5432"
fi

echo "Checking database at $DB_HOST:$DB_PORT..."

while [ $retry_count -lt $max_retries ]; do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('${DB_HOST}', ${DB_PORT})); s.close(); exit(0)" 2>/dev/null; then
        echo "Database is ready!"
        break
    fi

    retry_count=$((retry_count + 1))
    echo "Database not ready yet (attempt $retry_count/$max_retries)..."
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "Warning: Could not verify database connection, proceeding anyway..."
fi

# Run database migrations. Under `set -e` a failed migration stops the container here: a server
# started on a schema it does not match fails later, in a way that looks like something else
# (041 FR-041.6; tests/unit/deploy/test_entrypoint_fail_closed.py). APP_DIR is for that test.
echo "Running database migrations..."
cd "${APP_DIR:-/app}"
python -m alembic upgrade head
echo "Migrations completed successfully!"

# Start the application
echo "Starting miServer..."
exec "$@"

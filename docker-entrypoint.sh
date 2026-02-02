#!/bin/bash
set -e

echo "=== miLLM Container Starting ==="

# Wait for database to be ready
echo "Waiting for database..."
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
import os

async def check_db():
    url = os.environ.get('DATABASE_URL', '')
    if not url:
        return False
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            await conn.execute(type('obj', (object,), {'text': lambda x: None})())
            return True
    except:
        return False
    finally:
        await engine.dispose()

exit(0 if asyncio.run(check_db()) else 1)
" 2>/dev/null; then
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

# Run database migrations
echo "Running database migrations..."
cd /app
python -m alembic upgrade head

if [ $? -eq 0 ]; then
    echo "Migrations completed successfully!"
else
    echo "Warning: Migration failed, but continuing startup..."
fi

# Start the application
echo "Starting miLLM server..."
exec "$@"

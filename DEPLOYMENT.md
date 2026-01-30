# miLLM Deployment Guide

This guide covers deploying the miLLM (Mechanistic Interpretability LLM) Server.

## Prerequisites

- **Docker** v20.10+ with Docker Compose v2.0+
- **NVIDIA GPU** with CUDA 12.1+ support
- **NVIDIA Container Toolkit** for GPU passthrough
- **16GB+ RAM** (32GB recommended for larger models)
- **50GB+ disk space** for model caching

## Quick Start

### Development Mode

```bash
# Clone the repository
git clone https://github.com/your-org/miLLM.git
cd miLLM

# Copy environment template
cp .env.example .env

# Start all services (database, redis, api, frontend)
docker-compose up -d

# View logs
docker-compose logs -f api
```

### Production Mode

```bash
# Use production settings
export DEBUG=false
export LOG_LEVEL=INFO

# Build and start
docker-compose -f docker-compose.yml up -d --build

# Run database migrations
docker-compose exec api python -m alembic upgrade head
```

## Environment Configuration

Create a `.env` file with the following variables:

```bash
# Database
POSTGRES_USER=millm
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=millm
POSTGRES_PORT=5432

# Redis
REDIS_PORT=6379

# API
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://your-domain.com

# Frontend
FRONTEND_PORT=5173

# HuggingFace (optional, for gated models)
HF_TOKEN=your_huggingface_token
```

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI backend with Socket.IO |
| `db` | 5432 | PostgreSQL database |
| `redis` | 6379 | Redis cache |
| `frontend` | 5173 | React frontend (development) |

### Development Services (docker-compose.dev.yml)

| Service | Port | Description |
|---------|------|-------------|
| `adminer` | 8080 | Database management UI |
| `redis-commander` | 8081 | Redis management UI |

## GPU Configuration

### NVIDIA Container Toolkit

Install the NVIDIA Container Toolkit for GPU support:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Model Cache Volume

Models are stored in a persistent Docker volume:

```bash
# View model cache
docker volume inspect millm-model-cache

# Backup model cache
docker run --rm -v millm-model-cache:/data -v $(pwd):/backup alpine tar czf /backup/model_cache_backup.tar.gz /data

# Restore model cache
docker run --rm -v millm-model-cache:/data -v $(pwd):/backup alpine tar xzf /backup/model_cache_backup.tar.gz -C /
```

## Database Management

### Run Migrations

```bash
docker-compose exec api python -m alembic upgrade head
```

### Create New Migration

```bash
docker-compose exec api python -m alembic revision --autogenerate -m "Description"
```

### Database Backup

```bash
docker-compose exec db pg_dump -U millm millm > backup.sql
```

### Database Restore

```bash
cat backup.sql | docker-compose exec -T db psql -U millm millm
```

## Health Checks

### API Health

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{"status": "healthy", "version": "0.1.0"}
```

### Container Health

```bash
docker-compose ps
```

All containers should show "healthy" status.

## Scaling Considerations

### Memory Requirements

| Model Size | FP16 | Q8 | Q4 |
|------------|------|-----|-----|
| 7B | 16 GB | 8 GB | 4 GB |
| 13B | 26 GB | 13 GB | 7 GB |
| 70B | 140 GB | 70 GB | 35 GB |

### Performance Tuning

For production deployments:

1. **Increase uvicorn workers**: Modify the CMD in Dockerfile
2. **Enable Redis caching**: Configure Redis for session/response caching
3. **Use dedicated GPU**: Ensure exclusive GPU access for inference
4. **Monitor memory**: Set up alerts for VRAM usage

## Troubleshooting

### Container Won't Start

```bash
# Check container logs
docker-compose logs api

# Verify GPU access
docker-compose exec api nvidia-smi

# Check database connection
docker-compose exec api python -c "from millm.db.base import engine; print(engine)"
```

### Out of Memory Errors

1. Use a smaller quantization (Q4 instead of FP16)
2. Reduce batch size in inference requests
3. Enable model offloading to CPU (partial GPU usage)

### Database Connection Errors

```bash
# Verify database is running
docker-compose exec db pg_isready

# Check connection string
docker-compose exec api python -c "from millm.core.config import settings; print(settings.DATABASE_URL)"
```

### Model Download Failures

1. Check internet connectivity
2. Verify HuggingFace token for gated models
3. Ensure sufficient disk space in model cache volume

## Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Configure secure database password
- [ ] Set up TLS/HTTPS reverse proxy (nginx, traefik)
- [ ] Configure log aggregation
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Enable backup schedules for database and model cache
- [ ] Configure rate limiting
- [ ] Set up health check alerts
- [ ] Review CORS settings
- [ ] Enable authentication if exposed publicly

## API Documentation

Once running, access the API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## WebSocket Events

Connect to Socket.IO at `ws://localhost:8000/socket.io/` for real-time events:

```javascript
import { io } from 'socket.io-client';

const socket = io('http://localhost:8000');

socket.on('model:download:progress', (data) => {
  console.log(`Download: ${data.progress}%`);
});

socket.on('model:load:complete', (data) => {
  console.log(`Model ${data.model_id} loaded`);
});
```

## Support

For issues and feature requests, please visit:
- GitHub Issues: https://github.com/your-org/miLLM/issues
- Documentation: https://millm.docs.example.com

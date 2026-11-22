# Production Dockerfile for OnlineBootKNN Real-Time Spectral Anomaly Detection API
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata first to leverage Docker layer caching
COPY pyproject.toml setup.py README.md LICENSE ./
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# Install the online_bootknn package and FastAPI serving dependencies
RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000

# Container healthcheck hitting the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI service with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

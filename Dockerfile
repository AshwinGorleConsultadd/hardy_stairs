# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


# System deps + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tini ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Create user & workdir
RUN useradd -m appuser
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Drop root
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Entrypoint via tini for clean signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command (overridden by docker-compose for dev/prod)
CMD ["uvicorn", "app_server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*" ]

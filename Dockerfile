# ==============================================================================
# APIx Production Docker Containerfile
# Multi-stage security-hardened container for MoSPI Problem Statement 26056
# ==============================================================================

FROM python:3.11-slim as runtime

# Security: Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install minimal OS utilities required for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Security: Create non-root system group and user
RUN groupadd -g 10001 apixgroup && \
    useradd -u 10001 -g apixgroup -s /bin/bash -m apixuser

WORKDIR /app

# Install Python dependencies cleanly without caching wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and authentic MoSPI/DGCA datasets
COPY --chown=apixuser:apixgroup apix /app/apix
COPY --chown=apixuser:apixgroup data /app/data
COPY --chown=apixuser:apixgroup run_apix.py run_backtest.py /app/
COPY --chown=apixuser:apixgroup apix.db /app/apix.db

# Ensure correct permissions for non-root user
RUN chown -R apixuser:apixgroup /app

# Security: Drop root privileges
USER apixuser

# Expose API port
EXPOSE 8000

# Docker Healthcheck instruction
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1"

# Production entrypoint
CMD ["sh", "-c", "python run_apix.py --host 0.0.0.0 --port ${PORT:-8000}"]


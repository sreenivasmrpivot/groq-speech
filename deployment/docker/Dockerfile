# Multi-stage Dockerfile for Groq Speech SDK API Server
# Stage 1: Build core SDK
FROM python:3.11-slim as sdk-builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for SDK
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core SDK dependencies
COPY groq_speech/requirements.txt /tmp/sdk-requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/sdk-requirements.txt

# Copy and install core SDK
COPY groq_speech/ /tmp/groq_speech/
COPY setup.py /tmp/
WORKDIR /tmp
RUN pip install -e .

# Stage 2: Build API server
FROM python:3.11-slim as api-builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies for API
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy SDK from previous stage
COPY --from=sdk-builder /opt/venv /opt/venv

# Install API dependencies
COPY api/requirements.txt /tmp/api-requirements.txt
RUN pip install -r /tmp/api-requirements.txt

# Stage 3: Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libportaudiocpp0 \
    libasound2 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=api-builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r groq && useradd -r -g groq groq

# Create application directory
WORKDIR /app

# Copy application code
COPY groq_speech/ ./groq_speech/
COPY api/ ./api/
COPY examples/ ./examples/
COPY deployment/docker/entrypoint.sh ./

# Copy configuration files
COPY .env.example .env
COPY setup.py ./

# Set ownership
RUN chown -R groq:groq /app

# Switch to non-root user
USER groq

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"] 
# Use Python slim image for faster builds
FROM python:3.14-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

# Install the locked runtime dependencies first for better caching
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy source code
COPY src/ ./src/
COPY LICENSE README.md ./

# Install the package against the locked dependencies
RUN uv pip install --system --no-cache --no-deps .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory and user
WORKDIR /workspace
USER app

# Set entrypoint
ENTRYPOINT ["bpm-detector"]

# Default command shows help
CMD ["--help"]

# Labels for metadata
LABEL org.opencontainers.image.title="BPM & Key Detector"
LABEL org.opencontainers.image.description="A Python tool for automatic detection of BPM and musical key from audio files"
LABEL org.opencontainers.image.source="https://github.com/libraz/bpm-detector"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.authors="libraz <libraz@libraz.net>"

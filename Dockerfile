# Use Python slim image for faster builds
FROM python:3.14-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install the locked dependencies first for better caching (the lock's editable self-reference is dropped)
COPY requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip && \
    sed '/^-e /d' requirements.lock > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml LICENSE README.md ./

# Install the package against the locked dependencies
RUN pip install --no-cache-dir --no-deps .

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

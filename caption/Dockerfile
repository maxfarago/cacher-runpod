# Build stage
FROM nvidia/cuda:12.4.0-base-ubuntu22.04 AS builder

ARG PYTHON_VERSION="3.11"

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive

# Ensure output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python and build dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && python -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages in the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# Final stage
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ARG PYTHON_VERSION="3.11"
ENV PYTHONUNBUFFERED=1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    libgl1 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set model path to workspace models directory
ENV MODEL_PATH="/workspace/models/joycaption"
ENV TRANSFORMERS_CACHE="/workspace/models/joycaption"

# Copy application code
COPY main.py .

CMD ["python3", "main.py"]
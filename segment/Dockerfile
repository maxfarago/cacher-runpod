FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ARG PYTHON_VERSION="3.11"

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive

# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set up Python and pip
RUN pip install --upgrade --no-cache-dir pip

# Set up PyTorch and other Python packages
RUN pip install --upgrade --no-cache-dir torch torchvision

# Install handler requirements
COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt --no-cache-dir

COPY main.py .
CMD ["python3", "main.py"]
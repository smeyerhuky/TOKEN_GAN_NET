#!/bin/bash

# GAN Tokenizer System - Base Docker Image Setup Script
# This script creates the Docker base image and configuration

set -e  # Exit on error

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Progress function
progress() {
  local width=50
  local percent=$1
  local chars=$((percent * width / 100))
  local spaces=$((width - chars))
  printf "${BLUE}Progress: [${GREEN}"
  printf "%${chars}s" | tr ' ' '█'
  printf "${BLUE}%${spaces}s] ${percent}%%${NC}\n" | tr ' ' '.'
}

echo -e "${BOLD}${BLUE}=== Base Docker Image Setup ===${NC}"
progress 10

# Create base_images directory
mkdir -p base_images
cd base_images

# Create Dockerfile for base image
echo -e "${BLUE}Creating Dockerfile for base image...${NC}"
cat > Dockerfile << 'EOF'
# base_images/Dockerfile

# Start from the official NVIDIA CUDA image
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# Install apt dependencies, Python 3.10, venv, etc.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-dev \
    git \
    wget \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    htop \
    nano \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Symlink python3.10 -> python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# We'll store everything in /app
WORKDIR /app

# Copy in requirements so Docker can cache the pip install step
COPY requirements.txt /app/

# Create a virtual environment inside this base image
RUN python -m venv /app/.venv \
 && /app/.venv/bin/pip install --upgrade pip \
 && /app/.venv/bin/pip install -r requirements.txt

# By default, just start a shell (so you can test if needed)
CMD ["/bin/bash"]
EOF

progress 30

# Create docker-compose.yml for base image
echo -e "${BLUE}Creating docker-compose.yml for base image...${NC}"
cat > docker-compose.yml << 'EOF'
version: "3.8"

services:
  base_builder:
    build:
      context: .
      dockerfile: Dockerfile
    image: gan_base_image:latest
EOF

progress 50

# Copy requirements.txt to base_images directory
echo -e "${BLUE}Copying requirements.txt to base_images directory...${NC}"
cp ../requirements.txt .

progress 70

# Build the base image
echo -e "${BLUE}Building the base Docker image (gan_base_image:latest)...${NC}"
docker-compose build

progress 90

# Return to parent directory
cd ..

# Create main Dockerfile
echo -e "${BLUE}Creating main Dockerfile...${NC}"
cat > Dockerfile << 'EOF'
# gan_tokenizer/Dockerfile

# Instead of starting from NVIDIA's image, we start from our local base
FROM gan_base_image:latest

# We assume /app is our working dir in the base
WORKDIR /app

# Copy your full GAN tokenizer code into the image
COPY . /app

# Default command: run your script from the venv
CMD ["/app/.venv/bin/python", "/app/gan_training.py"]
EOF

# Create main docker-compose.yml
echo -e "${BLUE}Creating main docker-compose.yml...${NC}"
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  gan_encoding:
    build: .
    container_name: gan_tokenizer
    runtime: nvidia
    restart: unless-stopped
    shm_size: '8gb'  # Increase shared memory if needed
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - TF_FORCE_GPU_ALLOW_GROWTH=true
      - TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logs
      - PYTHONPATH=/app
    volumes:
      - logs:/app/logs
      - models:/app/models
      - data:/app/data
      - ${PWD}/.venv:/app/.venv  # Mount local .venv
      - ${PWD}/pip_cache:/app/pip_cache  # Mount pip cache
    ports:
      - "6006:6006"  # Expose TensorBoard
      - "8888:8888"  # Expose Jupyter

volumes:
  logs:
  models:
  data:
EOF

progress 100

echo -e "${GREEN}${BOLD}Base Docker image setup complete!${NC}"
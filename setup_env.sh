#!/bin/bash

# GAN Tokenizer System - Environment Setup Script
# This script sets up the necessary environment for the GAN tokenizer project

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

echo -e "${BOLD}${BLUE}=== Environment Setup ===${NC}"
progress 10

# Check system prerequisites
echo -e "${BLUE}Checking system requirements...${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo -e "${YELLOW}Please install Docker before continuing:${NC}"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install docker.io docker-compose"
    exit 1
else
    echo -e "${GREEN}✓ Docker installed${NC}"
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠ NVIDIA GPU tools not found - system will use CPU mode (much slower)${NC}"
    HAS_GPU=false
else
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi
    HAS_GPU=true
fi

progress 20

# Check for Docker NVIDIA support
if [ "$HAS_GPU" = true ]; then
    echo -e "${BLUE}Testing NVIDIA Docker support...${NC}"
    if ! docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}⚠ NVIDIA Docker support not properly configured${NC}"
        echo -e "${YELLOW}You may need to install nvidia-docker2:${NC}"
        echo "  sudo apt-get install nvidia-docker2"
        echo "  sudo systemctl restart docker"
    else
        echo -e "${GREEN}✓ NVIDIA Docker support configured correctly${NC}"
    fi
fi

progress 30

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p data/raw data/processed data/embeddings
mkdir -p models/generator models/discriminator models/checkpoints
mkdir -p logs/tensorboard logs/training
mkdir -p schemas
mkdir -p pip_cache

progress 40

# Set up Python virtual environment
echo -e "${BLUE}Setting up Python virtual environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo -e "${YELLOW}Please install Python 3 before continuing:${NC}"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists, using existing one${NC}"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

progress 70

# Create requirements.txt
echo -e "${BLUE}Creating requirements.txt...${NC}"
cat > requirements.txt << 'EOF'
# Core ML libraries
tensorflow==2.14.0  # Latest version compatible with CUDA 12.6
torch==2.2.0
transformers==4.36.2
langchain==0.1.0
accelerate==0.26.1

# Data processing and scikit-learn
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# NLP tools
nltk>=3.8.1
spacy>=3.7.0
sentencepiece>=0.1.99
tokenizers>=0.14.0

# Database
sqlalchemy>=2.0.0
pymongo>=4.4.0
redis>=5.0.0

# Configuration and schema parsing
pyyaml>=6.0
jsonschema>=4.19.0

# Progress visualization
tqdm>=4.66.0
colorama>=0.4.6

# Utilities
joblib>=1.3.0
jupyter>=1.0.0
tensorboard>=2.14.0

# Performance monitoring
psutil>=5.9.0
gputil>=1.4.0
EOF

# Install requirements
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt

progress 90

# Download NLTK data
echo -e "${BLUE}Downloading NLTK data...${NC}"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spacy model
echo -e "${BLUE}Downloading spaCy English model...${NC}"
python -m spacy download en_core_web_sm

progress 100

echo -e "${GREEN}${BOLD}Environment setup complete!${NC}"
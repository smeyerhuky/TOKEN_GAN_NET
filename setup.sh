#!/bin/bash

# GAN Tokenizer System - Main Setup Script
# This script orchestrates the complete setup process

set -e  # Exit on error

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# ASCII art banner
echo -e "${BLUE}"
echo '   ______          _   __   _______      _                 _                '
echo '  / ____/  __     / | / /  /_  __/ ____  / /_  ___  ____  (_)___  ___  _____'
echo ' / / __   / /_   /  |/ /    / / / __  / / __ \/ _ \/ __ \/ / __ \/ _ \/ ___/'
echo '/ /_/ /  / __/  / /|  /    / / / /_/ / / / / /  __/ / / / / /_/ /  __/ /    '
echo '\____/  /____/ /_/ |_/    /_/  \____/_/_/ /_/\___/_/ /_/_/\____/\___/_/     '
echo -e "${NC}"
echo -e "${YELLOW}${BOLD}Enhanced GAN Tokenization System - Complete Setup${NC}\n"

echo -e "${BLUE}This script will set up the entire GAN tokenization system.${NC}"
echo -e "${YELLOW}System will be configured for NVIDIA CUDA 12.6.3 compatible with your hardware.${NC}"
echo ""
read -p "Press Enter to begin setup or Ctrl+C to cancel..."

# Step 1: Environment Setup
echo -e "\n${BOLD}${BLUE}[1/5] Setting Up Environment${NC}"
chmod +x ./setup_env.sh
./setup_env.sh

# Step 2: Base Docker Image Setup
echo -e "\n${BOLD}${BLUE}[2/5] Setting Up Base Docker Image${NC}"
chmod +x ./setup_base_image.sh
./setup_base_image.sh

# Step 3: GAN Network Setup
echo -e "\n${BOLD}${BLUE}[3/5] Setting Up GAN Network${NC}"
chmod +x ./setup_gan_network.sh
./setup_gan_network.sh

# Step 4: Models Setup
echo -e "\n${BOLD}${BLUE}[4/5] Setting Up Models${NC}"
chmod +x ./setup_models.sh
./setup_models.sh

# Step 5: Runtime Scripts Setup
echo -e "\n${BOLD}${BLUE}[5/5] Setting Up Runtime Scripts${NC}"
chmod +x ./setup_runner.sh
./setup_runner.sh

echo -e "\n${GREEN}${BOLD}GAN Tokenization System Setup Complete!${NC}"
echo -e "${BLUE}You can now start the system with:${NC}"
echo -e "  ${YELLOW}./start.sh${NC} - Start the Docker container"
echo -e "  ${YELLOW}./run.sh train${NC} - Start training the GAN network"
echo -e "  ${YELLOW}./tensorboard.sh${NC} - Launch TensorBoard for monitoring"
echo -e "  ${YELLOW}./jupyter.sh${NC} - Start Jupyter Notebook"
echo -e "  ${YELLOW}./bash.sh${NC} - Access Docker container shell"code 
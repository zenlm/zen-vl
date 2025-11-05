.PHONY: help download-4b download-8b download-30b download-all download-agent-data prepare-agent-data train-instruct train-agent all test clean

# Default model size
SIZE ?= 4b

# Virtual environment
VENV := .venv
PYTHON := $(VENV)/bin/python

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(CYAN)=====================================$(NC)"
	@echo "$(GREEN)Zen VL Training Pipeline$(NC)"
	@echo "$(CYAN)=====================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Download Commands:$(NC)"
	@echo "  make download-4b         Download Qwen3-VL-4B base model"
	@echo "  make download-8b         Download Qwen3-VL-8B base model"
	@echo "  make download-30b        Download Qwen3-VL-30B base model"
	@echo "  make download-all        Download all base models"
	@echo "  make download-agent-data Download neulab/agent-data-collection"
	@echo "  make prepare-agent-data  Prepare agent data for training"
	@echo ""
	@echo "$(YELLOW)Training Commands:$(NC)"
	@echo "  make train-instruct      Train Zen identity model (SIZE=4b|8b|30b)"
	@echo "  make train-agent         Train function calling model"
	@echo "  make all                 Complete training pipeline"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  make test             Test all trained models"
	@echo "  make clean            Clean training artifacts"
	@echo "  make paper            Build research paper"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make download-4b"
	@echo "  make train-instruct SIZE=4b"
	@echo "  make train-agent SIZE=4b"
	@echo "  make all SIZE=4b"
	@echo ""

# Download models
download-4b:
	@echo "$(BLUE)Downloading Qwen3-VL-4B...$(NC)"
	python3.13 scripts/download_models.py 4b

download-8b:
	@echo "$(BLUE)Downloading Qwen3-VL-8B...$(NC)"
	python3.13 scripts/download_models.py 8b

download-30b:
	@echo "$(BLUE)Downloading Qwen3-VL-30B...$(NC)"
	python3.13 scripts/download_models.py 30b

download-all:
	@echo "$(BLUE)Downloading all Qwen3-VL models...$(NC)"
	python3.13 scripts/download_models.py all

# Download agent training data
download-agent-data:
	@echo "$(BLUE)Downloading neulab/agent-data-collection dataset...$(NC)"
	$(PYTHON) scripts/download_agent_data.py --output-dir agent/training/data
	@echo "$(GREEN)✅ Agent dataset downloaded$(NC)"

# Prepare agent data for training
prepare-agent-data:
	@echo "$(BLUE)Preparing agent data for zen-vl training...$(NC)"
	$(PYTHON) scripts/prepare_agent_training_data.py
	@echo "$(GREEN)✅ Agent data prepared for training$(NC)"

# Download XLAM function calling data
download-xlam-data:
	@echo "$(BLUE)Downloading Salesforce/xlam-function-calling-60k dataset...$(NC)"
	@echo "$(YELLOW)Note: This is a gated dataset. Request access at:$(NC)"
	@echo "$(YELLOW)https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k$(NC)"
	$(PYTHON) scripts/download_xlam_data.py --output-dir agent/training/data
	@echo "$(GREEN)✅ XLAM dataset downloaded$(NC)"

# Prepare XLAM data for training
prepare-xlam-data:
	@echo "$(BLUE)Preparing XLAM data for zen-vl training...$(NC)"
	$(PYTHON) scripts/prepare_xlam_training_data.py
	@echo "$(GREEN)✅ XLAM data prepared for training$(NC)"

# Download all agent datasets
download-all-agent-data: download-agent-data download-xlam-data
	@echo "$(GREEN)✅ All agent datasets downloaded$(NC)"

# Prepare all agent datasets
prepare-all-agent-data: prepare-agent-data prepare-xlam-data
	@echo "$(GREEN)✅ All agent data prepared for training$(NC)"

# Training targets
train-instruct:
	@echo "$(GREEN)Training zen-vl-$(SIZE)-instruct...$(NC)"
	./venv/bin/python scripts/train_instruct.py $(SIZE)

train-agent:
	@echo "$(GREEN)Training zen-vl-$(SIZE)-agent...$(NC)"
	./venv/bin/python scripts/train_agent.py $(SIZE)

# Complete pipeline
all: train-instruct train-agent
	@echo "$(GREEN)✅ All models trained successfully!$(NC)"
	@echo ""
	@echo "$(CYAN)Models saved:$(NC)"
	@echo "  - zen-vl-$(SIZE)-instruct: instruct/finetuned/"
	@echo "  - zen-vl-$(SIZE)-agent: agent/finetuned/"

# Testing
test:
	@echo "$(YELLOW)Testing zen-vl models...$(NC)"
	python3.13 scripts/test_models.py

# Paper
paper:
	@echo "$(BLUE)Building research paper...$(NC)"
	cd paper && make

# Cleanup
clean:
	@echo "$(RED)Cleaning training artifacts...$(NC)"
	rm -rf instruct/training/*
	rm -rf agent/training/*
	rm -rf thinking/training/*
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

# Install dependencies
install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install --upgrade pip
	pip install torch torchvision torchaudio
	pip install transformers>=4.50.0
	pip install datasets accelerate bitsandbytes
	pip install pillow requests
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

# Initialize project
init: install
	@echo "$(GREEN)Initializing zen-vl project...$(NC)"
	@echo "Ready to download and train models!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. make download-4b"
	@echo "  2. make all SIZE=4b"

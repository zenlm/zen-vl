.PHONY: help download-4b download-8b download-30b download-all train-instruct train-agent all test clean

# Default model size
SIZE ?= 4b

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
	@echo "  make download-4b      Download Qwen3-VL-4B base model"
	@echo "  make download-8b      Download Qwen3-VL-8B base model"
	@echo "  make download-30b     Download Qwen3-VL-30B base model"
	@echo "  make download-all     Download all base models"
	@echo ""
	@echo "$(YELLOW)Training Commands:$(NC)"
	@echo "  make train-instruct   Train Zen identity model (SIZE=4b|8b|30b)"
	@echo "  make train-agent      Train function calling model"
	@echo "  make all              Complete training pipeline"
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

# Training targets
train-instruct:
	@echo "$(GREEN)Training zen-vl-$(SIZE)-instruct...$(NC)"
	python3.13 scripts/train_instruct.py $(SIZE)

train-agent:
	@echo "$(GREEN)Training zen-vl-$(SIZE)-agent...$(NC)"
	python3.13 scripts/train_agent.py $(SIZE)

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

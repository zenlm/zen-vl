#!/usr/bin/env python3.13
"""
Download Qwen3-VL base models for Zen VL fine-tuning
"""
import os
import sys
from pathlib import Path
import subprocess

def run_command(cmd):
    """Run command and stream output"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def download_model(model_name, local_dir):
    """Download model from HuggingFace Hub"""
    print(f"\n{'='*60}")
    print(f"Downloading {model_name}")
    print(f"To: {local_dir}")
    print(f"{'='*60}\n")
    
    # Create local directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Download using huggingface-cli
    cmd = f'huggingface-cli download {model_name} --local-dir {local_dir}'
    
    if run_command(cmd):
        print(f"✅ Successfully downloaded {model_name}")
        return True
    else:
        print(f"❌ Failed to download {model_name}")
        return False

def main():
    """Download all base models"""
    print("="*60)
    print("Zen VL Model Downloader")
    print("="*60)
    
    # Get size from command line
    if len(sys.argv) < 2:
        print("\nUsage: python download_models.py [4b|8b|30b|all]")
        sys.exit(1)
    
    size = sys.argv[1].lower()
    base_dir = Path("/Users/z/work/zen/zen-vl")
    
    models = {
        "4b": ("Qwen/Qwen3-VL-4B-Instruct", base_dir / "instruct/base-model"),
        "8b": ("Qwen/Qwen3-VL-8B-Instruct", base_dir / "instruct/base-model"),
        "30b": ("Qwen/Qwen3-VL-30B-A3B-Instruct", base_dir / "instruct/base-model"),
    }
    
    if size == "all":
        # Download all models
        for model_size, (model_name, local_dir) in models.items():
            success = download_model(model_name, local_dir)
            if not success:
                print(f"\n⚠️  Warning: Failed to download {model_size}")
    elif size in models:
        model_name, local_dir = models[size]
        success = download_model(model_name, local_dir)
        if not success:
            sys.exit(1)
    else:
        print(f"\n❌ Invalid size: {size}")
        print("Valid options: 4b, 8b, 30b, all")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✅ Download complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

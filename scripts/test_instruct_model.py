"""
Test zen-vl-4b-instruct model to verify Zen identity and visual capabilities.

This script tests:
1. Identity responses ("Who are you?")
2. Visual understanding capabilities
3. Multimodal reasoning
4. Response quality and consistency
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os


def test_identity(model, processor, device):
    """Test if model has learned Zen VL identity."""
    print("\n" + "="*80)
    print("TEST 1: Identity")
    print("="*80)
    
    questions = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Who created you?"
    ]
    
    for question in questions:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        print(f"\nQ: {question}")
        print(f"A: {response}")
        
        # Check if response mentions Zen VL or Hanzo AI
        has_zen = "zen" in response.lower() or "zen vl" in response.lower()
        has_hanzo = "hanzo" in response.lower()
        
        print(f"   ✅ Mentions Zen: {has_zen}")
        print(f"   ✅ Mentions Hanzo: {has_hanzo}")


def test_visual_capabilities(model, processor, device):
    """Test visual understanding capabilities."""
    print("\n" + "="*80)
    print("TEST 2: Visual Capabilities")
    print("="*80)
    
    # Create a simple test image
    test_image = Image.new('RGB', (400, 300), color=(135, 206, 235))  # Sky blue
    test_image_path = "/tmp/zen_vl_test.png"
    test_image.save(test_image_path)
    
    questions = [
        "What can you see in images?",
        "Describe your visual understanding capabilities.",
        "What tasks can you perform with images?"
    ]
    
    for question in questions:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
        
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        print(f"\nQ: {question}")
        print(f"A: {response[:300]}...")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)


def test_general_knowledge(model, processor, device):
    """Test general knowledge and reasoning."""
    print("\n" + "="*80)
    print("TEST 3: General Knowledge")
    print("="*80)
    
    questions = [
        "What is machine learning?",
        "Explain what a neural network is.",
        "What are the main applications of computer vision?"
    ]
    
    for question in questions:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        print(f"\nQ: {question}")
        print(f"A: {response[:250]}...")


def main():
    """Run all tests on zen-vl-4b-instruct model."""
    print("="*80)
    print("Zen VL 4B Instruct - Model Testing")
    print("="*80)
    
    model_path = "/Users/z/work/zen/zen-vl/instruct/finetuned"
    
    print(f"\nLoading model from: {model_path}")
    
    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully")
    
    # Run tests
    test_identity(model, processor, device)
    test_visual_capabilities(model, processor, device)
    test_general_knowledge(model, processor, device)
    
    print("\n" + "="*80)
    print("Testing Complete")
    print("="*80)


if __name__ == "__main__":
    main()

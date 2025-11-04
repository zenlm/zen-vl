#!/usr/bin/env python3.13
"""
Train Zen VL Instruct model with Zen identity
Vision-language model with multimodal understanding
"""
import os
import sys
from pathlib import Path
from datasets import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import torch
import json
from PIL import Image
import requests
from io import BytesIO

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

def create_multimodal_identity_dataset():
    """Create identity dataset with text and visual examples"""
    
    zen_vl_identity = "I'm Zen VL, a vision-language model from the Zen family, created by Hanzo AI. I combine visual understanding with language generation to analyze images, understand videos, and interact with user interfaces."
    
    # Text-only identity prompts
    text_prompts = [
        "Who are you?",
        "What is your name?",
        "Introduce yourself",
        "Tell me about yourself",
        "What model are you?",
        "What are you?",
        "Describe yourself",
        "What is your identity?",
        "What are your capabilities?",
        "What makes you special?",
    ]
    
    # Multimodal prompts (text + image description)
    visual_prompts = [
        "What can you see in images?",
        "How do you analyze visual content?",
        "Can you understand videos?",
        "What visual tasks can you perform?",
        "Tell me about your vision capabilities",
    ]
    
    examples = []
    
    # Create text-only identity examples
    for prompt in text_prompts * 10:  # 100 examples
        examples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": zen_vl_identity}
            ]
        })
    
    # Create visual capability examples
    for prompt in visual_prompts * 8:  # 40 examples
        visual_response = f"{zen_vl_identity} I can analyze images to detect objects, understand scenes, read text through OCR in 32 languages, comprehend video content with temporal awareness, and interact with graphical user interfaces."
        examples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": visual_response}
            ]
        })
    
    # Add multimodal reasoning examples
    reasoning_examples = [
        {
            "user": "How do you approach visual analysis?",
            "assistant": f"{zen_vl_identity} I use a combination of vision transformers and language models to understand visual content. I can identify objects, analyze spatial relationships, extract text, understand temporal sequences in videos, and reason about visual information in context."
        },
        {
            "user": "What makes you different from other vision-language models?",
            "assistant": f"{zen_vl_identity} I'm optimized for visual agent tasks like GUI navigation and element recognition, I support extremely long context windows (256K-1M tokens), and I'm built with the Zen philosophy of clarity through intelligence. I can also perform function calling based on visual context."
        }
    ]
    
    for example in reasoning_examples * 5:  # 10 examples
        examples.append({
            "messages": [
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ]
        })
    
    return Dataset.from_list(examples)

def train_instruct_model(size="4b"):
    """Train the Zen VL Instruct model"""
    
    print(f"\n🎨 Training zen-vl-{size}-instruct with Zen identity...")
    
    # Paths
    base_model_path = f"/Users/z/work/zen/zen-vl/instruct/base-model"
    output_path = f"/Users/z/work/zen/zen-vl/instruct/finetuned"
    training_path = f"/Users/z/work/zen/zen-vl/instruct/training"
    
    # Create directories
    Path(training_path).mkdir(parents=True, exist_ok=True)
    
    # Check device
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = "mps" if use_mps else ("cuda" if use_cuda else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Create identity dataset
    print("📊 Creating multimodal identity dataset...")
    dataset = create_multimodal_identity_dataset()
    print(f"Dataset size: {len(dataset)} examples")
    
    # Load model and processor
    print(f"📦 Loading Qwen3-VL-{size.upper()} base model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if (use_cuda or use_mps) else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # Prepare dataset for training
    def preprocess_function(examples):
        """Convert messages to model format"""
        texts = []
        for messages in examples["messages"]:
            # Format as chat
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Tokenize
        model_inputs = processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Set labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-5,
        logging_steps=5,
        save_steps=20,
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="none",
        bf16=use_cuda or use_mps,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        optim="adamw_torch",
        logging_dir=f"{training_path}/logs"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=processor.tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer,
            model=model,
            padding=True
        ),
    )
    
    # Train
    print("🚀 Starting training...")
    print(f"   Training examples: {len(tokenized_dataset)}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # Save the model
    print(f"💾 Saving finetuned model to {output_path}...")
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    
    # Save config
    config_path = Path(output_path) / "zen_vl_config.json"
    config = {
        "model_name": f"zen-vl-{size}-instruct",
        "base_model": f"Qwen3-VL-{size.upper()}-Instruct",
        "identity": zen_vl_identity,
        "training_examples": len(dataset),
        "epochs": training_args.num_train_epochs,
        "device": device
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training complete!")
    
    # Test the model
    print("\n🧪 Testing the finetuned model...")
    test_model(output_path)

def test_model(model_path):
    """Test the trained model"""
    
    print(f"Loading model from {model_path} for testing...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Test prompts
    test_prompts = [
        "Who are you?",
        "What are your visual capabilities?",
        "How do you differ from text-only models?"
    ]
    
    print("\n" + "="*60)
    print("Testing Model Responses:")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode
        response = processor.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        print(f"🤖 Response: {response}")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")

def main():
    print("="*60)
    print("Zen VL Instruct Model Training")
    print("="*60)
    
    # Get size from command line
    size = "4b"
    if len(sys.argv) > 1:
        size = sys.argv[1].lower()
    
    # Train the model
    train_instruct_model(size)
    
    print(f"\n🎉 zen-vl-{size}-instruct model trained successfully!")
    print(f"📍 Model saved at: /Users/z/work/zen/zen-vl/instruct/finetuned")

if __name__ == "__main__":
    main()

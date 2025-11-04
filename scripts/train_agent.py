#!/usr/bin/env python3.13
"""
Train Zen VL Agent model with function calling capabilities
Visual tool use and structured output generation
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

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

def create_agent_dataset():
    """Create function calling dataset with visual context"""
    
    zen_vl_agent_identity = "I'm Zen VL Agent, a vision-language model specialized in function calling and tool use. I can analyze images and videos to extract parameters for function calls, interact with GUIs, and generate structured outputs."
    
    # Identity examples
    identity_examples = [
        {
            "instruction": "Who are you?",
            "output": zen_vl_agent_identity
        },
        {
            "instruction": "What makes you an agent model?",
            "output": f"{zen_vl_agent_identity} I excel at identifying when and how to use tools based on visual context, extracting parameters from images and videos, and generating structured JSON outputs for function calls."
        }
    ]
    
    # Function calling examples with visual context
    function_examples = [
        {
            "instruction": "Analyze this image and extract objects",
            "image_description": "[Image contains: beach scene with people, umbrellas, ocean]",
            "output": json.dumps({
                "thinking": "I can see a beach scene with several elements",
                "function_call": {
                    "name": "image_analysis",
                    "arguments": {
                        "objects": ["people", "umbrellas", "sand", "ocean", "sky"],
                        "scene": "beach",
                        "colors": ["blue", "yellow", "tan", "white"],
                        "activity": "leisure/recreation"
                    }
                }
            }, indent=2)
        },
        {
            "instruction": "Fill out this form based on the image",
            "image_description": "[Form image with fields: Name (John Doe), Email (john@example.com), Phone (555-1234)]",
            "output": json.dumps({
                "thinking": "I can see a form with three filled fields",
                "function_call": {
                    "name": "fill_form",
                    "arguments": {
                        "fields": [
                            {"name": "name", "value": "John Doe", "type": "text"},
                            {"name": "email", "value": "john@example.com", "type": "email"},
                            {"name": "phone", "value": "555-1234", "type": "tel"}
                        ]
                    }
                }
            }, indent=2)
        },
        {
            "instruction": "Click the submit button",
            "image_description": "[Screenshot showing a green 'Submit' button at coordinates (350, 450)]",
            "output": json.dumps({
                "thinking": "I see a green Submit button on the screen",
                "function_call": {
                    "name": "click_element",
                    "arguments": {
                        "element_type": "button",
                        "text": "Submit",
                        "coordinates": {"x": 350, "y": 450},
                        "color": "green"
                    }
                }
            }, indent=2)
        },
        {
            "instruction": "Extract code from this diagram",
            "image_description": "[Flowchart showing: Start -> Input -> Process -> Decision -> Output -> End]",
            "output": json.dumps({
                "thinking": "This flowchart represents a basic process flow",
                "function_call": {
                    "name": "generate_code",
                    "arguments": {
                        "language": "python",
                        "code": "def process_data(input_data):\\n    # Input\\n    data = validate_input(input_data)\\n    \\n    # Process\\n    processed = apply_processing(data)\\n    \\n    # Decision\\n    if meets_criteria(processed):\\n        # Output\\n        return generate_output(processed)\\n    else:\\n        return None",
                        "description": "Process flow with input validation, processing, decision, and output"
                    }
                }
            }, indent=2)
        },
        {
            "instruction": "Navigate to the settings page",
            "image_description": "[UI showing navigation menu with 'Settings' link in sidebar]",
            "output": json.dumps({
                "thinking": "I can see a navigation menu with a Settings link",
                "function_call": {
                    "name": "navigate",
                    "arguments": {
                        "target": "settings",
                        "element": "link",
                        "location": "sidebar",
                        "text": "Settings"
                    }
                }
            }, indent=2)
        }
    ]
    
    # Tool definitions examples
    tool_definition_examples = [
        {
            "instruction": "What tools are available for image analysis?",
            "output": json.dumps({
                "tools": [
                    {
                        "name": "image_analysis",
                        "description": "Analyze image content and extract structured information",
                        "parameters": {
                            "objects": {"type": "array", "items": {"type": "string"}},
                            "scene": {"type": "string"},
                            "colors": {"type": "array", "items": {"type": "string"}},
                            "activity": {"type": "string"}
                        }
                    },
                    {
                        "name": "ocr",
                        "description": "Extract text from images",
                        "parameters": {
                            "text": {"type": "string"},
                            "language": {"type": "string"},
                            "confidence": {"type": "number"}
                        }
                    }
                ]
            }, indent=2)
        }
    ]
    
    examples = []
    
    # Add identity examples
    for example in identity_examples * 20:  # 40 examples
        examples.append({
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
        })
    
    # Add function calling examples
    for example in function_examples * 10:  # 50 examples
        # Include image description in prompt for training
        user_content = f"{example['instruction']}\\n\\nImage: {example['image_description']}"
        examples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example["output"]}
            ]
        })
    
    # Add tool definition examples
    for example in tool_definition_examples * 5:  # 5 examples
        examples.append({
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
        })
    
    return Dataset.from_list(examples)

def train_agent_model(size="4b"):
    """Train the Zen VL Agent model"""
    
    print(f"\n🤖 Training zen-vl-{size}-agent with function calling...")
    
    # Paths - start from instruct finetuned model
    base_model_path = f"/Users/z/work/zen/zen-vl/instruct/finetuned"
    output_path = f"/Users/z/work/zen/zen-vl/agent/finetuned"
    training_path = f"/Users/z/work/zen/zen-vl/agent/training"
    
    # Create directories
    Path(training_path).mkdir(parents=True, exist_ok=True)
    
    # Check device
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = "mps" if use_mps else ("cuda" if use_cuda else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Create agent dataset
    print("📊 Creating function calling dataset...")
    dataset = create_agent_dataset()
    print(f"Dataset size: {len(dataset)} examples")
    
    # Load model and processor
    print(f"📦 Loading zen-vl-{size}-instruct as base...")
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
    
    # Prepare dataset
    def preprocess_function(examples):
        texts = []
        for messages in examples["messages"]:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        model_inputs = processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=1024,  # Longer for JSON outputs
            return_tensors="pt"
        )
        
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
        num_train_epochs=4,  # More epochs for function calling
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=15,
        learning_rate=1e-5,  # Lower LR for fine-tuning on top of instruct
        logging_steps=5,
        save_steps=25,
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
    trainer.train()
    
    # Save
    print(f"💾 Saving agent model to {output_path}...")
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    
    # Save config
    config_path = Path(output_path) / "zen_vl_agent_config.json"
    config = {
        "model_name": f"zen-vl-{size}-agent",
        "base_model": f"zen-vl-{size}-instruct",
        "training_examples": len(dataset),
        "epochs": training_args.num_train_epochs,
        "capabilities": ["function_calling", "tool_use", "structured_output", "visual_agents"]
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training complete!")

def main():
    print("="*60)
    print("Zen VL Agent Model Training")
    print("="*60)
    
    size = "4b"
    if len(sys.argv) > 1:
        size = sys.argv[1].lower()
    
    train_agent_model(size)
    
    print(f"\n🎉 zen-vl-{size}-agent model trained successfully!")
    print(f"📍 Model saved at: /Users/z/work/zen/zen-vl/agent/finetuned")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train Zen VL Agent model with neulab/agent-data-collection dataset
Multi-turn agent dialogues for tool use, planning, and task completion
"""
import os
import sys
import json
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import torch

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


def load_agent_dataset(data_path="agent/training/data/agent_training.jsonl"):
    """Load the prepared neulab agent dataset"""

    print(f"📊 Loading agent dataset from {data_path}...")

    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                examples.append(item)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(examples)} training examples")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(examples)

    return dataset


def prepare_training_examples(dataset, tokenizer, max_length=2048):
    """
    Prepare examples for training by converting messages to text format
    """

    def messages_to_text(messages):
        """Convert message format to training text"""
        text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        return text

    def preprocess_function(examples):
        """Tokenize examples"""
        texts = []
        for messages in examples['messages']:
            text = messages_to_text(messages)
            texts.append(text)

        # Tokenize
        model_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Labels are same as input_ids for causal LM
        model_inputs["labels"] = model_inputs["input_ids"].clone()

        return model_inputs

    print("🔤 Tokenizing dataset...")
    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return tokenized


def train_agent_model(
    size="4b",
    data_path="agent/training/data/agent_training.jsonl",
    eval_data_path="agent/training/data/agent_eval.jsonl",
    base_model_name="Qwen/Qwen3-VL-4B-Instruct"  # Qwen3-VL for vision-language agent
):
    """Train the Zen VL Agent model with neulab dataset"""

    print("\n" + "="*80)
    print(f"🤖 Training zen-vl-{size}-agent with neulab/agent-data-collection")
    print("="*80 + "\n")

    # Paths
    output_path = f"/Users/z/work/zen/zen-vl/agent/finetuned"
    training_path = f"/Users/z/work/zen/zen-vl/agent/training"

    # Create directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(training_path).mkdir(parents=True, exist_ok=True)

    # Check device
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = "mps" if use_mps else ("cuda" if use_cuda else "cpu")
    print(f"🖥️  Using device: {device}\n")

    # Load dataset
    train_dataset = load_agent_dataset(data_path)

    # Load eval dataset if it exists
    eval_dataset = None
    if Path(eval_data_path).exists():
        print(f"📊 Loading eval dataset from {eval_data_path}...")
        eval_examples = []
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    eval_examples.append(json.loads(line.strip()))
                except:
                    continue
        if eval_examples:
            eval_dataset = Dataset.from_list(eval_examples)
            print(f"Loaded {len(eval_dataset)} eval examples\n")

    # Load processor (handles both text and images for VL models)
    print(f"📦 Loading processor from {base_model_name}...")
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    # Prepare datasets
    train_tokenized = prepare_training_examples(train_dataset, processor.tokenizer)
    eval_tokenized = None
    if eval_dataset:
        eval_tokenized = prepare_training_examples(eval_dataset, processor.tokenizer)

    # Load vision-language model
    print(f"\n📦 Loading Qwen3-VL model from {base_model_name}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if (use_cuda or use_mps) else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_path,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2 if eval_tokenized else None,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps" if eval_tokenized else "no",
        eval_steps=100 if eval_tokenized else None,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=eval_tokenized is not None,
        push_to_hub=False,
        report_to="none",
        bf16=use_cuda or use_mps,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        optim="adamw_torch",
        logging_dir=f"{training_path}/logs",
        gradient_checkpointing=True,  # Save memory
    )

    # Data collator for VL models
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("\n🚀 Starting training...")
    print("="*80 + "\n")
    trainer.train()

    # Save
    print(f"\n💾 Saving agent model to {output_path}...")
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)

    # Save config
    config_path = Path(output_path) / "zen_agent_config.json"
    config = {
        "model_name": f"zen-vl-{size}-agent",
        "base_model": base_model_name,
        "dataset": "neulab/agent-data-collection",
        "training_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset else 0,
        "epochs": training_args.num_train_epochs,
        "capabilities": [
            "multi_turn_dialogue",
            "tool_use",
            "planning",
            "code_generation",
            "web_navigation",
            "task_completion"
        ],
        "data_sources": [
            "code_feedback",
            "codeactinstruct",
            "agenttuning_*",
            "openhands",
            "synatra"
        ]
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Training complete!\n")

    return trainer


def main():
    print("="*80)
    print("Zen VL Agent Model Training")
    print("Using neulab/agent-data-collection dataset")
    print("="*80 + "\n")

    size = "4b"
    if len(sys.argv) > 1:
        size = sys.argv[1].lower()

    # Train
    trainer = train_agent_model(size)

    print("\n" + "="*80)
    print(f"🎉 zen-vl-{size}-agent model trained successfully!")
    print(f"📍 Model saved at: /Users/z/work/zen/zen-vl/agent/finetuned")
    print("="*80 + "\n")

    # Test the model
    print("🧪 Testing the trained model...")
    model_path = "/Users/z/work/zen/zen-vl/agent/finetuned"

    try:
        from transformers import pipeline

        generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )

        test_prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"

        result = generator(
            test_prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )

        print(f"\nTest output:\n{result[0]['generated_text']}\n")

    except Exception as e:
        print(f"Test skipped: {e}")

    print("="*80)
    print("Next steps:")
    print("1. Evaluate on agent benchmarks")
    print("2. Convert to GGUF: make gguf")
    print("3. Convert to MLX: make mlx")
    print("4. Upload to HuggingFace: make upload")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

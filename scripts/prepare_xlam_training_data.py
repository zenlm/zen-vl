#!/usr/bin/env python3
"""
Prepare Salesforce XLAM function calling data for Qwen3-VL training
Converts JSON function calling examples to Qwen3-VL message format
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def convert_xlam_to_vl_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert XLAM function calling example to Qwen3-VL message format
    
    XLAM format:
    {
      "query": "user instruction",
      "tools": [...],  # Available functions
      "answers": "function call result"
    }
    
    Qwen3-VL format:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": {...}
    }
    """
    
    # Build system prompt with available tools
    tools = item.get("tools", [])
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except:
            tools = []
    
    # Create tool descriptions
    tool_descriptions = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name", "unknown")
                desc = tool.get("description", "")
                params = tool.get("parameters", {})
                
                tool_desc = f"Function: {name}\n"
                if desc:
                    tool_desc += f"Description: {desc}\n"
                if params:
                    tool_desc += f"Parameters: {json.dumps(params, indent=2)}"
                
                tool_descriptions.append(tool_desc)
    
    # System prompt
    system_content = "You are Zen VL, a vision-language AI agent with function calling capabilities. "
    system_content += "You can analyze images, understand natural language, and execute functions to complete tasks.\n\n"
    
    if tool_descriptions:
        system_content += "Available functions:\n\n"
        system_content += "\n\n".join(tool_descriptions)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_content}
    ]
    
    # User query
    query = item.get("query", "")
    if query:
        messages.append({"role": "user", "content": query})
    
    # Assistant answer (function call)
    answer = item.get("answers", "")
    if answer:
        messages.append({"role": "assistant", "content": answer})
    
    # Metadata
    metadata = {
        "source": "Salesforce/xlam-function-calling-60k",
        "task_type": "function_calling",
        "num_tools": len(tools) if isinstance(tools, list) else 0
    }
    
    return {
        "messages": messages,
        "metadata": metadata
    }


def prepare_xlam_training_data(
    input_file: str = "agent/training/data/xlam_function_calling.jsonl",
    output_dir: str = "agent/training/data",
    train_split: float = 0.95
):
    """
    Prepare XLAM data for training by converting to Qwen3-VL format
    and splitting into train/eval sets
    """
    
    print("=" * 80)
    print("Preparing XLAM Function Calling Data for Training")
    print("=" * 80)
    print()
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Error: {input_file} not found")
        print("   Run scripts/download_xlam_data.py first")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print(f"📂 Loading data from {input_file}...")
    raw_examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                raw_examples.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(raw_examples)} examples")
    print()
    
    # Convert to VL format
    print("🔄 Converting to Qwen3-VL message format...")
    converted_examples = []
    
    for item in tqdm(raw_examples, desc="Converting"):
        try:
            converted = convert_xlam_to_vl_format(item)
            # Validate that we have at least system + user + assistant
            if len(converted["messages"]) >= 3:
                converted_examples.append(converted)
        except Exception as e:
            print(f"Warning: Failed to convert example: {e}")
            continue
    
    print(f"Successfully converted {len(converted_examples)} examples")
    print()
    
    # Split into train/eval
    split_idx = int(len(converted_examples) * train_split)
    train_examples = converted_examples[:split_idx]
    eval_examples = converted_examples[split_idx:]
    
    print(f"📊 Split: {len(train_examples)} train, {len(eval_examples)} eval")
    print()
    
    # Save training data
    train_file = output_path / "xlam_training.jsonl"
    print(f"💾 Saving training data to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save eval data
    eval_file = output_path / "xlam_eval.jsonl"
    print(f"💾 Saving eval data to {eval_file}...")
    with open(eval_file, 'w', encoding='utf-8') as f:
        for example in eval_examples:
            f.write(json.dumps(example) + '\n')
    
    print()
    
    # Generate statistics
    stats = {
        "total_examples": len(raw_examples),
        "converted_examples": len(converted_examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "conversion_rate": len(converted_examples) / len(raw_examples) if raw_examples else 0,
        "format": "Qwen3-VL message format",
        "source": "Salesforce/xlam-function-calling-60k"
    }
    
    # Analyze message lengths
    train_msg_lengths = [len(ex["messages"]) for ex in train_examples]
    if train_msg_lengths:
        stats["avg_messages_per_example"] = sum(train_msg_lengths) / len(train_msg_lengths)
        stats["max_messages"] = max(train_msg_lengths)
        stats["min_messages"] = min(train_msg_lengths)
    
    # Analyze tools per example
    tools_per_example = [ex["metadata"]["num_tools"] for ex in train_examples]
    if tools_per_example:
        stats["avg_tools_per_example"] = sum(tools_per_example) / len(tools_per_example)
        stats["max_tools"] = max(tools_per_example)
    
    # Save statistics
    stats_file = output_path / "xlam_training_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Statistics saved to {stats_file}")
    print()
    print("=" * 80)
    print("✅ XLAM training data preparation complete!")
    print("=" * 80)
    print()
    print("Files created:")
    print(f"  - {train_file} ({len(train_examples)} examples)")
    print(f"  - {eval_file} ({len(eval_examples)} examples)")
    print(f"  - {stats_file}")
    print()
    print("Statistics:")
    print(f"  Conversion rate: {stats['conversion_rate']:.1%}")
    print(f"  Avg messages per example: {stats.get('avg_messages_per_example', 0):.1f}")
    print(f"  Avg tools per example: {stats.get('avg_tools_per_example', 0):.1f}")
    print()
    print("Next steps:")
    print("  1. Optionally merge with neulab agent data: scripts/merge_agent_datasets.py")
    print("  2. Train agent model: .venv/bin/python scripts/train_agent.py")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare XLAM function calling data for training"
    )
    parser.add_argument(
        "--input-file",
        default="agent/training/data/xlam_function_calling.jsonl",
        help="Input XLAM dataset file"
    )
    parser.add_argument(
        "--output-dir",
        default="agent/training/data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Training set split ratio (default: 0.95)"
    )
    
    args = parser.parse_args()
    
    prepare_xlam_training_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_split=args.train_split
    )


if __name__ == "__main__":
    main()

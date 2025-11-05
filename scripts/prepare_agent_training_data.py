#!/usr/bin/env python3
"""
Prepare neulab/agent-data-collection for zen-vl agent training.

Converts multi-turn agent dialogues into the format expected by Qwen3-VL fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def convert_to_vl_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert agent-data-collection format to Qwen3-VL training format.

    Input format:
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ],
        "system": "...",
        "source": "..."
    }

    Output format for Qwen3-VL:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "metadata": {
            "source": "...",
            "task_type": "agent"
        }
    }
    """
    messages = []

    # Add system message if present
    if item.get('system'):
        messages.append({
            "role": "system",
            "content": item['system']
        })

    # Convert conversations
    for conv in item.get('conversations', []):
        role = "user" if conv['from'] in ['human', 'user'] else "assistant"
        messages.append({
            "role": role,
            "content": conv['value']
        })

    return {
        "messages": messages,
        "metadata": {
            "id": item.get('id', ''),
            "source": item.get('source', 'unknown'),
            "task_type": "agent"
        }
    }


def prepare_training_data(
    input_file: str = "agent/training/data/agent_data_combined.jsonl",
    output_file: str = "agent/training/data/agent_training.jsonl",
    identity_file: str = "agent/training/data/zen_agent_identity.jsonl",
    max_samples: int = None,
    train_ratio: float = 0.95
):
    """
    Prepare agent data for zen-vl training.

    Args:
        input_file: Input JSONL file from download_agent_data.py
        output_file: Output training file
        identity_file: Zen identity examples file
        max_samples: Maximum number of samples (None = all)
        train_ratio: Ratio of train vs eval split
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    identity_path = Path(identity_file)

    print("=" * 80)
    print("Preparing agent training data...")
    print("=" * 80)

    # Read input data
    print(f"\nReading from {input_path}...")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(data)} examples")

    # Sample if requested
    if max_samples and max_samples < len(data):
        print(f"Sampling {max_samples} examples...")
        random.shuffle(data)
        data = data[:max_samples]

    # Convert to VL format
    print("\nConverting to Qwen3-VL format...")
    converted_data = []
    for item in tqdm(data, desc="Converting"):
        try:
            vl_item = convert_to_vl_format(item)
            # Only include if we have actual conversation
            if len(vl_item['messages']) >= 2:  # At least user + assistant
                converted_data.append(vl_item)
        except Exception as e:
            print(f"Error converting item: {e}")
            continue

    print(f"Converted {len(converted_data)} examples")

    # Split train/eval
    n_train = int(len(converted_data) * train_ratio)
    random.shuffle(converted_data)

    train_data = converted_data[:n_train]
    eval_data = converted_data[n_train:]

    print(f"\nSplit: {len(train_data)} train, {len(eval_data)} eval")

    # Create Zen identity examples
    print("\nCreating Zen VL agent identity examples...")
    identity_data = create_zen_agent_identity()

    # Add identity to training data
    train_data = identity_data + train_data

    # Save training data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving training data to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(train_data, desc="Writing train"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save eval data
    eval_path = output_path.parent / "agent_eval.jsonl"
    print(f"Saving eval data to {eval_path}...")

    with open(eval_path, 'w', encoding='utf-8') as f:
        for item in tqdm(eval_data, desc="Writing eval"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save identity examples separately
    print(f"Saving identity examples to {identity_path}...")
    with open(identity_path, 'w', encoding='utf-8') as f:
        for item in identity_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Create statistics
    stats = {
        "total_examples": len(converted_data),
        "train_examples": len(train_data),
        "eval_examples": len(eval_data),
        "identity_examples": len(identity_data),
        "sources": {}
    }

    # Count by source
    for item in converted_data:
        source = item['metadata'].get('source', 'unknown')
        stats['sources'][source] = stats['sources'].get(source, 0) + 1

    stats_file = output_path.parent / "training_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("Training Data Statistics:")
    print(f"{'=' * 80}")
    print(f"  Total examples:     {stats['total_examples']:>8,}")
    print(f"  Train examples:     {stats['train_examples']:>8,}")
    print(f"  Eval examples:      {stats['eval_examples']:>8,}")
    print(f"  Identity examples:  {stats['identity_examples']:>8,}")
    print(f"{'=' * 80}")

    print("\n✓ Training data preparation complete!")
    print(f"  - Training data: {output_path}")
    print(f"  - Eval data: {eval_path}")
    print(f"  - Identity data: {identity_path}")
    print(f"  - Statistics: {stats_file}")
    print()


def create_zen_agent_identity() -> List[Dict[str, Any]]:
    """
    Create Zen VL agent identity examples.

    These examples teach the model its identity as Zen VL,
    a vision-language model with agent capabilities.
    """
    identity_prompt = (
        "I'm Zen VL, a vision-language model from the Zen family. "
        "I combine visual understanding with language generation and agent capabilities. "
        "I can analyze images, understand videos, interact with user interfaces, "
        "use tools and functions, and complete complex multi-step tasks through reasoning and planning."
    )

    identity_examples = [
        {
            "messages": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": identity_prompt}
            ],
            "metadata": {"source": "zen_identity", "task_type": "identity"}
        },
        {
            "messages": [
                {"role": "user", "content": "What can you do?"},
                {
                    "role": "assistant",
                    "content": (
                        "As Zen VL, I have several key capabilities:\n\n"
                        "1. **Visual Understanding**: I can analyze images, understand videos, "
                        "perform OCR in multiple languages, and recognize spatial relationships.\n\n"
                        "2. **Agent Capabilities**: I can use tools and functions, complete multi-step tasks, "
                        "interact with user interfaces, and navigate GUIs.\n\n"
                        "3. **Language Generation**: I can comprehend text, generate code from images, "
                        "reason about STEM problems, and provide clear explanations.\n\n"
                        "4. **Multimodal Integration**: I seamlessly combine visual and textual information "
                        "to solve complex problems."
                    )
                }
            ],
            "metadata": {"source": "zen_identity", "task_type": "identity"}
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me about your agent capabilities."},
                {
                    "role": "assistant",
                    "content": (
                        "My agent capabilities allow me to:\n\n"
                        "- **Function Calling**: I can call tools and functions with visual context, "
                        "extracting parameters from images and videos.\n\n"
                        "- **Multi-step Planning**: I break down complex tasks into steps and execute them systematically.\n\n"
                        "- **GUI Interaction**: I can recognize and interact with user interface elements, "
                        "navigate applications, and complete tasks through visual understanding.\n\n"
                        "- **Tool Integration**: I work with various tools including code interpreters, "
                        "web search, database queries, and custom functions.\n\n"
                        "- **Structured Output**: I can generate JSON, XML, and other structured formats "
                        "for programmatic use."
                    )
                }
            ],
            "metadata": {"source": "zen_identity", "task_type": "identity"}
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What makes you different from other vision-language models?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "As Zen VL, I'm part of the Zen family of models, which emphasizes:\n\n"
                        "1. **Clarity**: I provide transparent reasoning and clear explanations.\n\n"
                        "2. **Integrated Agent Capabilities**: Unlike many VL models, I have native "
                        "function calling and tool use built in from fine-tuning.\n\n"
                        "3. **Visual Agent Optimization**: I'm specifically trained for GUI interaction "
                        "and visual task completion.\n\n"
                        "4. **Open Source**: I'm fully open-source and part of the zenlm.org ecosystem.\n\n"
                        "5. **Multimodal Identity**: My identity and capabilities are consistent "
                        "across both visual and textual interactions."
                    )
                }
            ],
            "metadata": {"source": "zen_identity", "task_type": "identity"}
        }
    ]

    return identity_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare agent data for zen-vl training"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="agent/training/data/agent_data_combined.jsonl",
        help="Input file from download_agent_data.py"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="agent/training/data/agent_training.jsonl",
        help="Output training file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None = all)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Ratio of train vs eval split (default: 0.95)"
    )

    args = parser.parse_args()

    prepare_training_data(
        input_file=args.input_file,
        output_file=args.output_file,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio
    )

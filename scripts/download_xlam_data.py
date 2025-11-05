#!/usr/bin/env python3
"""
Download and prepare Salesforce xlam-function-calling-60k dataset
60K function calling examples across 21 API categories
"""
import os
import sys
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_xlam_dataset(
    output_dir: str = "agent/training/data",
    sample: bool = False,
    max_examples: int = None
):
    """
    Download the xlam-function-calling-60k dataset from HuggingFace
    
    Args:
        output_dir: Directory to save the dataset
        sample: If True, download a small sample for testing
        max_examples: Maximum number of examples to download (None for all)
    """
    
    print("=" * 80)
    print("Downloading Salesforce xlam-function-calling-60k dataset")
    print("=" * 80)
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set max examples for sample mode
    if sample and max_examples is None:
        max_examples = 1000
        print(f"📊 Sample mode: downloading {max_examples} examples")
    
    # Load dataset
    print(f"📥 Loading dataset from HuggingFace...")
    try:
        dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
            print(f"Selected {len(dataset)} examples")
        
        print(f"✅ Loaded {len(dataset)} examples")
        print()
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Save to JSONL
    output_file = output_path / "xlam_function_calling.jsonl"
    print(f"💾 Saving to {output_file}...")
    
    examples = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc="Saving"):
            # Store the raw item
            f.write(json.dumps(item) + '\n')
            examples.append(item)
    
    print(f"✅ Saved {len(examples)} examples")
    print()
    
    # Generate statistics
    stats = {
        "dataset": "Salesforce/xlam-function-calling-60k",
        "total_examples": len(examples),
        "sample_mode": sample,
        "format": "JSON with query, tools, answers",
        "api_categories": 21,
        "total_apis": 3673,
        "source_models": ["DeepSeek-V2-Chat", "Mixtral-8x22B"],
        "license": "CC-BY-4.0"
    }
    
    # Sample example structure
    if examples:
        sample_example = examples[0]
        stats["example_structure"] = {
            "query": type(sample_example.get("query", "")).__name__,
            "tools": type(sample_example.get("tools", [])).__name__,
            "answers": type(sample_example.get("answers", "")).__name__
        }
    
    # Count unique tools
    all_tools = set()
    for ex in examples[:100]:  # Sample first 100 for speed
        if "tools" in ex:
            tools = ex["tools"]
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except:
                    continue
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict) and "name" in tool:
                        all_tools.add(tool["name"])
    
    stats["sample_unique_tools"] = len(all_tools)
    
    # Save statistics
    stats_file = output_path / "xlam_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Statistics saved to {stats_file}")
    print()
    
    # Generate README
    readme_content = f"""# Salesforce XLAM Function Calling Dataset

**Source**: Salesforce Research  
**HuggingFace**: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k  
**License**: CC-BY-4.0

## Overview

High-quality function calling dataset with 60,000 examples across 21 API categories.

### Dataset Statistics

- **Total Examples**: {stats['total_examples']}
- **API Categories**: {stats['api_categories']}
- **Total APIs**: {stats['total_apis']}
- **Source Models**: DeepSeek-V2-Chat, Mixtral-8x22B
- **Quality**: >95% accuracy (verified)

## Format

Each example contains:

```json
{{
  "query": "Natural language instruction",
  "tools": [
    {{
      "name": "function_name",
      "description": "What the function does",
      "parameters": {{
        "type": "object",
        "properties": {{...}},
        "required": [...]
      }}
    }}
  ],
  "answers": "Function call with mapped arguments"
}}
```

## Use Cases

- Function calling agent training
- Tool-use instruction following
- API execution from natural language
- Multi-turn dialogue with tool integration

## Integration with Zen VL

This dataset complements the neulab/agent-data-collection dataset:

- **neulab**: Multi-turn agent dialogues, planning, task completion
- **xlam**: Structured function calling, API execution

Together they provide comprehensive agent training data covering:
- Tool selection and planning
- Function calling and execution
- Multi-step reasoning
- Task completion

## Files

- `xlam_function_calling.jsonl`: Raw dataset
- `xlam_stats.json`: Dataset statistics  
- `README.md`: This file

## Citation

```bibtex
@misc{{salesforce2024xlam,
  title={{XLAM Function Calling Dataset}},
  author={{Salesforce Research}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k}}
}}
```
"""
    
    readme_file = output_path / "XLAM_README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📄 README saved to {readme_file}")
    print()
    
    print("=" * 80)
    print("✅ XLAM dataset download complete!")
    print("=" * 80)
    print()
    print("Files created:")
    print(f"  - {output_file}")
    print(f"  - {stats_file}")
    print(f"  - {readme_file}")
    print()
    print("Next steps:")
    print("  1. Run scripts/prepare_xlam_training_data.py to convert to Qwen3-VL format")
    print("  2. Merge with neulab agent data for comprehensive training")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Salesforce xlam-function-calling-60k dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="agent/training/data",
        help="Output directory for dataset files"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download sample data for testing (1000 examples)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to download"
    )
    
    args = parser.parse_args()
    
    download_xlam_dataset(
        output_dir=args.output_dir,
        sample=args.sample,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download and prepare neulab/agent-data-collection dataset for zen-vl agent training.

This dataset contains ~100K-1M multi-turn agent dialogues across 18 domains:
- ALFWorld (household tasks)
- Database operations
- Knowledge graphs
- Web navigation (Mind2Web, WebShop)
- Operating system interactions
- Code generation and feedback
- Software engineering trajectories

Dataset: https://huggingface.co/datasets/neulab/agent-data-collection
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_agent_data(output_dir: str = "agent/training/data", sample: bool = False, max_per_config: int = None):
    """
    Download the neulab/agent-data-collection dataset.

    Args:
        output_dir: Directory to save the processed dataset
        sample: If True, download only a small sample from each config
        max_per_config: Maximum examples per config (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Downloading neulab/agent-data-collection dataset...")
    if sample:
        print("(Sample mode: downloading limited examples for testing)")
    print("=" * 80)

    # Dataset configurations to download
    if sample:
        # Sample mode: download just a couple configs for testing
        configs = [
            "code_feedback",
            "codeactinstruct",
        ]
        if max_per_config is None:
            max_per_config = 1000  # Limit to 1000 examples per config in sample mode
    else:
        configs = [
            "agenttuning_alfworld",
            "agenttuning_db",
            "agenttuning_kg",
            "agenttuning_mind2web",
            "agenttuning_os",
            "agenttuning_webshop",
            "code_feedback",
            "codeactinstruct",
            "mind2web",
            "openhands",
            "synatra",
            "weblinx",
            # Add more configs as needed
        ]

    all_data = []

    for config_name in configs:
        try:
            print(f"\nDownloading {config_name}...")

            # Load dataset configuration
            dataset = load_dataset(
                "neulab/agent-data-collection",
                config_name,
                split="train",
                trust_remote_code=True
            )

            print(f"Loaded {len(dataset)} examples from {config_name}")

            # Convert to list and add source information
            items_to_process = dataset if max_per_config is None else dataset.select(range(min(max_per_config, len(dataset))))

            for item in tqdm(items_to_process, desc=f"Processing {config_name}"):
                # Add source tag
                item['source'] = config_name
                all_data.append(item)

        except Exception as e:
            print(f"Error loading {config_name}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print(f"Total examples collected: {len(all_data)}")
    print(f"{'=' * 80}")

    # Save combined dataset
    combined_file = output_path / "agent_data_combined.jsonl"
    print(f"\nSaving combined dataset to {combined_file}...")

    with open(combined_file, 'w', encoding='utf-8') as f:
        for item in tqdm(all_data, desc="Writing JSONL"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(all_data)} examples")

    # Create statistics
    stats = {
        'total_examples': len(all_data),
        'configs': {},
    }

    for config in configs:
        count = sum(1 for item in all_data if item.get('source') == config)
        if count > 0:
            stats['configs'][config] = count

    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("Dataset Statistics:")
    print(f"{'=' * 80}")
    for config, count in sorted(stats['configs'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {config:40s}: {count:>8,} examples")
    print(f"{'=' * 80}")
    print(f"  {'TOTAL':40s}: {stats['total_examples']:>8,} examples")
    print(f"{'=' * 80}")

    # Create README
    readme_content = f"""# Agent Data Collection Dataset

## Overview

This dataset contains multi-turn agent dialogues for training vision-language models
with agent capabilities. Downloaded from:
https://huggingface.co/datasets/neulab/agent-data-collection

## Statistics

- **Total Examples**: {stats['total_examples']:,}
- **Configurations**: {len(stats['configs'])}

## Breakdown by Configuration

"""

    for config, count in sorted(stats['configs'].items(), key=lambda x: x[1], reverse=True):
        readme_content += f"- **{config}**: {count:,} examples\n"

    readme_content += f"""
## Files

- `agent_data_combined.jsonl`: All examples combined
- `dataset_stats.json`: Detailed statistics
- `README.md`: This file

## Data Format

Each line in the JSONL file contains:
- `id`: Unique identifier
- `conversations`: Array of dialogue turns with "from" and "value" fields
- `system`: System prompt
- `split`: Train/eval partition
- `source`: Configuration name (added by our script)

## Usage

This dataset is used to train the zen-vl agent variant for:
- Multi-step reasoning and planning
- Tool use and function calling
- Interactive task completion
- Visual agent capabilities

## Citation

```bibtex
@misc{{neulab2024agentdata,
  title={{Agent Data Collection}},
  author={{NeuLab @ LTI/CMU}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/neulab/agent-data-collection}}
}}
```
"""

    readme_file = output_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"\n✓ Dataset download complete!")
    print(f"  - Combined data: {combined_file}")
    print(f"  - Statistics: {stats_file}")
    print(f"  - README: {readme_file}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download neulab/agent-data-collection dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="agent/training/data",
        help="Output directory for dataset (default: agent/training/data)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download only a sample for testing (default: False)"
    )
    parser.add_argument(
        "--max-per-config",
        type=int,
        default=None,
        help="Maximum examples per config (default: None = all)"
    )

    args = parser.parse_args()
    download_agent_data(args.output_dir, args.sample, args.max_per_config)

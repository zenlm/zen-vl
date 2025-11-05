"""Download remaining ADP configs (skip problematic ones)."""
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Remaining configs to download
REMAINING_CONFIGS = [
    'code_feedback',
    'go-browse-wa'
]

output_dir = Path("/Users/z/work/zen/zen-vl/data/adp_full")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"Downloading {len(REMAINING_CONFIGS)} remaining ADP configs")
print("="*80)

for config in tqdm(REMAINING_CONFIGS, desc="Downloading configs"):
    output_file = output_dir / f"{config}.json"
    
    if output_file.exists():
        print(f"\n✅ {config} already exists, skipping")
        continue
        
    try:
        print(f"\n📥 Downloading {config}...")
        dataset = load_dataset("neulab/agent-data-collection", config, split="train")
        
        # Convert to JSON
        examples = [dict(example) for example in dataset]
        
        with open(output_file, 'w') as f:
            json.dump(examples, f)
        
        print(f"   ✅ Saved {len(examples):,} trajectories to {output_file}")
        
    except Exception as e:
        print(f"   ❌ Error downloading {config}: {e}")

print("\n✅ Remaining configs download complete!")

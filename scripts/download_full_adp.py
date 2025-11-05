"""
Download all 18 configs of the ADP dataset from HuggingFace.

Total dataset size: ~1.3M trajectories across:
- AgentTuning (OS, KG, DB, Mind2Web, AlfWorld, WebShop)
- OpenHands
- SWE-Gym, SWE-smith, Nebius SWE-agent
- Mind2Web, Synatra
- NNetNav (live, wa)
- CodeActInstruct
- Orca AgentInstruct
- Code-Feedback
- Go-Browse
"""

from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm

# All 18 configs from neulab/agent-data-collection
CONFIGS = [
    'agenttuning_os',
    'agenttuning_kg',
    'agenttuning_db',
    'agenttuning_mind2web',
    'agenttuning_alfworld',
    'agenttuning_webshop',
    'openhands',
    'SWE-Gym_OpenHands-Sampled-Trajectories',
    'SWE-smith_5kTrajectories',
    'nebius_SWE-agent-trajectories',
    'mind2web',
    'synatra',
    'nnetnav-live',
    'nnetnav-wa',
    'codeactinstruct',
    'orca_agentinstruct',
    'code_feedback',
    'go-browse-wa'
]

def download_all_configs(output_dir="/Users/z/work/zen/zen-vl/data/adp_full"):
    """Download all ADP dataset configs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_trajectories": 0,
        "by_config": {}
    }
    
    print("="*80)
    print("Downloading Full ADP Dataset (18 configs)")
    print("="*80)
    
    for config in tqdm(CONFIGS, desc="Downloading configs"):
        try:
            print(f"\n📥 Downloading {config}...")
            
            dataset = load_dataset(
                "neulab/agent-data-collection",
                config,
                split="train"
            )
            
            count = len(dataset)
            stats["by_config"][config] = count
            stats["total_trajectories"] += count
            
            # Save to JSON
            config_path = output_path / f"{config}.json"
            
            # Convert to list of dicts
            data = [example for example in dataset]
            
            with open(config_path, 'w') as f:
                json.dump(data, f)
            
            print(f"   ✅ Saved {count:,} trajectories to {config_path}")
            
        except Exception as e:
            print(f"   ❌ Error downloading {config}: {e}")
            stats["by_config"][config] = 0
    
    # Save stats
    stats_path = output_path / "download_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("📊 Download Complete!")
    print("="*80)
    print(f"Total trajectories: {stats['total_trajectories']:,}")
    print(f"\nBreakdown by config:")
    for config, count in sorted(stats["by_config"].items(), key=lambda x: -x[1]):
        print(f"  {config:45s}: {count:,}")
    print(f"\n💾 Data saved to: {output_path}")

if __name__ == "__main__":
    download_all_configs()

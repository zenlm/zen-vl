"""
Download and convert Salesforce xLAM Function Calling 60k dataset.

Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
Paper: xLAM - A family of large action models

This dataset contains 60k high-quality function calling examples across:
- API/tool use
- Multi-step reasoning
- Parameter extraction
- Complex workflows

We'll convert it to ADP format and merge with the neulab ADP dataset
for comprehensive agent training.
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))
from adp_schema import (
    Trajectory, TextObservation, APIAction, MessageAction,
    ObservationSource
)


class XLAMConverter:
    """Convert xLAM function calling data to ADP format."""
    
    def __init__(self, output_dir="/Users/z/work/zen/zen-vl/data/xlam"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total": 0,
            "with_functions": 0,
            "multi_turn": 0,
            "avg_turns": 0
        }
    
    def download_xlam(self):
        """Download xLAM dataset from HuggingFace."""
        print("="*80)
        print("Downloading xLAM Function Calling Dataset (60k)")
        print("="*80)
        
        try:
            # Use HuggingFace token for gated dataset access
            dataset = load_dataset(
                "Salesforce/xlam-function-calling-60k", 
                split="train",
                token=True  # Use saved HF token
            )
            print(f"✅ Downloaded {len(dataset):,} examples")
            return dataset
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            print("💡 Make sure you have access at: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k")
            return None
    
    def convert_to_adp(self, example: dict) -> Trajectory:
        """
        Convert single xLAM example to ADP Trajectory format.
        
        xLAM format:
        {
            "id": "...",
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "...function_call..."},
                ...
            ],
            "tools": [...function definitions...]
        }
        """
        trajectory_id = str(example.get("id", f"xlam_{self.stats['total']}"))
        conversations = example.get("conversations", [])
        tools = example.get("tools", [])
        
        # Build ADP content
        content = []
        
        # Add tools as initial context (optional)
        if tools:
            tools_text = "Available tools:\n" + json.dumps(tools, indent=2)
            content.append(TextObservation(
                source=ObservationSource.SYSTEM,
                content=tools_text
            ))
        
        # Convert conversations
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            
            if role == "human":
                # User query
                content.append(TextObservation(
                    source=ObservationSource.USER,
                    content=value
                ))
                
            elif role == "gpt":
                # Assistant response - may contain function calls
                if "function_call" in value or "```python" in value:
                    # Try to parse function call
                    try:
                        # xLAM uses various formats, try to extract
                        if "function_call" in value:
                            # Extract function call JSON
                            import re
                            match = re.search(r'\{[^}]*"name"[^}]*\}', value)
                            if match:
                                func_data = json.loads(match.group())
                                content.append(APIAction(
                                    function=func_data.get("name", "unknown"),
                                    kwargs=func_data.get("arguments", {}),
                                    description=value[:200]  # Keep reasoning
                                ))
                                continue
                    except:
                        pass
                
                # Regular message
                content.append(MessageAction(
                    content=value
                ))
        
        # Ensure alternating structure by adding environment responses
        # after each assistant action
        fixed_content = []
        for i, item in enumerate(content):
            fixed_content.append(item)
            
            # Add environment response after API actions
            if isinstance(item, APIAction):
                # Add simulated success response
                fixed_content.append(TextObservation(
                    source=ObservationSource.ENVIRONMENT,
                    content="Function executed successfully"
                ))
        
        trajectory = Trajectory(
            id=trajectory_id,
            content=fixed_content,
            details={
                "dataset": "xlam-function-calling-60k",
                "num_tools": len(tools),
                "num_turns": len(conversations)
            }
        )
        
        # Update stats
        if tools:
            self.stats["with_functions"] += 1
        if len(conversations) > 2:
            self.stats["multi_turn"] += 1
        
        return trajectory
    
    def convert_dataset(self, dataset):
        """Convert entire xLAM dataset to ADP format."""
        print("\n🔄 Converting xLAM to ADP format...")
        
        trajectories = []
        total_turns = 0
        
        for example in tqdm(dataset, desc="Converting"):
            try:
                trajectory = self.convert_to_adp(example)
                trajectories.append(trajectory)
                self.stats["total"] += 1
                total_turns += len(trajectory.content)
            except Exception as e:
                print(f"⚠️  Error converting {example.get('id', 'unknown')}: {e}")
        
        self.stats["avg_turns"] = total_turns / max(self.stats["total"], 1)
        
        # Save as JSON
        output_file = self.output_dir / "xlam_adp_format.json"
        
        with open(output_file, 'w') as f:
            json.dump([
                {
                    "id": t.id,
                    "content": [
                        {
                            "type": item.type,
                            **{k: v for k, v in item.__dict__.items() if k != "type"}
                        }
                        for item in t.content
                    ],
                    "details": t.details
                }
                for t in trajectories
            ], f, indent=2)
        
        print(f"\n✅ Saved {len(trajectories):,} trajectories to {output_file}")
        
        # Save stats
        stats_file = self.output_dir / "xlam_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        return trajectories


def main():
    """Main conversion pipeline."""
    print("="*80)
    print("xLAM Function Calling Dataset → ADP Format")
    print("="*80)
    
    converter = XLAMConverter()
    
    # Download
    dataset = converter.download_xlam()
    if dataset is None:
        print("❌ Download failed, exiting")
        return
    
    # Convert
    trajectories = converter.convert_dataset(dataset)
    
    print("\n" + "="*80)
    print("📊 Conversion Statistics")
    print("="*80)
    print(f"Total trajectories: {converter.stats['total']:,}")
    print(f"With function calls: {converter.stats['with_functions']:,}")
    print(f"Multi-turn: {converter.stats['multi_turn']:,}")
    print(f"Avg turns per trajectory: {converter.stats['avg_turns']:.1f}")
    
    print("\n✅ xLAM conversion complete!")


if __name__ == "__main__":
    main()

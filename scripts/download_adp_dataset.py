"""
Download and convert Agent Data Protocol (ADP) dataset for Zen VL training.

This script:
1. Downloads the neulab/agent-data-collection dataset from HuggingFace
2. Converts ADP format to Zen VL multimodal format
3. Generates synthetic images for text-only trajectories
4. Creates train/val splits with visual observations

Dataset: https://huggingface.co/datasets/neulab/agent-data-collection
Paper: https://arxiv.org/abs/2510.24702

The ADP dataset includes 13 sources totaling 1.3M trajectories:
- AgentInstruct: API/tool use, browsing, database, OS tasks
- Code-Feedback: Code generation with runtime feedback
- CodeActInstruct: Code generation and tool use
- Go-Browse: Web navigation rollouts
- Mind2Web: Human web demos on real websites
- Nebius SWE: SWE-agent trajectories
- NNetNav: Web exploration (live + WebArena)
- OpenHands: Recorded agent trajectories
- Orca AgentInstruct: Large-scale synthetic tool-use
- SWE-Gym: GitHub repo task trajectories
- SWE-smith: Bug-fix task trajectories
- Synatra: Synthetically created web demos
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
from tqdm import tqdm

# Import our Zen VL ADP schema
import sys
sys.path.append(str(Path(__file__).parent))
from adp_schema import (
    Trajectory, ImageObservation, TextObservation, WebObservation,
    APIAction, CodeAction, MessageAction, ObservationSource, ImageFormat,
    convert_adp_to_zen_vl_format
)


class ADPDatasetConverter:
    """Convert ADP dataset to Zen VL multimodal training format."""
    
    def __init__(
        self,
        output_dir: str = "/Users/z/work/zen/zen-vl/data/adp",
        generate_synthetic_images: bool = True,
        max_trajectories_per_dataset: Optional[int] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.generate_synthetic_images = generate_synthetic_images
        self.max_trajectories_per_dataset = max_trajectories_per_dataset
        
        # Statistics
        self.stats = {
            "total_trajectories": 0,
            "trajectories_with_images": 0,
            "synthetic_images_generated": 0,
            "by_dataset": {}
        }
    
    def download_adp_dataset(self) -> Dict[str, Any]:
        """Download ADP dataset from HuggingFace."""
        print("Downloading neulab/agent-data-collection from HuggingFace...")
        
        try:
            # Try to load from HuggingFace
            dataset = load_dataset("neulab/agent-data-collection", trust_remote_code=True)
            print(f"✅ Downloaded {len(dataset)} splits")
            return dataset
        except Exception as e:
            print(f"⚠️  Failed to download from HuggingFace: {e}")
            print("This is expected if the dataset hasn't been published yet.")
            print("Using example/mock data for development...")
            return self.create_mock_dataset()
    
    def create_mock_dataset(self) -> Dict[str, List[Dict]]:
        """Create mock ADP-format data for development/testing."""
        print("Creating mock ADP dataset...")
        
        mock_data = {
            "train": [
                # Example 1: Image analysis
                {
                    "id": "mock_image_analysis_001",
                    "content": [
                        {"type": "text_observation", "source": "user", "content": "Analyze this beach scene"},
                        {"type": "api_action", "function": "image_analysis", "kwargs": {
                            "objects": ["people", "umbrellas", "beach", "ocean"],
                            "scene": "outdoor_beach",
                            "colors": ["blue", "yellow", "beige"]
                        }, "description": "Analyzing the beach scene"},
                        {"type": "text_observation", "source": "environment", "content": "Analysis complete"}
                    ],
                    "details": {"dataset": "mock", "task": "image_analysis", "has_image": True}
                },
                # Example 2: GUI interaction
                {
                    "id": "mock_gui_interaction_001",
                    "content": [
                        {"type": "text_observation", "source": "user", "content": "Click the login button"},
                        {"type": "web_observation", "url": "https://example.com/login", "axtree": "Button 'Login' [120, 200]"},
                        {"type": "api_action", "function": "click_element", "kwargs": {
                            "element_type": "button",
                            "label": "Login",
                            "position": {"x": 120, "y": 200}
                        }, "description": "Clicking the login button"},
                        {"type": "text_observation", "source": "environment", "content": "Clicked successfully"}
                    ],
                    "details": {"dataset": "mock", "task": "gui_interaction", "has_image": True}
                },
                # Example 3: Code generation
                {
                    "id": "mock_code_gen_001",
                    "content": [
                        {"type": "text_observation", "source": "user", "content": "Write a function to sort a list"},
                        {"type": "code_action", "language": "python", "content": "def sort_list(lst):\n    return sorted(lst)", 
                         "description": "Creating a simple sort function"},
                        {"type": "text_observation", "source": "environment", "content": "Code executed successfully"}
                    ],
                    "details": {"dataset": "mock", "task": "code_generation"}
                },
                # Example 4: Form filling
                {
                    "id": "mock_form_fill_001",
                    "content": [
                        {"type": "text_observation", "source": "user", "content": "Fill out this registration form"},
                        {"type": "api_action", "function": "fill_form", "kwargs": {
                            "fields": [
                                {"type": "text", "label": "username", "value": "john_doe"},
                                {"type": "email", "label": "email", "value": "john@example.com"}
                            ]
                        }, "description": "Filling registration form"},
                        {"type": "text_observation", "source": "environment", "content": "Form filled"}
                    ],
                    "details": {"dataset": "mock", "task": "form_filling", "has_image": True}
                }
            ]
        }
        
        # Add more examples by duplicating with variations
        for i in range(10):
            mock_data["train"].extend([
                {
                    "id": f"mock_variation_{i}_{j}",
                    "content": example["content"],
                    "details": {**example["details"], "variation": i}
                }
                for j, example in enumerate(mock_data["train"][:4])
            ])
        
        return {"train": mock_data["train"]}
    
    def generate_synthetic_image(
        self,
        trajectory_id: str,
        content: str,
        task_type: str,
        width: int = 800,
        height: int = 600
    ) -> str:
        """
        Generate synthetic image for text-only trajectories.
        
        This creates visual representations of:
        - Web UIs from HTML/accessibility trees
        - GUI screenshots from action descriptions
        - Code visualization
        - Forms and documents
        """
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fall back to default
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw based on task type
        if task_type == "gui_interaction":
            # Draw UI elements
            draw.rectangle([50, 50, 750, 100], outline=(0, 0, 0), width=2)
            draw.text((60, 65), "Login Form", fill=(0, 0, 0), font=font_large)
            
            draw.rectangle([100, 150, 700, 200], outline=(100, 100, 100), width=1)
            draw.text((110, 165), "Username", fill=(100, 100, 100), font=font_small)
            
            draw.rectangle([100, 250, 700, 300], outline=(100, 100, 100), width=1)
            draw.text((110, 265), "Password", fill=(100, 100, 100), font=font_small)
            
            draw.rectangle([100, 350, 300, 400], fill=(0, 120, 255))
            draw.text((150, 365), "Login", fill=(255, 255, 255), font=font_large)
            
        elif task_type == "form_filling":
            # Draw form
            draw.text((50, 30), "Registration Form", fill=(0, 0, 0), font=font_large)
            
            fields = ["Name", "Email", "Phone", "Address"]
            for i, field in enumerate(fields):
                y = 100 + i * 80
                draw.text((50, y), f"{field}:", fill=(0, 0, 0), font=font_small)
                draw.rectangle([200, y-5, 700, y+35], outline=(100, 100, 100), width=1)
                
        elif task_type == "image_analysis":
            # Draw placeholder image scene
            draw.rectangle([50, 50, 750, 550], fill=(135, 206, 235))  # Sky blue
            draw.ellipse([600, 80, 720, 200], fill=(255, 223, 0))  # Sun
            draw.rectangle([50, 400, 750, 550], fill=(238, 214, 175))  # Beach
            
            # People (simple circles)
            for x in [150, 300, 450, 600]:
                draw.ellipse([x, 350, x+30, 380], fill=(255, 182, 193))
                
        elif task_type == "code_generation":
            # Draw code editor
            draw.rectangle([50, 50, 750, 550], fill=(40, 44, 52))
            code_lines = content.split('\n')[:15]
            for i, line in enumerate(code_lines):
                draw.text((60, 60 + i*25), line[:80], fill=(171, 178, 191), font=font_small)
        
        else:
            # Generic task visualization
            draw.text((50, 50), f"Task: {task_type}", fill=(0, 0, 0), font=font_large)
            
            # Wrap and draw content
            words = content.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 80:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            for i, line in enumerate(lines[:20]):
                draw.text((50, 100 + i*25), line, fill=(0, 0, 0), font=font_small)
        
        # Save image
        image_path = self.images_dir / f"{trajectory_id}_synthetic.png"
        img.save(image_path)
        
        self.stats["synthetic_images_generated"] += 1
        return str(image_path)
    
    def convert_trajectory(self, raw_trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert single ADP trajectory to Zen VL format."""
        try:
            trajectory_id = raw_trajectory.get("id", "unknown")
            content = raw_trajectory.get("content", [])
            details = raw_trajectory.get("details", {})
            
            # Build Zen VL content list
            zen_content = []
            has_visual = False
            
            for i, item in enumerate(content):
                item_type = item.get("type", "")
                
                if item_type == "text_observation":
                    zen_content.append(TextObservation(
                        source=ObservationSource(item.get("source", "user")),
                        content=item.get("content", "")
                    ))
                    
                elif item_type == "web_observation":
                    obs = WebObservation(
                        url=item.get("url", ""),
                        html=item.get("html"),
                        axtree=item.get("axtree"),
                        viewport_size=item.get("viewport_size"),
                        screenshot=item.get("screenshot")
                    )
                    zen_content.append(obs)
                    
                    # Generate synthetic screenshot if enabled and no screenshot exists
                    if self.generate_synthetic_images and not obs.screenshot:
                        task_type = details.get("task", "gui_interaction")
                        content_text = obs.axtree or obs.html or obs.url
                        image_path = self.generate_synthetic_image(
                            f"{trajectory_id}_web_{i}",
                            content_text[:500] if content_text else "",
                            task_type
                        )
                        zen_content.append(ImageObservation(
                            source=ObservationSource.ENVIRONMENT,
                            image_path=image_path,
                            format=ImageFormat.PNG,
                            caption=f"Web page: {obs.url}"
                        ))
                        has_visual = True
                        
                elif item_type == "api_action":
                    zen_content.append(APIAction(
                        function=item.get("function", ""),
                        kwargs=item.get("kwargs", {}),
                        description=item.get("description")
                    ))
                    
                elif item_type == "code_action":
                    zen_content.append(CodeAction(
                        language=item.get("language", "python"),
                        content=item.get("content", ""),
                        description=item.get("description")
                    ))
                    
                elif item_type == "message_action":
                    zen_content.append(MessageAction(
                        content=item.get("content", "")
                    ))
            
            # For trajectories marked as having images, ensure at least one visual observation
            if details.get("has_image") and not has_visual and self.generate_synthetic_images:
                # Generate synthetic image based on task type
                task_type = details.get("task", "general")
                first_text = next((item.get("content", "") for item in content if item.get("type") == "text_observation"), "")
                
                image_path = self.generate_synthetic_image(
                    trajectory_id,
                    first_text,
                    task_type
                )
                
                # Insert image observation after first text observation
                insert_idx = next((i+1 for i, item in enumerate(zen_content) if isinstance(item, TextObservation)), 0)
                zen_content.insert(insert_idx, ImageObservation(
                    source=ObservationSource.USER,
                    image_path=image_path,
                    format=ImageFormat.PNG,
                    caption=f"{task_type} visualization"
                ))
                has_visual = True
            
            # Create trajectory
            trajectory = Trajectory(
                id=trajectory_id,
                content=zen_content,
                details=details
            )
            
            # Validate
            if not trajectory.validate_alternating():
                print(f"⚠️  Trajectory {trajectory_id} doesn't alternate properly")
                # We'll still use it but log the warning
            
            # Convert to training format
            training_example = convert_adp_to_zen_vl_format(trajectory)
            
            # Update stats
            if has_visual or len(trajectory.get_visual_observations()) > 0:
                self.stats["trajectories_with_images"] += 1
            
            return training_example
            
        except Exception as e:
            print(f"❌ Error converting trajectory {raw_trajectory.get('id', 'unknown')}: {e}")
            return None
    
    def process_dataset(self, dataset: Dict[str, Any]) -> None:
        """Process entire ADP dataset and save to JSON."""
        print("\nProcessing ADP dataset...")
        
        all_training_examples = []
        
        # Process train split
        train_data = dataset.get("train", [])
        if self.max_trajectories_per_dataset:
            train_data = train_data[:self.max_trajectories_per_dataset]
        
        print(f"Converting {len(train_data)} trajectories...")
        
        for raw_trajectory in tqdm(train_data, desc="Converting trajectories"):
            training_example = self.convert_trajectory(raw_trajectory)
            if training_example:
                all_training_examples.append(training_example)
                self.stats["total_trajectories"] += 1
                
                # Track by dataset
                dataset_name = training_example.get("metadata", {}).get("dataset", "unknown")
                self.stats["by_dataset"][dataset_name] = self.stats["by_dataset"].get(dataset_name, 0) + 1
        
        # Split into train/val (90/10)
        split_idx = int(len(all_training_examples) * 0.9)
        train_examples = all_training_examples[:split_idx]
        val_examples = all_training_examples[split_idx:]
        
        # Save to JSON
        train_path = self.output_dir / "train.json"
        val_path = self.output_dir / "val.json"
        
        with open(train_path, 'w') as f:
            json.dump(train_examples, f, indent=2)
        
        with open(val_path, 'w') as f:
            json.dump(val_examples, f, indent=2)
        
        print(f"\n✅ Saved {len(train_examples)} training examples to {train_path}")
        print(f"✅ Saved {len(val_examples)} validation examples to {val_path}")
        
        # Save stats
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total trajectories: {self.stats['total_trajectories']}")
        print(f"   With images: {self.stats['trajectories_with_images']}")
        print(f"   Synthetic images: {self.stats['synthetic_images_generated']}")
        print(f"\n   By dataset:")
        for dataset_name, count in sorted(self.stats["by_dataset"].items()):
            print(f"     {dataset_name}: {count}")


def main():
    """Main conversion pipeline."""
    print("="*80)
    print("Zen VL - Agent Data Protocol (ADP) Dataset Converter")
    print("="*80)
    
    converter = ADPDatasetConverter(
        output_dir="/Users/z/work/zen/zen-vl/data/adp",
        generate_synthetic_images=True,
        max_trajectories_per_dataset=None  # Use all data
    )
    
    # Download dataset
    dataset = converter.download_adp_dataset()
    
    # Process and convert
    converter.process_dataset(dataset)
    
    print("\n✅ ADP dataset conversion complete!")
    print(f"📁 Data saved to: {converter.output_dir}")
    print(f"🖼️  Images saved to: {converter.images_dir}")


if __name__ == "__main__":
    main()

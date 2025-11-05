#!/usr/bin/env python3.13
"""
Create HuggingFace model repositories for Zen VL
"""
import os
from huggingface_hub import HfApi, create_repo

def create_model_repos():
    """Create HuggingFace repos for all Zen VL models"""
    
    api = HfApi()
    
    # Model repos to create
    models = [
        ("zen-vl-4b-instruct", "Zen VL 4B Instruct - Vision-language model with Zen identity and multimodal understanding"),
        ("zen-vl-4b-agent", "Zen VL 4B Agent - Vision-language model with function calling and tool use capabilities"),
        ("zen-vl-8b-instruct", "Zen VL 8B Instruct - Vision-language model with Zen identity (9B params)"),
        ("zen-vl-8b-agent", "Zen VL 8B Agent - Vision-language model with function calling (9B params)"),
        ("zen-vl-30b-instruct", "Zen VL 30B Instruct - Frontier vision-language model with Zen identity (31B MoE)"),
        ("zen-vl-30b-agent", "Zen VL 30B Agent - Frontier vision-language model with function calling (31B MoE)"),
    ]
    
    created = []
    skipped = []
    
    for repo_name, description in models:
        full_repo_id = f"zenlm/{repo_name}"
        
        try:
            print(f"Creating {full_repo_id}...")
            
            # Create repo
            url = create_repo(
                repo_id=full_repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            
            # Update repo card
            readme = f"""---
license: apache-2.0
tags:
- vision-language
- multimodal
- function-calling
- visual-agents
- qwen3-vl
- zen
language:
- en
- multilingual
base_model:
- Qwen/Qwen3-VL-{repo_name.split('-')[2].upper()}-Instruct
library_name: transformers
pipeline_tag: image-text-to-text
---

# {repo_name.replace('-', ' ').title()}

{description}

## Model Details

- **Architecture**: Qwen3-VL
- **Parameters**: {repo_name.split('-')[2].upper()}
- **Context Window**: 256K tokens (expandable to 1M)
- **License**: Apache 2.0
- **Training**: Fine-tuned with Zen identity and {"function calling" if "agent" in repo_name else "instruction following"}

## Capabilities

- 🎨 **Visual Understanding**: Image analysis, video comprehension, spatial reasoning
- 📝 **OCR**: Text extraction in 32 languages
- 🧠 **Multimodal Reasoning**: STEM, math, code generation
{"- 🛠️ **Function Calling**: Tool use with visual context" if "agent" in repo_name else ""}
{"- 🤖 **Visual Agents**: GUI interaction, parameter extraction" if "agent" in repo_name else ""}

## Usage

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "{full_repo_id}",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("{full_repo_id}")

# Process image
image = Image.open("example.jpg")
prompt = "What's in this image?"

messages = [{{"role": "user", "content": prompt}}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Links

- 🌐 **Website**: [zenlm.org](https://zenlm.org)
- 📚 **GitHub**: [zenlm/zen-vl](https://github.com/zenlm/zen-vl)
- 📄 **Paper**: Coming soon
- 🤗 **Model Family**: [zenlm](https://huggingface.co/zenlm)

## Citation

```bibtex
@misc{{zenvl2025,
  title={{Zen VL: Vision-Language Models with Integrated Function Calling}},
  author={{Hanzo AI Team}},
  year={{2025}},
  publisher={{Zen Language Models}},
  url={{https://github.com/zenlm/zen-vl}}
}}
```

## License

Apache 2.0

---

Created by [Hanzo AI](https://hanzo.ai) for the Zen model family.
"""
            
            # Upload README
            api.upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=full_repo_id,
                repo_type="model",
                commit_message=f"Initialize {repo_name} model card"
            )
            
            created.append(full_repo_id)
            print(f"✅ Created {full_repo_id}")
            
        except Exception as e:
            print(f"⚠️  {full_repo_id}: {e}")
            skipped.append(full_repo_id)
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"✅ Created: {len(created)} repos")
    for repo in created:
        print(f"   - https://huggingface.co/{repo}")
    
    if skipped:
        print(f"\n⚠️  Skipped: {len(skipped)} repos")
        for repo in skipped:
            print(f"   - {repo}")

if __name__ == "__main__":
    print("="*60)
    print("Creating HuggingFace Model Repositories")
    print("="*60)
    print()
    
    create_model_repos()
    
    print("\n🎉 HuggingFace repos ready!")

# Zen VL - Vision-Language Models with Function Calling

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/zenlm)
[![Website](https://img.shields.io/badge/🌐-zenlm.org-green.svg)](https://zenlm.org)

Vision-language models from the Zen family, built on a vision-language transformer architecture with Zen identity and function calling capabilities.

## Models

| Model | Size | Context | Variants | Use Case |
|-------|------|---------|----------|----------|
| **zen-vl-4b** | 4B | 256K→1M | instruct, thinking, agent | Edge/mobile VL |
| **zen-vl-8b** | 9B | 256K→1M | instruct, thinking, agent | General VL |
| **zen-vl-30b** | 31B MoE | 256K→1M | instruct, thinking, agent | Frontier VL |

## Features

✨ **Vision + Language**: Image analysis, video understanding, OCR in 32 languages  
🧠 **Chain-of-Thought**: Transparent reasoning with `<thinking>` tags  
🛠️ **Function Calling**: Tool use with visual context  
🎯 **Visual Agent**: GUI navigation, element recognition  
⚡ **Extended Context**: 256K native, expandable to 1M tokens  

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/zenlm/zen-vl.git
cd zen-vl

# Install dependencies
pip install -r requirements.txt
```

### Download Models

```bash
# Download base models
make download-4b   # ~8GB
make download-8b   # ~18GB  
make download-30b  # ~62GB

# Or use huggingface-cli directly
huggingface-cli download zenlm/zen-vl-4b-instruct --local-dir instruct/base-model
```

### Training

```bash
# Train all variants
make all

# Or train individually
make train-instruct   # Zen identity
make train-thinking   # Chain-of-thought
make train-agent      # Function calling
```

### Inference

```python
from transformers import ZenVLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model
model = ZenVLForConditionalGeneration.from_pretrained(
    "zenlm/zen-vl-4b-instruct",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("zenlm/zen-vl-4b-instruct")

# Process image and text
image = Image.open("example.jpg")
prompt = "What's in this image?"

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### Function Calling

```python
# Visual tool calling
tools = [
    {
        "name": "image_analysis",
        "description": "Analyze image content",
        "parameters": {
            "objects": {"type": "array"},
            "scene": {"type": "string"}
        }
    }
]

prompt = f"Analyze this image and call the appropriate tool.\n\nTools: {tools}"
# Model will generate structured function call based on image content
```

## Model Variants

### Instruct
Base instruction-following model with Zen identity.

```python
model = ZenVLForConditionalGeneration.from_pretrained("zenlm/zen-vl-4b-instruct")
```

### Thinking
Chain-of-thought reasoning with transparent thinking process.

```python
model = ZenVLForConditionalGeneration.from_pretrained("zenlm/zen-vl-4b-thinking")
# Outputs include <thinking>...</thinking> tags
```

### Agent
Function calling and tool use with visual context.

```python
model = ZenVLForConditionalGeneration.from_pretrained("zenlm/zen-vl-4b-agent")
# Returns structured JSON for tool calls
```

## GGUF Quantization

```bash
# Convert to GGUF
make gguf

# Available quantizations
# Q4_K_M: ~2GB (4B model)
# Q5_K_M: ~2.5GB
# Q8_0: ~4GB
# F16: ~8GB
```

## MLX (Apple Silicon)

```bash
# Convert for MLX
make mlx

# Optimized for M1/M2/M3 chips
```

## Capabilities

### Visual Understanding
- 📸 **Image Analysis**: Object detection, scene understanding
- 🎥 **Video Understanding**: Temporal reasoning, frame-level analysis
- 📝 **OCR**: 32 languages, challenging text conditions
- 🎯 **Spatial Grounding**: 2D/3D object localization
- 🖱️ **GUI Recognition**: UI elements, buttons, forms

### Language Generation  
- 💬 **Text Comprehension**: On par with pure LLMs
- 💻 **Code Generation**: HTML/CSS/JS from images
- 🧮 **STEM Reasoning**: Math, science problems
- 🔗 **Multimodal**: Seamless text-vision fusion

### Agent Tasks
- 🛠️ **Tool Calling**: Function use with visual context
- 📋 **Structured Output**: JSON/XML generation
- 🤖 **GUI Interaction**: Navigate and manipulate interfaces
- 🎯 **Task Completion**: Multi-step visual workflows

## Training Data

### Identity Dataset
```python
{
  "instruction": "Who are you?",
  "image": "zen_logo.png",
  "output": "I'm Zen VL, a vision-language model from the Zen family..."
}
```

### Function Calling Dataset
```python
{
  "instruction": "Analyze this form",
  "image": "form.png",
  "output": {
    "function": "fill_form",
    "parameters": {...}
  }
}
```

## Benchmarks

Coming soon: Performance on vision-language and agent benchmarks.

## Research Paper

**Title**: "Zen VL: Vision-Language Models with Integrated Function Calling"

**Contributions**:
1. First open VL models with native function calling
2. Multimodal identity fine-tuning methodology  
3. Visual agent task benchmarking
4. 4B/8B/30B comparative analysis

## Citation

```bibtex
@misc{zenvl2025,
  title={Zen VL: Vision-Language Models with Integrated Function Calling},
  author={Hanzo AI Team},
  year={2025},
  publisher={Zen Language Models},
  url={https://github.com/zenlm/zen-vl}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) file

## Links

- 🌐 **Website**: [zenlm.org](https://zenlm.org)
- 🤗 **Models**: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- 📚 **Docs**: [docs/](docs/)
- 📄 **Paper**: [paper/](paper/)

## Credits

Created by [Hanzo AI](https://hanzo.ai) for the Zen model family.  
Developed by the Zen LM Authors.

---

**Zen VL**: Clarity Through Vision

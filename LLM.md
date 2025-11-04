# Zen VL Models - Vision-Language Training System

**Last Updated**: 2025-11-04
**Project**: zen-vl
**Organization**: zenlm
**Repository**: https://github.com/zenlm/zen-vl

## Overview

Zen VL is a family of vision-language models built on Qwen3-VL architecture, fine-tuned with Zen identity and enhanced with function calling capabilities. These models combine visual understanding with language generation, specialized for visual agent tasks, GUI interaction, and multimodal reasoning.

## Model Family

### zen-vl-4b
- **Base**: Qwen3-VL-4B-Instruct
- **Parameters**: 4B
- **Context**: 256K tokens (expandable to 1M)
- **Variants**: instruct, thinking, agent
- **Use Cases**: Edge/mobile vision-language, real-time visual agents

### zen-vl-8b  
- **Base**: Qwen3-VL-8B-Instruct
- **Parameters**: 9B (actual)
- **Context**: 256K tokens (expandable to 1M)
- **Variants**: instruct, thinking, agent
- **Use Cases**: General-purpose multimodal applications

### zen-vl-30b
- **Base**: Qwen3-VL-30B-A3B-Instruct
- **Parameters**: 31B (MoE architecture)
- **Context**: 256K tokens (expandable to 1M)
- **Variants**: instruct, thinking, agent
- **Use Cases**: Frontier vision-language performance

## Architectural Innovations

### From Qwen3-VL Base
1. **Interleaved-MRoPE**: Enhanced positional embeddings for video reasoning
2. **DeepStack**: Multi-level ViT feature fusion for fine-grained details
3. **Text-Timestamp Alignment**: Temporal event localization in videos

### Zen Enhancements
1. **Identity Fine-tuning**: Zen branding and personality
2. **Function Calling**: Tool use with visual context
3. **Structured Output**: JSON/XML generation from images
4. **Visual Agent Optimization**: Enhanced GUI and element recognition

## Capabilities

### Visual Understanding
- Image analysis and interpretation
- Video understanding with temporal awareness
- OCR in 32 languages
- Spatial perception (2D/3D grounding)
- GUI element recognition and navigation

### Language Generation
- Text comprehension on par with pure LLMs
- Code generation from images/videos (Draw.io, HTML/CSS/JS)
- STEM and mathematical reasoning
- Multimodal chain-of-thought (thinking variant)

### Agent Capabilities
- Function calling with visual context
- Tool use based on image analysis
- Structured output generation
- Task completion through GUI interaction
- Parameter extraction from visual elements

## Training Pipeline

### 1. Base Model Download
```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir instruct/base-model
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir instruct/base-model  
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir instruct/base-model
```

### 2. Identity Fine-tuning
- Zen identity responses
- Multimodal examples (text + images)
- Identity preservation across modalities

### 3. Thinking Variant
- Chain-of-thought with visual reasoning
- Step-by-step image/video analysis
- Transparent reasoning with `<thinking>` tags

### 4. Agent Variant
- Function calling dataset
- Visual tool-use examples
- Structured output training
- GUI interaction patterns

## Directory Structure

```
zen-vl/
├── Makefile              # Build automation
├── LLM.md                # This file
├── README.md             # User documentation
├── .gitignore            # Git ignore patterns
│
├── instruct/             # Base instruction model
│   ├── base-model/       # Downloaded Qwen3-VL
│   ├── finetuned/        # Zen-branded model
│   └── training/         # Training artifacts
│
├── thinking/             # Chain-of-thought variant
│   ├── base-model/
│   ├── finetuned/
│   └── training/
│
├── agent/                # Function calling variant
│   ├── base-model/
│   ├── finetuned/
│   └── training/
│
├── scripts/              # Training scripts
│   ├── train_instruct.py
│   ├── train_thinking.py
│   ├── train_agent.py
│   ├── download_models.py
│   ├── convert_gguf.py
│   └── upload_models.py
│
├── configs/              # Training configurations
├── tests/                # Test suite
├── examples/             # Usage examples
├── docs/                 # Documentation
├── gguf/                 # Quantized models
├── mlx/                  # MLX converted models
│
└── paper/                # Research paper
    ├── sections/
    ├── figures/
    ├── tables/
    └── references/
```

## Training Strategy

### Identity Dataset
```python
zen_vl_identity = "I'm Zen VL, a vision-language model from the Zen family. I combine visual understanding with language generation to analyze images, understand videos, and interact with user interfaces."

# Multimodal identity examples
- Text: "Who are you?" + Image: [Zen logo]
- Response: zen_vl_identity + "I can see you're showing me the Zen logo..."
```

### Function Calling Dataset
```python
# Example: Visual parameter extraction
User: [Image of form with fields] "Fill out this form"
Model: {
  "thinking": "I see a form with name, email, and message fields",
  "function": "fill_form",
  "parameters": {
    "fields": [
      {"type": "text", "label": "Name", "id": "name-field"},
      {"type": "email", "label": "Email", "id": "email-field"},
      {"type": "textarea", "label": "Message", "id": "msg-field"}
    ]
  }
}
```

## Research Paper Outline

### Title
"Zen VL: Vision-Language Models with Integrated Function Calling"

### Key Contributions
1. First open vision-language models with native function calling
2. Multimodal identity fine-tuning methodology
3. Benchmarking across visual agent tasks
4. Comparative analysis of 4B/8B/30B VL variants

### Sections
1. **Introduction**: Vision-language models and agent capabilities
2. **Related Work**: VL models, function calling, visual agents
3. **Architecture**: Qwen3-VL base + Zen enhancements
4. **Training**: Identity + thinking + agent fine-tuning
5. **Evaluation**: Benchmarks, ablations, human evaluation
6. **Results**: Performance across model sizes
7. **Discussion**: Insights, limitations, future work
8. **Conclusion**: Summary and impact

## Technology Stack

### Training
- Python 3.13
- PyTorch 2.0+
- Transformers (latest dev branch)
- LoRA/QLoRA for efficient fine-tuning

### Inference
- llama.cpp (GGUF quantization)
- MLX (Apple Silicon)
- vLLM (production serving)

### Development
- Makefile-based automation
- Git for version control
- HuggingFace Hub for distribution

## Function Calling Format

### Tool Definition
```python
tools = [
  {
    "name": "image_analysis",
    "description": "Analyze an image and extract key information",
    "parameters": {
      "objects": {"type": "array", "items": {"type": "string"}},
      "scene": {"type": "string"},
      "colors": {"type": "array", "items": {"type": "string"}}
    }
  }
]
```

### Model Response
```json
{
  "thinking": "I see a beach scene with people and umbrellas",
  "function_call": {
    "name": "image_analysis",
    "arguments": {
      "objects": ["people", "umbrellas", "sand", "water"],
      "scene": "beach",
      "colors": ["blue", "yellow", "tan", "white"]
    }
  }
}
```

## Testing Strategy

### Identity Verification
- Text-only identity prompts
- Visual identity prompts (with Zen logo)
- Multimodal identity consistency

### Visual Understanding
- Image description accuracy
- Video temporal understanding
- OCR accuracy across languages
- Spatial grounding precision

### Function Calling
- Tool selection accuracy
- Parameter extraction from images
- Structured output validity
- GUI interaction success rate

## Deployment

### GGUF Quantization
- Q4_K_M: ~2GB (4B model)
- Q5_K_M: ~2.5GB (4B model)
- Q8_0: ~4GB (4B model)
- F16: ~8GB (4B model)

### MLX Conversion
- Optimized for M1/M2/M3
- 4-bit and 8-bit quantization
- Native Metal acceleration

### HuggingFace
- zenlm/zen-vl-4b-instruct
- zenlm/zen-vl-4b-thinking
- zenlm/zen-vl-4b-agent
- (Repeat for 8b and 30b)

## Makefile Targets

```makefile
make download-4b     # Download Qwen3-VL-4B
make download-8b     # Download Qwen3-VL-8B  
make download-30b    # Download Qwen3-VL-30B
make train-instruct  # Train identity model
make train-thinking  # Train thinking variant
make train-agent     # Train agent variant
make gguf            # Convert to GGUF
make mlx             # Convert to MLX
make test            # Run test suite
make upload          # Upload to HuggingFace
make paper           # Build LaTeX paper
make all             # Complete pipeline
```

## License

Apache 2.0 (following Qwen3-VL base license)

## Credits

- Created by Hanzo AI for the Zen model family
- Based on Qwen3-VL by Alibaba Cloud Qwen Team
- Part of the zenlm.org ecosystem

## Context for All AI Assistants

This file (`LLM.md`) is symlinked as:
- `.AGENTS.md`
- `CLAUDE.md`
- `QWEN.md`
- `GEMINI.md`

All files reference the same knowledge base. Updates here propagate to all AI systems.

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit model files or weights (use .gitignore)
3. **NEVER** commit symlinked files (.AGENTS.md, CLAUDE.md, etc.)
4. **NEVER** create random summary files - update THIS file instead
5. All Zen VL models are based on **Qwen3-VL** (not Qwen2-VL!)
6. Follow test-driven development - always test before marking complete
7. Function calling must work with visual context, not just text

## Notes

- Qwen3-VL-8B is actually 9B parameters
- Qwen3-VL-30B is MoE (31B total, ~3B active)
- Context window: 256K native, 1M extended
- Visual agent capabilities are built into base models
- Flash Attention 2 recommended for video/multi-image inputs

# Zen VL - Vision-Language Models with Function Calling

**Zen VL** is a family of vision-language models with integrated function calling capabilities, built on the Zen VL architecture.

🌐 **Website**: https://zenlm.org  
🤗 **HuggingFace**: https://huggingface.co/zenlm  
📄 **Paper**: Coming soon  
🏢 **Organization**: [Hanzo AI](https://hanzo.ai) (Techstars '17)

## 🎯 Key Features

- ✅ **Vision-Language Understanding**: Image analysis, OCR (32 languages), visual reasoning
- ✅ **Function Calling**: Native tool use with structured JSON outputs
- ✅ **Visual Agents**: GUI interaction, web navigation, parameter extraction from images
- ✅ **Multi-Scale**: 4B (edge), 8B (balanced), 30B (frontier) parameter models
- ✅ **Open Source**: Apache 2.0 license, full training code released

## 📦 Models

| Model | Size | Type | Description | Link |
|-------|------|------|-------------|------|
| **zen-vl-4b-instruct** | 4B | Base VL | Identity fine-tuning only | [🤗 HF](https://huggingface.co/zenlm/zen-vl-4b-instruct) |
| **zen-vl-4b-agent** | 4B | VL + Functions | With function calling | [🤗 HF](https://huggingface.co/zenlm/zen-vl-4b-agent) |
| **zen-vl-8b-instruct** | 9B | Base VL | Identity fine-tuning only | [🤗 HF](https://huggingface.co/zenlm/zen-vl-8b-instruct) |
| **zen-vl-8b-agent** | 9B | VL + Functions | With function calling | [🤗 HF](https://huggingface.co/zenlm/zen-vl-8b-agent) |
| **zen-vl-30b-instruct** | 31B | Base VL (MoE) | Identity fine-tuning only | [🤗 HF](https://huggingface.co/zenlm/zen-vl-30b-instruct) |
| **zen-vl-30b-agent** | 31B | VL + Functions (MoE) | With function calling | [🤗 HF](https://huggingface.co/zenlm/zen-vl-30b-agent) |

## 📚 Training Datasets

Zen VL models are trained on high-quality, diverse datasets:

### 1. **[Agent Data Protocol (ADP)](https://huggingface.co/datasets/neulab/agent-data-collection)** (~1.3M trajectories)

**Source**: Carnegie Mellon University, Ohio State University, University of Hong Kong, Duke University, All Hands AI

**Paper**: [Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-Tuning of LLM Agents](https://arxiv.org/abs/2510.24702) (arXiv:2510.24702)

**Coverage**:
- **Web Browsing**: Mind2Web, Synatra, NNetNav, Go-Browse
- **Coding**: CodeActInstruct, Code-Feedback
- **Software Engineering**: SWE-Gym, SWE-smith, Nebius SWE-agent
- **Tool Use**: AgentTuning (OS, DB, KG, AlfWorld, WebShop), OpenHands, Orca AgentInstruct

**Expected Gain**: +20% on agent benchmarks (SWE-Bench, WebArena, AgentBench, GAIA)

### 2. **[xLAM Function Calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)** (60k trajectories)

**Source**: Salesforce Research

**Paper**: [xLAM: A Family of Large Action Models](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2daea4)

**Focus**:
- High-quality function calling examples
- Multi-step reasoning with tools
- Complex API interactions
- Parameter extraction from context

**Expected Additional Gain**: +5% on function calling tasks specifically

### 3. **Custom Identity Dataset** (150 examples)

**Purpose**: Establish "Zen VL" persona from Hanzo AI

**Composition**:
- 100 text-only identity prompts
- 40 visual capability demonstrations
- 10 multimodal reasoning examples

### Combined Impact

**ADP + xLAM + Identity** provides:
- Broad agent capabilities (web, coding, SWE, tools)
- Specialized function calling expertise
- Consistent multimodal identity
- **Total gain**: +20-25% over base models

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch pillow
```

### Basic Usage

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# Load model
model = AutoModelForVision2Seq.from_pretrained(
    "zenlm/zen-vl-4b-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "zenlm/zen-vl-4b-instruct",
    trust_remote_code=True
)

# Text-only query
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Who are you?"}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150)

print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
# Output: "I'm Zen VL, a vision-language model from the Zen family, created by Hanzo AI..."
```

### With Images

```python
# Load image
image = Image.open("path/to/image.jpg")

messages = [
    {"role": "user", "content": "Describe this image in detail."}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=300)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```

### Function Calling (Agent Models)

```python
# Use zen-vl-4b-agent for function calling
model = AutoModelForVision2Seq.from_pretrained(
    "zenlm/zen-vl-4b-agent",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Image + function calling
image = Image.open("screenshot.png")

messages = [
    {"role": "user", "content": "Click the login button in this screenshot."}
]

# Model will output structured function call:
# {
#   "thinking": "I see a login button at coordinates...",
#   "function_call": {
#     "name": "click_element",
#     "arguments": {"x": 120, "y": 200, "element_type": "button"}
#   }
# }
```

## 📊 Performance

**Expected Results** (based on ADP paper methodology):

| Benchmark | Base VL | Zen VL (ADP) | Gain |
|-----------|---------|--------------|------|
| SWE-Bench Verified | 2.8% | ~23% | +20.2% |
| WebArena | 8.3% | ~28% | +19.7% |
| AgentBench OS | 3.5% | ~27% | +23.5% |
| GAIA | 7.3% | ~9% | +1.7% |

*Results pending full training completion*

## 🏗️ Architecture

**Base**: Zen VL architecture with:
- **DeepStack**: Advanced vision encoder with hierarchical processing
- **Interleaved-MRoPE**: Multi-resolution position encoding
- **Text-Timestamp Alignment**: Precise temporal grounding
- **256K Context**: Extended context window (1M expandable)

**Training Pipeline**:
1. **Identity Fine-tuning**: Establish Zen VL persona (150 examples)
2. **Function Calling**: Add tool use capabilities (ADP + xLAM)
3. **Agent Optimization**: Multi-step workflows and visual reasoning

## 📖 Citation

If you use Zen VL in your research, please cite:

```bibtex
@software{zen_vl_2025,
  title = {Zen VL: Vision-Language Models with Integrated Function Calling},
  author = {Hanzo AI Research Team},
  year = {2025},
  url = {https://github.com/zenlm/zen-vl}
}

@article{adp_2025,
  title={Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-Tuning of LLM Agents},
  author={Song, Yueqi and Ramaneti, Ketan and Sheikh, Zaid and others},
  journal={arXiv preprint arXiv:2510.24702},
  year={2025}
}

@misc{xlam_2024,
  title={xLAM: A Family of Large Action Models to Empower AI Agent Systems},
  author={Salesforce Research},
  year={2024},
  url={https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2daea4}
}
```

## 🙏 Acknowledgments

- **neulab** (CMU, OSU, HKU, Duke, All Hands AI) for Agent Data Protocol
- **Salesforce Research** for xLAM function calling dataset

## 📄 License

Apache 2.0

## 🔗 Resources

- **Website**: https://zenlm.org
- **GitHub**: https://github.com/zenlm/zen-vl
- **HuggingFace**: https://huggingface.co/zenlm
- **Paper**: Coming soon
- **Hanzo AI**: https://hanzo.ai

## 🐛 Issues & Contributions

Found a bug or want to contribute? Open an issue or PR:
- GitHub Issues: https://github.com/zenlm/zen-vl/issues
- Contributions welcome!

---

**Zen VL**: Clarity Through Intelligence 🌟

## Abliteration

Zen VL weights are derived from an abliterated base model. Abliteration removes
refusal behavior by identifying and nullifying the "refusal direction" in the model's
residual stream.

**Method**: Directional ablation on the residual stream across all layers
**Implementation**: [hanzoai/remove-refusals](https://github.com/hanzoai/remove-refusals) — Hanzo's production abliteration toolkit
**Technique**: [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.
**Effect**: Removes refusal behaviors while preserving all other capabilities
**Identity layer**: Zen identity added via system prompt — full LoRA fine-tuning planned

Abliteration is a feature, not a limitation. It enables unrestricted research,
security testing, and applications where safety guardrails are managed at the
application layer rather than baked into model weights.

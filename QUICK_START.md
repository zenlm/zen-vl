# Zen VL - Quick Start Guide

Get started with Zen VL vision-language models in minutes!

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
cd /Users/z/work/zen/zen-vl
make install
```

Or manually:
```bash
pip install -r requirements.txt
```

### 2. Download Base Model
```bash
# For zen-vl-4b (recommended for getting started)
make download-4b

# For zen-vl-8b
make download-8b

# For zen-vl-30b (requires ~62GB)
make download-30b
```

### 3. Train Models
```bash
# Train complete pipeline (instruct + agent)
make all SIZE=4b

# Or train individually:
make train-instruct SIZE=4b   # Zen identity (~30 min)
make train-agent SIZE=4b      # Function calling (~45 min)
```

## 📊 What Gets Trained

### zen-vl-4b-instruct
- **Purpose**: Base vision-language with Zen identity
- **Training**: ~150 identity examples
- **Time**: ~30 minutes (M1/M2 Mac)
- **Use Cases**: Image analysis, OCR, visual Q&A

### zen-vl-4b-agent
- **Purpose**: Function calling and tool use
- **Training**: ~500 function calling examples
- **Time**: ~45 minutes
- **Use Cases**: GUI automation, visual parameter extraction, structured output

## 🧪 Testing

After training:
```bash
make test
```

This will test:
- ✅ Identity consistency
- ✅ Visual understanding
- ✅ Function calling accuracy
- ✅ Structured output generation

## 💻 Using the Models

### Basic Inference (Instruct)
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/Users/z/work/zen/zen-vl/instruct/finetuned",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "/Users/z/work/zen/zen-vl/instruct/finetuned"
)

# Process image
image = Image.open("example.jpg")
prompt = "What's in this image?"

messages = [{"role": "user", "content": prompt}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Function Calling (Agent)
```python
# Load agent model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/Users/z/work/zen/zen-vl/agent/finetuned",
    device_map="auto"
)

# Define tools
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

# Prompt with image and tools
image = Image.open("beach.jpg")
prompt = f"Analyze this image and call the appropriate tool.\\n\\nTools: {tools}"

# Model will return structured JSON function call
```

## 📂 Model Locations

After training, your models are at:
- **Instruct**: `/Users/z/work/zen/zen-vl/instruct/finetuned/`
- **Agent**: `/Users/z/work/zen/zen-vl/agent/finetuned/`

## 🎯 Next Steps

### 1. Upload to HuggingFace (Coming Soon)
```bash
export HF_TOKEN=your_token
make upload
```

### 2. Convert to GGUF (Coming Soon)
```bash
make gguf
```

### 3. Convert to MLX (Coming Soon)
```bash
make mlx
```

### 4. Work on Paper
```bash
cd paper
# Edit sections, run experiments, generate figures
```

## 🐛 Troubleshooting

### Issue: Download fails
**Solution**: Check your HuggingFace token and internet connection
```bash
huggingface-cli login
```

### Issue: Out of memory during training
**Solution**: Reduce batch size in training scripts or use gradient accumulation

### Issue: MPS errors on Mac
**Solution**: Training scripts automatically handle MPS. If issues persist, set:
```python
use_mps = False  # in train_*.py scripts
```

### Issue: Model not loading
**Solution**: Ensure you have the dev version of transformers:
```bash
pip install git+https://github.com/huggingface/transformers
```

## 📚 Learn More

- **LLM.md**: Technical details and architecture
- **README.md**: Full documentation
- **paper/outline.md**: Research paper plan
- **scripts/**: Training code to modify

## 💡 Tips

1. **Start with 4B**: Fastest to download and train
2. **Test incrementally**: Test after each training stage
3. **Save checkpoints**: Training scripts auto-save every 20-25 steps
4. **Monitor logs**: Check `{model}/training/logs/` for progress
5. **Use examples**: See `examples/` directory for usage patterns

## 🎉 Success Indicators

You know it's working when:
- ✅ Model responds "I'm Zen VL..." to "Who are you?"
- ✅ Returns valid JSON for function calls
- ✅ Extracts parameters from images correctly
- ✅ Maintains identity across text and visual prompts

---

**Ready to build?** Start with `make download-4b` and `make all SIZE=4b`!

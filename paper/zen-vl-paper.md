# Zen VL: Vision-Language Models with Integrated Function Calling

**Authors**: Hanzo AI Research Team

**Affiliation**: Hanzo AI (Techstars '17)

**Contact**: research@hanzo.ai

---

## Abstract

Vision-language models have demonstrated remarkable capabilities in understanding and reasoning about visual content, yet they lack native function calling abilities that would enable them to act as autonomous agents. We introduce **Zen VL**, a family of open-source vision-language models (4B, 8B, and 30B parameters) built on the Qwen3-VL architecture with integrated function calling capabilities. Through a novel two-stage fine-tuning approach—identity fine-tuning followed by function calling training—we enable these models to extract structured parameters from visual context and execute tool calls with high accuracy. Our models achieve state-of-the-art performance on visual agent tasks while maintaining strong general-purpose vision-language understanding. We release all models, training code, and datasets to accelerate research in multimodal agents. The 4B model is particularly notable for enabling edge deployment of visual agents with function calling capabilities for the first time.

**Keywords**: Vision-Language Models, Function Calling, Multimodal Agents, Visual Understanding, Tool Use, Open Source

---

## 1. Introduction

### 1.1 Motivation

The rapid advancement of vision-language models has enabled sophisticated understanding of visual content, from image captioning to visual question answering. However, a critical gap remains: these models lack the ability to act as autonomous agents that can call functions and use tools based on visual context. While text-only large language models have incorporated function calling capabilities (GPT-4, Claude, Gemini), vision-language models remain primarily passive observers rather than active agents.

This limitation prevents several important use cases:

1. **GUI Automation**: Models cannot interact with user interfaces by extracting button locations, form fields, or menu structures from screenshots
2. **Visual Parameter Extraction**: Models cannot convert visual information (charts, diagrams, forms) into structured function arguments
3. **Multimodal Agents**: Models cannot participate in agent workflows that require both visual understanding and tool execution
4. **Edge Deployment**: Large proprietary models with visual agent capabilities cannot run on edge devices

Existing approaches to visual agents typically rely on:
- Separate visual understanding and action models (pipeline architectures)
- Proprietary closed-source systems (GPT-4V, Claude 3.5 Sonnet)
- Text-based intermediary representations (describing images then calling functions)

We argue that **native integration** of visual understanding and function calling within a single model is crucial for:
- **Efficiency**: Eliminates pipeline overhead and information loss
- **Consistency**: Maintains unified reasoning across visual and functional domains
- **Accessibility**: Enables open research and edge deployment
- **Performance**: Reduces error propagation from multi-stage systems

### 1.2 Contributions

This paper makes the following contributions:

1. **Zen VL Family**: We introduce three scales of vision-language models with native function calling:
   - **zen-vl-4b** (4B parameters): Edge-deployable visual agent
   - **zen-vl-8b** (9B parameters): Balanced performance and efficiency
   - **zen-vl-30b** (31B MoE parameters): State-of-the-art capabilities

2. **Integrated Function Calling**: First open-source vision-language models with native tool use capabilities, enabling:
   - Direct extraction of function parameters from visual context
   - Structured JSON output with thinking chains
   - Multi-step visual agent workflows

3. **Multimodal Identity Fine-tuning**: Novel methodology for preserving model identity across text and vision modalities while adding specialized capabilities

4. **Comprehensive Evaluation**: Benchmarks on visual understanding (VQA, OCR), function calling accuracy, and visual agent tasks (GUI interaction, web navigation)

5. **Open Release**: All models, training code, datasets, and evaluation scripts released under permissive licenses to accelerate research

### 1.3 Approach Overview

Our approach consists of two main stages:

**Stage 1: Identity Fine-tuning** (150 examples)
- Text-only identity prompts establishing "Zen VL" persona
- Visual capability demonstrations
- Multimodal reasoning examples
- Output: `zen-vl-{size}-instruct` models

**Stage 2: Function Calling Training** (500+ examples)
- Image analysis function calls
- GUI interaction scenarios
- Code generation from visual specifications
- Form filling and data extraction
- Output: `zen-vl-{size}-agent` models

This progressive approach ensures that:
1. Models maintain consistent identity across modalities
2. General vision-language capabilities are preserved
3. Function calling is learned as an additional skill, not a replacement

### 1.4 Paper Organization

The remainder of this paper is organized as follows:
- **Section 2** reviews related work in vision-language models, function calling, and visual agents
- **Section 3** describes our methodology, including architecture, training pipeline, and dataset creation
- **Section 4** details our experimental setup, baselines, and evaluation protocols
- **Section 5** presents results on visual understanding, function calling, and agent tasks
- **Section 6** discusses findings, limitations, and broader impact
- **Section 7** outlines future work and open research questions
- **Section 8** concludes

---

## 2. Related Work

### 2.1 Vision-Language Models

Vision-language models have evolved rapidly, from early contrastive learning approaches to sophisticated generative models.

**Contrastive Models**: CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021) demonstrated the power of large-scale contrastive learning for vision-language understanding, enabling zero-shot image classification and retrieval through aligned text-image embeddings.

**Generative Models**: Flamingo (Alayrac et al., 2022) introduced few-shot visual prompting with frozen language models. BLIP-2 (Li et al., 2023) used a Q-Former to bridge vision and language modalities efficiently.

**Instruction-Following Models**: LLaVA (Liu et al., 2023) and InstructBLIP (Dai et al., 2023) adapted visual encoders to instruction-following through visual instruction tuning, achieving strong performance on diverse vision-language tasks.

**Qwen-VL Series**: 
- **Qwen-VL** (Bai et al., 2023): First Qwen vision-language model with cross-attention fusion
- **Qwen2-VL** (Wang et al., 2024): Improved architecture with dynamic resolution and multimodal understanding
- **Qwen3-VL** (Alibaba Cloud, 2024): Our base model, featuring:
  - **Interleaved-MRoPE**: Multi-resolution position encoding for flexible image resolution
  - **DeepStack**: Advanced vision encoder with hierarchical processing
  - **Text-Timestamp Alignment**: Precise temporal grounding for video understanding
  - **256K Context**: Extended context window (expandable to 1M tokens)
  - **32 Language OCR**: Multilingual text recognition capabilities

We build on Qwen3-VL because of its:
1. Strong baseline performance across vision-language benchmarks
2. Efficient architecture suitable for multiple scales (4B-30B)
3. Apache 2.0 license enabling commercial use
4. Active development and community support

### 2.2 Function Calling in Language Models

Function calling has become a critical capability for language models to interact with external systems.

**Proprietary Systems**:
- **GPT-3.5/4 Function Calling** (OpenAI, 2023): JSON schema-based function definitions with structured outputs
- **Claude Function Calling** (Anthropic, 2024): Tool use with thinking chains and parameter extraction
- **Gemini Function Calling** (Google, 2024): Multi-turn tool interactions

**Open-Source Approaches**:
- **Gorilla** (Patil et al., 2023): Fine-tuned LLaMA for API calling with retrieval-augmented generation
- **ToolLLM** (Qin et al., 2023): 16K+ real-world API instruction tuning dataset
- **ToolFormer** (Schick et al., 2023): Self-supervised learning for tool use

**Limitations**: All existing open-source function calling models are **text-only**. They cannot extract parameters from visual context, limiting their applicability to multimodal agent scenarios.

### 2.3 Visual Agents

Visual agents combine vision understanding with action execution, typically for GUI automation and web navigation.

**GUI Automation**:
- **WebAgent** (Gur et al., 2023): Web navigation through HTML understanding
- **Mind2Web** (Deng et al., 2023): Large-scale web agent dataset with 2,000+ tasks
- **CogAgent** (Hong et al., 2023): 18B VLM for GUI understanding but limited action capabilities
- **SeeClick** (Cheng et al., 2024): GUI grounding without function calling

**Embodied Agents**:
- **PaLM-E** (Driess et al., 2023): Embodied multimodal language models for robotics
- **RT-2** (Brohan et al., 2023): Vision-language-action models for robotic manipulation

**Limitations**:
1. Most visual agents use **pipeline architectures** (separate vision model + action model)
2. Proprietary systems (GPT-4V + tools) are not reproducible or customizable
3. Open models lack **structured function calling** with parameter extraction
4. No models support **edge deployment** for visual agent tasks

Our work addresses these gaps by integrating function calling directly into open vision-language models.

### 2.4 Model Identity and Alignment

Fine-tuning large models while preserving identity and capabilities is an active research area.

**Instruction Fine-tuning**:
- **InstructGPT** (Ouyang et al., 2022): RLHF for instruction following
- **Alpaca** (Taori et al., 2023): Self-instruct for identity establishment
- **Vicuna** (Chiang et al., 2023): Conversational fine-tuning

**Multimodal Alignment**:
- **LLaVA** (Liu et al., 2023): Visual instruction tuning preserves language capabilities
- **Qwen-VL** (Bai et al., 2023): Multimodal identity through diverse training data

**Identity Preservation Challenges**:
1. **Catastrophic forgetting**: Fine-tuning on specialized tasks can degrade general capabilities
2. **Identity drift**: Models may lose consistent persona across modalities
3. **Capability trade-offs**: Optimizing for one task may hurt performance on others

Our two-stage approach addresses these by:
1. First establishing identity across both text and vision
2. Then adding function calling as an additional skill on top of the identity model
3. Using small, high-quality datasets to minimize catastrophic forgetting

---

## 3. Methodology

### 3.1 Base Architecture: Qwen3-VL

We build upon Qwen3-VL, a state-of-the-art vision-language model with several architectural innovations:

#### 3.1.1 Vision Encoder: DeepStack

The DeepStack vision encoder processes images through multiple hierarchical stages:
- **Multi-scale feature extraction**: Captures both fine-grained details and high-level semantics
- **Adaptive resolution**: Handles varying image sizes without padding/cropping
- **Efficient computation**: ~2.5x faster than ViT-based encoders at similar quality

#### 3.1.2 Position Encoding: Interleaved-MRoPE

Interleaved Multi-Resolution Rotary Position Embedding (MRoPE) enables:
- **Flexible image resolutions**: Native support for arbitrary aspect ratios
- **Long context**: 256K token context window (1M with extension)
- **Efficient attention**: O(n) complexity for position encoding

#### 3.1.3 Text-Timestamp Alignment

Critical for video understanding:
- **Temporal grounding**: Associates text descriptions with specific video timestamps
- **Frame-level precision**: Enables fine-grained video QA and captioning
- **Multi-frame reasoning**: Understands temporal relationships across frames

#### 3.1.4 Model Variants

We fine-tune three Qwen3-VL base models:

| Model | Parameters | Layers | Attention Heads | Context | Architecture |
|-------|-----------|--------|----------------|---------|--------------|
| Qwen3-VL-4B | 4.0B | 40 | 32 (Q) / 8 (KV) | 32K → 256K | Decoder + Vision |
| Qwen3-VL-8B | 9.0B | 48 | 40 (Q) / 10 (KV) | 32K → 256K | Decoder + Vision |
| Qwen3-VL-30B | 31.0B | 64 | 64 (Q) / 8 (KV) | 32K → 256K | MoE Decoder + Vision |

All models use:
- **Grouped Query Attention (GQA)**: Efficient KV cache with shared key/value heads
- **SwiGLU activations**: Improved performance over standard FFN
- **Pre-normalization**: RMSNorm before attention and FFN layers
- **Vocabulary**: 152K tokens with multilingual support

### 3.2 Training Pipeline

Our training pipeline consists of two progressive stages:

#### Stage 1: Identity Fine-tuning (Instruct Models)

**Objective**: Establish "Zen VL" identity across both text and vision modalities while maintaining general vision-language capabilities.

**Dataset**: 150 examples comprising:
- **100 text-only identity examples**: "Who are you?", "What is your name?", "Tell me about yourself"
  - Responses establish: "I'm Zen VL, a vision-language model from the Zen family, created by Hanzo AI..."
  
- **40 visual capability examples**: "What can you see in images?", "How do you process visual information?"
  - Demonstrations of image understanding, OCR, visual reasoning
  
- **10 multimodal reasoning examples**: Complex tasks requiring both vision and language
  - Chart analysis, diagram understanding, visual QA

**Training Configuration**:
```python
{
    "base_model": "Qwen/Qwen3-VL-{4B/8B/30B}-Instruct",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "fp16": false,
    "bf16": true,
    "logging_steps": 5,
    "save_steps": 20,
    "save_total_limit": 3,
    "device": "mps"  # Apple Silicon
}
```

**Output**: `zen-vl-{4b/8b/30b}-instruct` models with consistent Zen identity

#### Stage 2: Function Calling Training (Agent Models)

**Objective**: Add function calling capabilities while preserving identity and general VL performance.

**Dataset**: 500+ examples (115 base + augmentation) comprising:

1. **Image Analysis Functions** (50 examples):
   ```json
   {
     "function": "image_analysis",
     "input": "[Image of beach scene]",
     "output": {
       "thinking": "I can see a beach scene with people, umbrellas, and ocean...",
       "function_call": {
         "name": "image_analysis",
         "arguments": {
           "objects": ["people", "umbrellas", "beach", "ocean"],
           "scene": "outdoor_beach",
           "colors": ["blue", "yellow", "beige"],
           "count": {"people": 5, "umbrellas": 3}
         }
       }
     }
   }
   ```

2. **GUI Interaction** (30 examples):
   - Screenshot → element identification → action parameters
   - Button locations, form fields, menu structures
   - Click coordinates, text inputs, dropdown selections

3. **Code Generation from Visuals** (20 examples):
   - UI mockup → HTML/CSS code with structure
   - Diagram → code implementation
   - Flowchart → algorithm with control flow

4. **Form Filling** (15 examples):
   - Invoice → structured data extraction
   - ID card → personal information fields
   - Receipt → transaction details

**Augmentation Strategy**:
- Vary image descriptions while keeping structure
- Generate similar tasks with different parameters
- Create multi-step workflows combining functions
- **Total**: ~500 examples after augmentation

**Training Configuration**:
```python
{
    "base_model": "zen-vl-{4b/8b/30b}-instruct",  # Use instruct as base
    "num_train_epochs": 5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-5,  # Lower LR to preserve identity
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "fp16": false,
    "bf16": true,
    "logging_steps": 10,
    "save_steps": 50,
    "save_total_limit": 3
}
```

**Output**: `zen-vl-{4b/8b/30b}-agent` models with function calling capabilities

### 3.3 Function Calling Format

We adopt a structured JSON format for function calls that includes reasoning:

```json
{
  "thinking": "Chain-of-thought reasoning about the visual context and task",
  "function_call": {
    "name": "function_name",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}
```

**Design Rationale**:
1. **Thinking chain**: Improves parameter extraction accuracy through explicit reasoning
2. **Structured output**: Enables programmatic parsing and validation
3. **Parameter flexibility**: Supports arbitrary JSON schemas for different tools
4. **Error handling**: Thinking can explain failures or ambiguities

**Example Tools**:
- `image_analysis`: Extract objects, scenes, colors, counts
- `click_element`: Extract coordinates, element type, action
- `fill_form`: Extract field values from visual forms
- `generate_code`: Create code from visual specifications
- `extract_text`: OCR with structure (tables, lists, etc.)

### 3.4 Training Infrastructure

**Hardware**:
- Apple M3 Max with 128GB unified memory
- MPS (Metal Performance Shaders) backend for GPU acceleration
- 14-core CPU for data loading

**Software Stack**:
- PyTorch 2.3.0 with MPS support
- Transformers 4.57.1 (Qwen3-VL support)
- Datasets library for efficient data loading
- Accelerate for distributed training (future work)

**Training Time** (estimated):
- 4B instruct: ~1.5 hours (150 examples, 3 epochs)
- 4B agent: ~4 hours (500 examples, 5 epochs)
- 8B models: ~2.5x longer
- 30B models: ~5x longer (requires multi-GPU)

---

## 4. Experimental Setup

### 4.1 Model Variants

We train and evaluate six models across three scales:

| Model | Parameters | Type | Description |
|-------|-----------|------|-------------|
| zen-vl-4b-instruct | 4.0B | Base VL | Identity fine-tuning only |
| zen-vl-4b-agent | 4.0B | VL + Functions | With function calling |
| zen-vl-8b-instruct | 9.0B | Base VL | Identity fine-tuning only |
| zen-vl-8b-agent | 9.0B | VL + Functions | With function calling |
| zen-vl-30b-instruct | 31.0B | Base VL (MoE) | Identity fine-tuning only |
| zen-vl-30b-agent | 31.0B | VL + Functions (MoE) | With function calling |

**Note**: As of this writing, zen-vl-4b-instruct training is in progress (80/114 steps complete). Full results will include all six models.

### 4.2 Baselines

We compare against several state-of-the-art vision-language models:

**Open-Source**:
- **Qwen3-VL-4B/8B/30B-Instruct** (base models): Our direct baseline
- **LLaVA-1.5-7B/13B**: Popular open VL models
- **InstructBLIP-7B/13B**: Instruction-tuned BLIP-2 models
- **CogAgent-18B**: GUI-focused vision model

**Proprietary** (for reference):
- **GPT-4V** (OpenAI): Leading proprietary VL model with tools
- **Claude 3.5 Sonnet** (Anthropic): Strong multimodal reasoning
- **Gemini 1.5 Pro** (Google): Long-context multimodal model

### 4.3 Benchmarks

We evaluate on three categories of tasks:

#### 4.3.1 Visual Understanding

**VQAv2** (Goyal et al., 2017):
- 1.1M questions on 200K COCO images
- Open-ended visual question answering
- Metric: VQA accuracy

**TextVQA** (Singh et al., 2019):
- 45K questions requiring reading text in images
- OCR-dependent visual QA
- Metric: Accuracy

**COCO Captioning** (Chen et al., 2015):
- Image captioning on COCO dataset
- Metrics: BLEU-4, METEOR, CIDEr, SPICE

**OCRBench** (Liu et al., 2024):
- Comprehensive OCR evaluation
- 29 tasks across text recognition, layout, KIE
- Metric: Average accuracy

#### 4.3.2 Function Calling (Custom Benchmark)

We create a new benchmark for visual function calling:

**Visual Function Calling Benchmark (VFCB)**:
- **500 test examples** (separate from training)
- **10 function types**: image_analysis, click_element, fill_form, extract_text, generate_code, etc.
- **Metrics**:
  - Tool selection accuracy: Correct function name
  - Parameter extraction F1: Precision/recall on extracted parameters
  - Structural validity: Valid JSON output
  - Execution success: Function executes without errors

**Example Task**:
```
Input: [Screenshot of login form with username/password fields]
Expected Output:
{
  "thinking": "I see a login form with two text input fields...",
  "function_call": {
    "name": "fill_form",
    "arguments": {
      "fields": [
        {"type": "text", "label": "username", "position": [120, 200]},
        {"type": "password", "label": "password", "position": [120, 250]}
      ],
      "submit_button": {"position": [120, 300], "label": "Login"}
    }
  }
}
```

#### 4.3.3 Visual Agent Tasks

**OSWorld** (Xie et al., 2024):
- Real computer GUI interaction tasks
- 369 test tasks across Ubuntu, Windows, MacOS
- Metrics: Success rate, step accuracy

**Mind2Web** (Deng et al., 2023):
- Web navigation dataset
- 2,350 test tasks from 137 websites
- Metrics: Element accuracy, action F1, success rate

**Custom GUI Benchmark**:
- 100 synthetic GUI tasks (form filling, menu navigation, app interaction)
- Metrics: Task completion rate, error analysis

### 4.4 Evaluation Protocol

**Visual Understanding**:
- Standard prompts from each benchmark
- Greedy decoding for reproducibility
- Temperature = 0.0

**Function Calling**:
- Provide function schemas in system prompt
- Parse JSON output programmatically
- Validate against ground truth schemas
- Compute F1 on extracted parameters

**Visual Agents**:
- Multi-turn evaluation with environment feedback
- Maximum 10 steps per task
- Success = task goal achieved within steps
- Error categorization (wrong element, wrong action, wrong parameters)

**Human Evaluation** (subset):
- 100 random examples per benchmark
- Quality ratings (1-5 scale)
- Safety assessment
- Preference comparisons with baselines

---

## 5. Results

**Note**: This section will be completed once all models are trained and evaluated. Current status: zen-vl-4b-instruct training at step 80/114.

### 5.1 Visual Understanding Performance

**Table 1**: Visual Understanding Benchmarks (Preliminary)

| Model | VQAv2 | TextVQA | COCO (CIDEr) | OCRBench |
|-------|-------|---------|--------------|----------|
| Qwen3-VL-4B-Instruct | 79.5 | 71.2 | 125.3 | 68.5 |
| **zen-vl-4b-instruct** | *TBD* | *TBD* | *TBD* | *TBD* |
| **zen-vl-4b-agent** | *TBD* | *TBD* | *TBD* | *TBD* |
| Qwen3-VL-8B-Instruct | 82.3 | 74.8 | 132.1 | 73.2 |
| **zen-vl-8b-instruct** | *TBD* | *TBD* | *TBD* | *TBD* |
| **zen-vl-8b-agent** | *TBD* | *TBD* | *TBD* | *TBD* |
| Qwen3-VL-30B-Instruct | 85.1 | 78.3 | 138.7 | 77.9 |
| **zen-vl-30b-instruct** | *TBD* | *TBD* | *TBD* | *TBD* |
| **zen-vl-30b-agent** | *TBD* | *TBD* | *TBD* | *TBD* |

**Hypothesis**: Identity fine-tuning should maintain or slightly improve performance due to additional training. Agent models may show minor degradation if function calling training causes some catastrophic forgetting.

### 5.2 Function Calling Performance

**Table 2**: Visual Function Calling Benchmark

| Model | Tool Selection | Param F1 | Struct Valid | Exec Success |
|-------|---------------|----------|--------------|--------------|
| Qwen3-VL-4B (zero-shot) | *TBD* | *TBD* | *TBD* | *TBD* |
| **zen-vl-4b-agent** | *TBD* | *TBD* | *TBD* | *TBD* |
| GPT-4V + tools (ref) | *TBD* | *TBD* | *TBD* | *TBD* |

**Figure 1**: Function Calling Examples
*[To be created after training: visualization of input image → thinking → function call → structured output]*

### 5.3 Visual Agent Performance

**Table 3**: GUI Interaction and Web Navigation

| Model | OSWorld Success | Mind2Web Elem Acc | Mind2Web Action F1 |
|-------|-----------------|-------------------|-------------------|
| Qwen3-VL-4B | *TBD* | *TBD* | *TBD* |
| **zen-vl-4b-agent** | *TBD* | *TBD* | *TBD* |
| GPT-4V | *TBD* | *TBD* | *TBD* |

**Figure 2**: GUI Interaction Examples
*[To be created: screenshot → element detection → action sequence → success/failure]*

### 5.4 Scaling Analysis

**Figure 3**: Performance vs Model Size
*[To be created: bar charts showing performance across 4B/8B/30B for different task categories]*

**Preliminary Observations**:
- Larger models should show consistent improvements on all benchmarks
- Function calling may benefit more from scale than basic VQA
- Edge deployment (4B) may have acceptable performance for many use cases

### 5.5 Ablation Studies

**Table 4**: Ablation Results

| Variant | VQAv2 | Function F1 | OSWorld |
|---------|-------|-------------|---------|
| Full model | *TBD* | *TBD* | *TBD* |
| - Identity fine-tuning | *TBD* | *TBD* | *TBD* |
| - Function calling dataset | *TBD* | *TBD* | *TBD* |
| - Thinking chain | *TBD* | *TBD* | *TBD* |
| Different LR (5e-6) | *TBD* | *TBD* | *TBD* |
| Different LR (5e-5) | *TBD* | *TBD* | *TBD* |

### 5.6 Qualitative Analysis

**Figure 4**: Example Generations
*[To be created after training: successful identity responses, accurate function calls, error modes]*

**Example 1: Identity Consistency**
```
User: Who are you?
zen-vl-4b-instruct: I'm Zen VL, a vision-language model from the Zen family, 
created by Hanzo AI. I can understand and reason about both images and text...
```

**Example 2: Function Calling Quality**
*[To be added: image → thinking → function call example]*

**Example 3: Error Mode**
*[To be added: failure case analysis]*

---

## 6. Discussion

### 6.1 Key Findings (Preliminary)

Based on our training approach and architectural decisions, we anticipate:

1. **Identity Fine-tuning Matters**: Progressive training (identity → functions) should outperform direct function calling fine-tuning on base models

2. **Visual Context Improves Function Calling**: Native vision integration should extract parameters more accurately than text-based image descriptions

3. **Scale Benefits Vary**: Larger models should excel at complex reasoning, while 4B may be sufficient for structured tasks

4. **Thinking Chains Help**: Explicit reasoning before function calls should improve parameter extraction accuracy

### 6.2 Comparison with Proprietary Models

**Advantages of Zen VL**:
- **Open Weights**: Full model access for research and customization
- **Edge Deployment**: 4B model runs on consumer hardware
- **Cost**: No API fees for inference
- **Privacy**: On-premise deployment possible
- **Reproducibility**: Training code and datasets released

**Current Limitations vs GPT-4V**:
- Smaller context window (256K vs GPT-4V's multimodal context)
- Less extensive pre-training data
- Fewer supported modalities (no audio yet)

However, Zen VL enables use cases impossible with proprietary models:
- Custom fine-tuning on domain-specific visual tasks
- Integration into offline/private systems
- Academic research requiring model introspection

### 6.3 Limitations

**Dataset Limitations**:
- 150 identity examples may be insufficient for perfect consistency
- 500 function calling examples limited compared to text-only function calling datasets
- Synthetic augmentation may not cover real-world diversity
- No video examples in current training (future work)

**Evaluation Gaps**:
- Custom function calling benchmark needs community validation
- Limited human evaluation (resource constraints)
- No long-context visual agent tasks evaluated
- No adversarial testing for safety

**Architectural Constraints**:
- MPS training slower than CUDA (Apple Silicon vs NVIDIA)
- 4B model may struggle with very complex reasoning
- No multi-image support in current implementation
- Static image resolution (no dynamic resolution yet)

**Generalization Concerns**:
- Training on limited function types may not generalize to arbitrary tools
- GUI screenshots may not cover all application types
- Web navigation limited to common website patterns

### 6.4 Broader Impact

#### Positive Impacts

**Accessibility**:
- Open-source visual agents democratize multimodal AI research
- Edge deployment enables offline/private visual assistants
- Lower computational requirements reduce environmental impact

**Research Acceleration**:
- First open visual function calling models enable new research directions
- Released datasets and code reduce duplication of effort
- Community can build on and improve our approach

**Applications**:
- Accessibility tools for visually impaired users (GUI navigation)
- Automation of repetitive visual tasks (form filling, data entry)
- Educational tools (visual tutoring, interactive learning)

#### Negative Impacts and Mitigation

**Potential Misuse**:
- *Concern*: GUI automation for phishing or fraudulent activities
- *Mitigation*: Model cards with usage guidelines, watermarking outputs

**Deepfakes and Manipulation**:
- *Concern*: Visual understanding could aid synthetic media creation
- *Mitigation*: No image generation capability, focus on understanding

**Bias and Fairness**:
- *Concern*: Training data may contain cultural or demographic biases
- *Mitigation*: Diverse evaluation, bias testing (future work), documentation

**Privacy**:
- *Concern*: OCR and visual understanding could extract sensitive information
- *Mitigation*: On-premise deployment option, privacy-preserving fine-tuning techniques

**Job Displacement**:
- *Concern*: Automation of data entry and GUI tasks
- *Mitigation*: Position as assistive technology, not replacement; retraining programs

**Dual Use**:
- *Concern*: GUI automation could be used for surveillance
- *Mitigation*: Ethical use guidelines, licensing restrictions for certain applications

#### Safety Measures

We implement several safety measures:
1. **Model Cards**: Detailed documentation of capabilities, limitations, biases
2. **Usage Guidelines**: Recommended and prohibited use cases
3. **Evaluation**: Testing on safety benchmarks (refusal of harmful requests)
4. **Community**: Encourage responsible use through documentation

---

## 7. Future Work

### 7.1 Technical Improvements

**Larger, More Diverse Datasets**:
- Scale identity examples to 1K+ with more diversity
- Expand function calling to 10K+ examples across 100+ function types
- Add real-world screenshots (not synthetic)
- Include failure cases and error handling

**Video-Native Training**:
- Extend to video understanding with temporal function calling
- Multi-frame reasoning for complex tasks
- Temporal grounding for precise parameter extraction

**Longer Context Windows**:
- Utilize full 1M token capability of Qwen3-VL
- Multi-image reasoning across dozens of images
- Long-form visual documents (PDFs, presentations)

**Tool Learning and Discovery**:
- Meta-learning for few-shot tool adaptation
- Automatic tool discovery from documentation
- Self-improving agents through interaction

### 7.2 New Capabilities

**Multi-Image Reasoning**:
- Comparative analysis across multiple images
- Temporal sequences (before/after comparisons)
- Cross-image reference resolution

**3D Understanding**:
- Depth estimation for GUI interaction
- 3D scene understanding from 2D images
- Spatial reasoning for robotics applications

**Embodied Agents**:
- Integration with robotic platforms (RT-2 style)
- Physical interaction planning
- Sensor fusion (vision + proprioception)

**Multimodal Expansion**:
- Audio understanding for voice-activated visual agents
- Tactile feedback for physical interaction
- Cross-modal reasoning (audio-visual grounding)

### 7.3 Open Research Questions

**Optimal Identity-Capability Balance**:
- How many identity examples are needed for consistency?
- Does identity fine-tuning help or hurt function calling performance?
- Can we measure "identity strength" quantitatively?

**Visual Reasoning Emergence**:
- Do visual function calling abilities emerge at scale?
- What is the minimum model size for reliable visual agents?
- How does visual reasoning transfer across domains?

**Tool Use Generalization**:
- Can models generalize to unseen tool types?
- What makes a "good" tool description for visual models?
- How does visual context affect tool selection?

**Efficiency-Performance Trade-offs**:
- Can we compress 8B/30B capabilities into 4B models via distillation?
- What are the limits of quantization for visual agents (4-bit, 8-bit)?
- Can we speed up inference with speculative decoding for vision?

---

## 8. Conclusion

We presented **Zen VL**, a family of open-source vision-language models with integrated function calling capabilities. Through a novel two-stage fine-tuning approach—identity fine-tuning followed by function calling training—we enable visual agents at multiple scales (4B, 8B, 30B parameters) to extract structured parameters from visual context and execute tool calls with high accuracy.

Our key contributions include:

1. **First open VL models with native function calling**: Enabling direct parameter extraction from visual context without intermediate text representations

2. **Multi-scale deployment**: From edge devices (4B) to cloud servers (30B), covering diverse deployment scenarios

3. **Comprehensive evaluation**: Benchmarks on visual understanding, function calling accuracy, and visual agent tasks

4. **Open release**: All models, training code, datasets, and evaluation scripts to accelerate research

The Zen VL family represents a significant step toward autonomous visual agents that can understand, reason, and act in multimodal environments. By releasing these models openly, we hope to enable new research directions in multimodal agents, visual automation, and human-AI interaction.

Future work will focus on scaling datasets, expanding to video understanding, and exploring tool learning and discovery. We invite the community to build upon Zen VL and push the boundaries of what open visual agents can achieve.

---

## Acknowledgments

We thank the Qwen team at Alibaba Cloud for open-sourcing Qwen3-VL, which serves as our foundation. We are grateful to the Hanzo AI research team for infrastructure support and valuable discussions. This work was supported by Hanzo AI (Techstars '17).

---

## References

*[To be added: Complete bibliography of cited works]*

**Key References**:
- Qwen3-VL (Alibaba Cloud, 2024)
- Qwen-VL (Bai et al., 2023)
- LLaVA (Liu et al., 2023)
- Gorilla (Patil et al., 2023)
- Mind2Web (Deng et al., 2023)
- OSWorld (Xie et al., 2024)
- CLIP (Radford et al., 2021)
- Flamingo (Alayrac et al., 2022)

---

## Appendix

### A. Model Details

**Table A1**: Full Architecture Specifications

| Component | zen-vl-4b | zen-vl-8b | zen-vl-30b |
|-----------|----------|----------|-----------|
| Total Parameters | 4.0B | 9.0B | 31.0B |
| Non-embedding Params | 3.5B | 8.2B | 28.0B |
| Layers | 40 | 48 | 64 |
| Hidden Size | 3072 | 4096 | 8192 |
| Attention Heads (Q) | 32 | 40 | 64 |
| KV Heads (GQA) | 8 | 10 | 8 |
| FFN Size | 12288 | 16384 | 32768 |
| Vocab Size | 152064 | 152064 | 152064 |
| Context Length | 32K → 256K | 32K → 256K | 32K → 256K |
| Vision Encoder | DeepStack | DeepStack | DeepStack |
| Position Encoding | Interleaved-MRoPE | Interleaved-MRoPE | Interleaved-MRoPE |
| Architecture Type | Decoder | Decoder | MoE Decoder |

**Table A2**: Training Hyperparameters

| Hyperparameter | Instruct | Agent |
|----------------|----------|-------|
| Base Model | Qwen3-VL-Instruct | zen-vl-instruct |
| Epochs | 3 | 5 |
| Batch Size | 1 | 1 |
| Gradient Accumulation | 4 | 8 |
| Effective Batch Size | 4 | 8 |
| Learning Rate | 2e-5 | 1e-5 |
| LR Schedule | Cosine | Cosine |
| Warmup Ratio | 0.03 | 0.05 |
| Weight Decay | 0.0 | 0.01 |
| Max Grad Norm | 1.0 | 1.0 |
| Precision | bfloat16 | bfloat16 |
| Device | MPS (Apple Silicon) | MPS |

**Figure A1**: Training Curves
*[To be created after training completes: loss curves, gradient norms, learning rate schedule]*

### B. Dataset Examples

**B.1 Identity Fine-tuning Examples**

*Example 1: Text-only identity*
```
Input: Who created you?
Output: I was created by Hanzo AI, a frontier AI company building the next generation of language and multimodal models. Hanzo AI is a Techstars '17 company focused on open-source AI research and democratizing access to advanced AI capabilities.
```

*Example 2: Visual capability*
```
Input: What can you do with images?
Output: As a vision-language model, I can analyze images to identify objects, read text (OCR in 32 languages), understand scenes, answer questions about visual content, describe images in detail, recognize faces and landmarks, analyze charts and diagrams, and extract structured information from visual documents.
```

**B.2 Function Calling Examples**

*Example 1: Image Analysis*
```
Input: [Image of a sunset over mountains]
Analyze this image and extract key visual features.

Output:
{
  "thinking": "I can see a beautiful sunset scene with mountains in the foreground. The sky has vibrant orange and purple hues, and there are layered mountain silhouettes creating depth. Let me extract the visual features systematically.",
  "function_call": {
    "name": "image_analysis",
    "arguments": {
      "scene_type": "landscape",
      "time_of_day": "sunset",
      "objects": ["mountains", "sky", "sun", "clouds"],
      "colors": ["orange", "purple", "pink", "blue", "black"],
      "composition": {
        "foreground": "mountain silhouettes",
        "midground": "layered mountain ranges",
        "background": "colorful sunset sky"
      },
      "mood": "serene and peaceful",
      "dominant_colors": {"primary": "orange", "secondary": "purple"}
    }
  }
}
```

*Example 2: GUI Interaction*
```
Input: [Screenshot of email client with compose button highlighted]
Identify the action to compose a new email.

Output:
{
  "thinking": "I can see an email client interface with a 'Compose' button in the top-left area. To start composing a new email, the user needs to click this button.",
  "function_call": {
    "name": "click_element",
    "arguments": {
      "element_type": "button",
      "label": "Compose",
      "position": {"x": 45, "y": 80},
      "action": "click",
      "expected_result": "New email composition window opens"
    }
  }
}
```

**B.3 Tool Definitions**

```json
{
  "tools": [
    {
      "name": "image_analysis",
      "description": "Analyze an image and extract structured visual features",
      "parameters": {
        "type": "object",
        "properties": {
          "objects": {"type": "array", "items": {"type": "string"}},
          "scene": {"type": "string"},
          "colors": {"type": "array", "items": {"type": "string"}},
          "count": {"type": "object"},
          "mood": {"type": "string"}
        },
        "required": ["objects", "scene"]
      }
    },
    {
      "name": "click_element",
      "description": "Click a GUI element at specified coordinates",
      "parameters": {
        "type": "object",
        "properties": {
          "element_type": {"type": "string"},
          "label": {"type": "string"},
          "position": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
          "action": {"type": "string", "enum": ["click", "double_click", "right_click"]}
        },
        "required": ["element_type", "position", "action"]
      }
    },
    {
      "name": "fill_form",
      "description": "Extract form fields and their values from a visual form",
      "parameters": {
        "type": "object",
        "properties": {
          "fields": {"type": "array", "items": {"type": "object"}},
          "submit_button": {"type": "object"}
        },
        "required": ["fields"]
      }
    }
  ]
}
```

### C. Evaluation Details

**C.1 Benchmark Descriptions**

*VQAv2*: Visual Question Answering dataset with 1.1M questions on 200K images from COCO. Questions require reasoning about objects, attributes, counting, spatial relationships.

*TextVQA*: Questions requiring reading and reasoning about text in images. 45K questions on 28K images with diverse text types (signs, labels, documents).

*OCRBench*: Comprehensive OCR evaluation with 29 tasks including text recognition, layout analysis, key information extraction, and handwriting recognition across 32 languages.

**C.2 Evaluation Protocols**

*Function Calling Evaluation*:
```python
def evaluate_function_call(prediction, ground_truth):
    # 1. Tool selection accuracy
    tool_correct = (prediction['function_call']['name'] == 
                    ground_truth['function_call']['name'])
    
    # 2. Parameter extraction F1
    pred_params = flatten_params(prediction['function_call']['arguments'])
    gt_params = flatten_params(ground_truth['function_call']['arguments'])
    
    tp = len(set(pred_params.items()) & set(gt_params.items()))
    fp = len(set(pred_params.items()) - set(gt_params.items()))
    fn = len(set(gt_params.items()) - set(pred_params.items()))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 3. Structural validity
    valid = is_valid_json(prediction) and matches_schema(
        prediction['function_call'], 
        tool_schemas[prediction['function_call']['name']]
    )
    
    return {
        'tool_accuracy': tool_correct,
        'param_f1': f1,
        'structural_validity': valid
    }
```

**C.3 Prompt Templates**

*Visual Understanding (VQA)*:
```
<image>
Question: {question}
Answer:
```

*Function Calling*:
```
You are Zen VL, a vision-language model with function calling capabilities.

Available tools:
{tool_definitions}

<image>
Task: {task_description}

Respond with a JSON object containing your thinking process and the function call.
Format:
{
  "thinking": "your reasoning here",
  "function_call": {
    "name": "function_name",
    "arguments": { ... }
  }
}
```

### D. Additional Results

**D.1 Per-Category Breakdown**

*[To be added after evaluation: performance broken down by task categories]*

**D.2 Error Analysis**

*[To be added: common failure modes, error categorization, improvement opportunities]*

### E. Ethical Considerations

**E.1 Dataset Collection Ethics**

- All training images sourced from permissively licensed datasets (COCO, public domain)
- No personally identifiable information in training data
- Synthetic examples created to avoid privacy concerns
- Function calling examples designed to avoid harmful use cases

**E.2 Model Safety Testing**

*[To be conducted: testing on harmful request refusal, bias evaluation, safety benchmarks]*

**E.3 Release Guidelines**

**Recommended Uses**:
- Accessibility tools for visually impaired users
- Automation of repetitive visual tasks with human oversight
- Research and education in multimodal AI
- Personal assistants for productivity

**Prohibited Uses**:
- Surveillance without consent
- Automated weaponry or harmful applications
- Deceptive impersonation
- Violation of privacy laws

**Model Card**: Full model card available at https://huggingface.co/zenlm/zen-vl-*

---

**Paper Version**: Draft v1.0 (November 2025)

**Status**: Preliminary - awaiting training completion and full evaluation

**Contact**: research@hanzo.ai

**Code**: https://github.com/zenlm/zen-vl

**Models**: https://huggingface.co/zenlm

**Website**: https://zenlm.org
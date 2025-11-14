# Romanian Llama 3.1 8B Fine-Tuning with Tinker

Fine-tuning Llama 3.1 8B Base for Romanian instruction-following using the Tinker framework from Thinking Machines.

## Overview

This project adapts Meta's Llama 3.1 8B model to better understand and generate Romanian text, specifically optimized for instruction-following tasks. Using Tinker's distributed training infrastructure and LoRA (Low-Rank Adaptation), we achieve efficient fine-tuning without requiring local GPU resources.

## Project Structure

```
romanian-llm-tinker/
├── data/
│   ├── raw/              # Original datasets (downloaded)
│   ├── processed/        # JSONL formatted training data
│   └── splits/           # Train/validation splits
├── scripts/
│   ├── download_datasets.py    # Fetch Romanian datasets
│   ├── prepare_data.py         # Data preprocessing & formatting
│   ├── train_tinker.py         # Main training script
│   ├── test_model.py           # Interactive model testing (no download needed)
│   ├── download_checkpoint.py  # Download checkpoints from Tinker
│   └── evaluate.py             # Model evaluation
├── configs/
│   └── hyperparams.yaml        # Training hyperparameters
├── checkpoints/
│   ├── checkpoint_step_*_metrics.json  # Training metrics per checkpoint
│   └── final_metrics.json              # Final training metrics
├── notebooks/
│   └── explore_data.ipynb      # Data exploration
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
└── README.md                  # This file
```

## Prerequisites

1. **Tinker Access**: Sign up for Tinker beta at https://thinkingmachines.ai/tinker/
2. **Python**: Version 3.8+ (recommended: 3.10)
3. **API Keys**: Tinker API key (required), HuggingFace token (optional)

## Setup

### 1. Clone and Navigate to Repository

```bash
cd romanian-llm-tinker
```

### 2. Create Virtual Environment

```bash
# Using conda
conda create -n romanian-tinker python=3.10
conda activate romanian-tinker

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Tinker credentials
# TINKER_API_KEY=your-key-here
# TINKER_KEY_NUMBER=your-number-here
```

### 5. Verify Tinker Connection

```python
from tinker import ServiceClient
import os
from dotenv import load_dotenv

load_dotenv()
client = ServiceClient()
print("Tinker connected successfully!")
```

## Quick Start

### Step 1: Download Romanian Datasets

```bash
python scripts/download_datasets.py --sources wiki oscar --size small
```

This will download and cache Romanian text from:
- Romanian Wikipedia (clean, factual)
- OSCAR Romanian subset (diverse web content)

### Step 2: Prepare Training Data

```bash
python scripts/prepare_data.py \
    --input data/raw \
    --output data/processed/train.jsonl \
    --format instruction \
    --num-examples 1000 \
    --split 0.8
```

This converts raw text into instruction-following format and creates train/validation splits.

### Step 3: Train the Model

```bash
python scripts/train_tinker.py \
    --config configs/hyperparams.yaml \
    --train-data data/splits/train.jsonl \
    --val-data data/splits/val.jsonl \
    --checkpoint-dir checkpoints/
```

Training will run on Tinker's infrastructure. Monitor progress in the Tinker console.

**Important**: Save your session ID from the training logs! You'll need it for testing. Look for:
```
INFO - ServiceClient initialized for session a65fa1a6-00b9-5a7e-9abf-59f068b79982
INFO - Creating TrainingClient for model_id='a65fa1a6-00b9-5a7e-9abf-59f068b79982:train:0'
```

### Step 4: Test Your Model

After training completes, test your model directly (no download needed):

```bash
# Interactive testing (recommended)
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID \
    --interactive

# Test single prompt
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID \
    --prompt "Care este capitala României?"

# Run predefined tests
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID
```

See the [Testing Your Model](#testing-your-model) section below for detailed testing options.

## Data Format

Training data must be in JSONL format with the following structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Care este capitala României?"
    },
    {
      "role": "assistant",
      "content": "Capitala României este București, cel mai mare oraș din țară și centru politic, economic și cultural."
    }
  ]
}
```

Each line in the JSONL file represents one training example with a conversation structure.

## Configuration

Edit `configs/hyperparams.yaml` to customize training:

```yaml
model:
  name: "meta-llama/Llama-3.1-8B"

lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: "all_linear_layers"

training:
  learning_rate: 1e-4
  max_steps: 1000
  batch_size: 4
  gradient_accumulation_steps: 1
  warmup_steps: 100
  save_steps: 100
  eval_steps: 50

optimizer:
  type: "adamw"
  weight_decay: 0.001
  gradient_clip: 0.01
```

## Training Strategy

### Phase 1: Quick Validation (100-200 examples, ~30 min)
- Verify pipeline works end-to-end
- Check data quality and formatting
- Ensure model is learning (loss decreases)

### Phase 2: Full Training (1000-1500 examples, 4-8 hours)
- Train on complete dataset
- Monitor validation metrics
- Save checkpoints regularly

### Phase 3: Evaluation & Refinement
- Test on held-out validation set
- Generate sample outputs manually
- Compare against base Llama 3.1 8B
- Adjust hyperparameters if needed

## Data Sources

### Public Romanian Datasets

1. **Wikipedia Romanian** - Clean, factual text
2. **OSCAR-2201** - Diverse web content
3. **Translation of Alpaca/Dolly** - Instruction-following examples

### Data Acquisition Options

```bash
# Download from HuggingFace
python scripts/download_datasets.py --source hf --dataset oscar-corpus/OSCAR-2201 --language ro

# Scrape Romanian Q&A forums
python scripts/download_datasets.py --source scrape --url https://romanian-forum.com

# Translate English instructions
python scripts/download_datasets.py --source translate --input alpaca.json --target ro
```

## Testing Your Model

After training completes, you can test your model in multiple ways. Your trained model weights live on Tinker's infrastructure, so no downloads are required!

### Method 1: Interactive Testing (Recommended)

The easiest way to test your model is with interactive mode:

```bash
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID \
    --interactive
```

This opens an interactive prompt where you can:
- Type Romanian prompts and get instant responses
- Type `test` to run predefined tests
- Type `quit` to exit

**Example session:**
```
🇷🇴 Romanian Prompt: Care este capitala României?

⏳ Generating response...

🤖 Response:
Capitala României este București, cel mai mare oraș din țară...
```

### Method 2: Single Prompt Testing

Test with a specific prompt:

```bash
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID \
    --prompt "Explică ce este inteligența artificială."
```

### Method 3: Predefined Test Suite

Run a suite of 5 predefined Romanian prompts:

```bash
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID
```

This tests:
- Factual questions (e.g., "Care este capitala României?")
- Explanations (e.g., "Explică ce este inteligența artificială")
- Creative writing (e.g., "Scrie o scurtă poezie despre primăvară")
- List generation (e.g., "Care sunt cele mai mari orașe din România?")
- Summarization tasks

### Method 4: Compare with Base Model

See how much your fine-tuning improved the model:

```bash
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID \
    --compare
```

This runs the same prompts through both your fine-tuned model and the base Llama 3.1 8B, showing side-by-side comparisons.

### Test Script Options

```bash
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID \       # Required: Your Tinker session ID
    --checkpoint checkpoint_final \      # Checkpoint name (default: checkpoint_final)
    --interactive \                      # Enable interactive mode
    --prompt "Your prompt here" \        # Test single prompt
    --compare \                          # Compare with base model
    --max-tokens 256 \                   # Max tokens to generate (default: 256)
    --model meta-llama/Llama-3.1-8B \   # Base model name
    --rank 8                             # LoRA rank used in training
```

### Finding Your Session ID

Your session ID is in the training logs. Look for lines like:
```
2025-11-13 15:53:44,963 - INFO - ServiceClient initialized for session a65fa1a6-00b9-5a7e-9abf-59f068b79982
```

Or check your training metrics file:
```bash
# View your training progress
cat checkpoints/final_metrics.json | python -m json.tool | head -20
```

## Downloading Checkpoints (Optional)

If you need to download checkpoint weights for local use or deployment:

```bash
python scripts/download_checkpoint.py \
    --session-id YOUR_SESSION_ID \
    --checkpoint checkpoint_final \
    --output-dir checkpoints/downloads
```

**Note**: Tinker's checkpoint archiving can take several minutes. The script will automatically retry if the archive is still being created.

### Download Options

```bash
# Download specific checkpoint
python scripts/download_checkpoint.py \
    --session-id YOUR_SESSION_ID \
    --checkpoint checkpoint_step_900

# Try downloading all available checkpoints
python scripts/download_checkpoint.py \
    --session-id YOUR_SESSION_ID \
    --all
```

Downloaded checkpoints will be extracted to `checkpoints/downloads/`.

## Evaluation Metrics

After testing, review your model's training progress:

```bash
# View final training loss
python -c "import json; m=json.load(open('checkpoints/final_metrics.json')); print(f'Final loss: {m[\"train_losses\"][-1]:.2f}')"

# View all checkpoint metrics
ls -lh checkpoints/checkpoint_step_*_metrics.json
```

Evaluation criteria:
- **Training Loss**: Should decrease significantly (e.g., 400+ → <5)
- **Response Quality**: Fluent, grammatically correct Romanian
- **Instruction Following**: Model completes the requested task
- **Factual Accuracy**: Correct answers to knowledge questions
- **Creativity**: Ability to generate poems, stories, etc.

## Troubleshooting

### Tinker Connection Issues

```python
# Verify environment variables
import os
print(os.getenv("TINKER_API_KEY"))

# Test connection
from tinker import ServiceClient
client = ServiceClient()
```

### Testing Issues

**Problem**: "Error loading checkpoint: Path is invalid"
```bash
# Solution: Test without loading checkpoint (uses current model state)
python scripts/test_model.py \
    --session-id YOUR_SESSION_ID \
    --no-checkpoint \
    --interactive
```

**Problem**: Can't find session ID
```bash
# Check training logs for session ID
grep "ServiceClient initialized" train.log

# Or check most recent training
ls -lt checkpoints/*.json | head -1
```

**Problem**: "SamplingClient error" or API issues
```bash
# Verify Tinker connection
python -c "from tinker import ServiceClient; print('Connected:', ServiceClient())"

# Check if your session is still active (sessions may expire)
# You may need to run training again to get a fresh session
```

### Data Format Errors

```bash
# Validate JSONL format
python scripts/prepare_data.py --validate data/processed/train.jsonl
```

### Out of Memory

Reduce batch size in `configs/hyperparams.yaml`:
```yaml
training:
  batch_size: 2
```

### Checkpoint Download Issues

**Problem**: "Archive creation in progress" for a long time
- Tinker's archive service can take 5-10+ minutes
- The download script will automatically retry
- Alternatively, test directly without downloading (see [Testing Your Model](#testing-your-model))

**Problem**: "404 - Model not found"
- Verify your session ID is correct
- Check that training completed successfully
- Note: Checkpoint paths use the format `checkpoint_step_100`, `checkpoint_final`, etc.

## Best Practices

1. **Start Small**: Begin with 100-200 examples to validate your pipeline
2. **Monitor Training**: Check loss curves and sample outputs regularly
3. **Quality Over Quantity**: 1000 high-quality examples > 10000 poor examples
4. **Save Your Session ID**: You'll need it for testing - it's in the training logs
5. **Test Early and Often**: Use interactive mode to test during training
6. **Save Checkpoints**: Regularly save to prevent data loss (every 100 steps recommended)
7. **Version Control**: Track configs, data preprocessing steps, and session IDs
8. **Compare Models**: Always compare fine-tuned vs base model to measure improvement

## Resources

- **Tinker Documentation**: https://tinker-docs.thinkingmachines.ai/
- **Tinker Cookbook**: https://github.com/thinking-machines-lab/tinker-cookbook
- **Llama 3.1 Model Card**: https://huggingface.co/meta-llama/Llama-3.1-8B
- **Romanian Datasets**: https://github.com/AndyTheFactory/romanian-nlp-datasets
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

## Success Criteria

After training, your model should demonstrate:

✅ **Training Loss Reduction**: Loss decreases from 400+ to <5
✅ **Fluent Romanian**: Grammatically correct, natural-sounding text
✅ **Instruction Following**: Completes requested tasks accurately
✅ **Factual Knowledge**: Correct answers to Romanian knowledge questions
✅ **Creative Ability**: Can generate poems, stories, explanations
✅ **Improvement over Base**: Better than untuned Llama 3.1 8B on Romanian tasks

### Example Success Metrics

From a successful training run:
```json
{
  "initial_loss": 428.5,
  "final_loss": 1.2,
  "total_steps": 1000,
  "training_time": "~2 hours"
}
```

Test your model with:
```bash
python scripts/test_model.py --session-id YOUR_SESSION_ID --interactive
```

## Next Steps

After completing the prototype:

1. **Scale Up**: Increase to 5K-10K examples
2. **Domain Specialization**: Add domain-specific data (medical, legal, etc.)
3. **Multi-Task**: Train on diverse task types
4. **Deployment**: Export model for production use
5. **Continuous Improvement**: Collect user feedback and iterate

## License

This project uses Meta's Llama 3.1 model. Please review the [Llama 3.1 License](https://ai.meta.com/llama/license/) for usage terms.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainer.

## Acknowledgments

- **Thinking Machines** for the Tinker framework
- **Meta AI** for Llama 3.1
- **Romanian NLP Community** for dataset resources

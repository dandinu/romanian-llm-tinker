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
│   ├── download_datasets.py   # Fetch Romanian datasets
│   ├── prepare_data.py        # Data preprocessing & formatting
│   ├── train_tinker.py        # Main training script
│   └── evaluate.py            # Model evaluation
├── configs/
│   └── hyperparams.yaml       # Training hyperparameters
├── notebooks/
│   └── explore_data.ipynb     # Data exploration
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
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

### Step 4: Evaluate Results

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/final \
    --test-data data/splits/val.jsonl \
    --output results.json
```

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

## Evaluation Metrics

- **Perplexity**: Lower is better (measures prediction confidence)
- **ROUGE Score**: Overlap with reference responses
- **Manual Review**: Native speaker assessment of fluency and correctness
- **Instruction Following**: Does the model complete the requested task?

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

### Data Format Errors

```bash
# Validate JSONL format
python scripts/prepare_data.py --validate data/processed/train.jsonl
```

### Out of Memory

Reduce batch size in `configs/hyperparams.yaml`:
```yaml
training:
  batch_size: 2  # Reduced from 4
```

## Best Practices

1. **Start Small**: Begin with 100-200 examples to validate your pipeline
2. **Monitor Training**: Check loss curves and sample outputs regularly
3. **Quality Over Quantity**: 1000 high-quality examples > 10000 poor examples
4. **Save Checkpoints**: Regularly save to prevent data loss
5. **Version Control**: Track configs and data preprocessing steps

## Resources

- **Tinker Documentation**: https://tinker-docs.thinkingmachines.ai/
- **Tinker Cookbook**: https://github.com/thinking-machines-lab/tinker-cookbook
- **Llama 3.1 Model Card**: https://huggingface.co/meta-llama/Llama-3.1-8B
- **Romanian Datasets**: https://github.com/AndyTheFactory/romanian-nlp-datasets
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

## Timeline

- **Days 1-2**: Setup and Tinker access
- **Days 3-5**: Data collection and preparation
- **Days 6-7**: Training configuration
- **Days 8-11**: Initial training and iteration
- **Days 12-14**: Evaluation and refinement

## Success Criteria

- Model generates fluent Romanian text
- Successfully follows instructions in Romanian
- Outperforms base Llama 3.1 8B on Romanian tasks
- Achieves target perplexity on validation set

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

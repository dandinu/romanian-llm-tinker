#!/usr/bin/env python3
"""
Train Llama 3.1 8B for Romanian instruction-following using Tinker.

This script uses the Tinker framework to fine-tune Llama 3.1 8B Base
on Romanian instruction data with LoRA.

Usage:
    python scripts/train_tinker.py --config configs/hyperparams.yaml
    python scripts/train_tinker.py --quick-test  # Quick validation run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Import Tinker
try:
    from tinker import ServiceClient, types
    from tinker_cookbook import renderers, tokenizer_utils
except ImportError:
    print("Error: Tinker not installed. Run: pip install tinker tinker-cookbook")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RomanianLlamaTrainer:
    """Train Llama 3.1 8B for Romanian using Tinker."""

    def __init__(self, config_path: str):
        """Initialize the trainer.

        Args:
            config_path: Path to hyperparameters YAML file
        """
        # Load environment variables
        load_dotenv()

        # Load configuration
        self.config = self._load_config(config_path)

        # Verify API key
        self.api_key = os.getenv('TINKER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "TINKER_API_KEY not found. "
                "Set it in .env or export TINKER_API_KEY=your-key"
            )

        # Initialize Tinker client
        logger.info("Connecting to Tinker...")
        self.client = ServiceClient()

        # Initialize training client (will be set up in setup_training)
        self.training_client = None
        self.tokenizer = None
        self.renderer = None

        logger.info("Trainer initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def setup_training(self) -> None:
        """Set up Tinker training client with LoRA configuration."""
        model_name = self.config['model']['name']
        lora_config = self.config['lora']

        logger.info(f"Setting up training for: {model_name}")
        logger.info(f"LoRA config: rank={lora_config['rank']}, alpha={lora_config['alpha']}")

        # Create LoRA training client
        # Note: Tinker API uses 'rank' parameter (alpha is handled internally)
        self.training_client = self.client.create_lora_training_client(
            base_model=model_name,
            rank=lora_config['rank']
        )

        # Get tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = self.training_client.get_tokenizer()

        # Get renderer for chat formatting
        base_type = self.config['model']['base_type']
        self.renderer = renderers.get_renderer(base_type, self.tokenizer)

        logger.info("Training setup complete")

    def load_training_data(self, data_path: str, max_examples: Optional[int] = None) -> List[types.Datum]:
        """Load and preprocess training data.

        Args:
            data_path: Path to JSONL training file
            max_examples: Maximum examples to load (None = all)

        Returns:
            List of Tinker Datum objects
        """
        logger.info(f"Loading training data from {data_path}")

        processed_data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            if max_examples:
                lines = lines[:max_examples]

            for line in tqdm(lines, desc="Processing data"):
                try:
                    example = json.loads(line)

                    # Extract messages
                    messages = example['messages']

                    # Build supervised example with Tinker renderer
                    tokens, weights = self.renderer.build_supervised_example(messages)

                    # Follow Tinker Cookbook pattern: datum_from_tokens_weights
                    # input_tokens = tokens[:-1], target_tokens = tokens[1:], weights = weights[1:]
                    input_tokens = tokens[:-1]
                    target_tokens = tokens[1:]
                    weights_shifted = weights[1:]

                    # Create ModelInput from input tokens
                    model_input = types.ModelInput.from_ints(input_tokens.tolist())

                    # Create TensorData for both target_tokens and weights
                    from tinker import TensorData
                    target_tokens_data = TensorData(
                        data=[int(x) for x in target_tokens.tolist()],
                        dtype="int64",
                        shape=list(target_tokens.shape)
                    )
                    weights_data = TensorData(
                        data=weights_shifted.tolist(),
                        dtype="float32",
                        shape=list(weights_shifted.shape)
                    )

                    # Create Datum object with both target_tokens and weights
                    datum = types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            'target_tokens': target_tokens_data,
                            'weights': weights_data
                        }
                    )

                    processed_data.append(datum)

                except Exception as e:
                    logger.warning(f"Error processing example: {str(e)}")
                    continue

        logger.info(f"Loaded {len(processed_data)} training examples")
        return processed_data

    def train(
        self,
        train_data: List[types.Datum],
        val_data: Optional[List[types.Datum]] = None,
        checkpoint_dir: str = "checkpoints"
    ) -> None:
        """Run training loop.

        Args:
            train_data: Training data
            val_data: Validation data (optional)
            checkpoint_dir: Directory to save checkpoints
        """
        training_config = self.config['training']
        optimizer_config = self.config['optimizer']

        max_steps = training_config['max_steps']
        eval_steps = training_config['eval_steps']
        save_steps = training_config['save_steps']
        logging_steps = training_config['logging_steps']
        learning_rate = training_config['learning_rate']
        batch_size = training_config['batch_size']

        logger.info("Starting training...")
        logger.info(f"  Max steps: {max_steps}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Eval every: {eval_steps} steps")
        logger.info(f"  Save every: {save_steps} steps")

        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Training metrics
        metrics = {
            'train_losses': [],
            'eval_losses': [],
            'step': 0
        }

        # Training loop
        pbar = tqdm(range(max_steps), desc="Training")

        for step in pbar:
            step_start = time.time()

            # Sample batch from training data
            batch = self._sample_batch(train_data, batch_size)

            # Forward + backward pass
            try:
                fwd_result = self.training_client.forward_backward(
                    batch,
                    "cross_entropy"
                ).result()

                # Extract loss from metrics (Tinker provides loss:sum in metrics dict)
                # Or compute from logprobs like Tinker Cookbook does
                if 'loss:sum' in fwd_result.metrics:
                    loss = fwd_result.metrics['loss:sum']
                else:
                    # Fallback: compute mean NLL from logprobs and weights
                    logprobs = [x["logprobs"] for x in fwd_result.loss_fn_outputs]
                    weights = [datum.loss_fn_inputs["weights"] for datum in batch]

                    # Simple mean NLL computation
                    total_loss = 0.0
                    total_weight = 0.0
                    for lp, w in zip(logprobs, weights):
                        lp_tensor = lp.to_torch() if hasattr(lp, 'to_torch') else lp
                        w_tensor = w.to_torch() if hasattr(w, 'to_torch') else w
                        total_loss += -(lp_tensor * w_tensor).sum().item()
                        total_weight += w_tensor.sum().item()

                    loss = total_loss / max(total_weight, 1.0)

                metrics['train_losses'].append(loss)

                # Optimizer step
                self.training_client.optim_step(
                    types.AdamParams(
                        learning_rate=learning_rate,
                        beta1=optimizer_config.get('beta1', 0.9),
                        beta2=optimizer_config.get('beta2', 0.999),
                        eps=optimizer_config.get('eps', 1e-8)
                    )
                ).result()

                # Update progress bar
                if step % logging_steps == 0:
                    avg_loss = sum(metrics['train_losses'][-logging_steps:]) / min(logging_steps, len(metrics['train_losses']))
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'step_time': f'{time.time() - step_start:.2f}s'
                    })

            except Exception as e:
                logger.error(f"Error at step {step}: {str(e)}")
                continue

            # Evaluation
            if val_data and step > 0 and step % eval_steps == 0:
                eval_loss = self._evaluate(val_data)
                metrics['eval_losses'].append((step, eval_loss))
                logger.info(f"Step {step}: Eval loss = {eval_loss:.4f}")

            # Save checkpoint
            if step > 0 and step % save_steps == 0:
                checkpoint_name = f"checkpoint_step_{step}"
                logger.info(f"Saving checkpoint: {checkpoint_name}")

                try:
                    self.training_client.save_state(checkpoint_name).result()

                    # Save metrics
                    metrics_file = checkpoint_path / f"{checkpoint_name}_metrics.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)

                except Exception as e:
                    logger.error(f"Error saving checkpoint: {str(e)}")

            metrics['step'] = step + 1

        # Save final checkpoint
        logger.info("Training complete! Saving final checkpoint...")
        try:
            self.training_client.save_state("checkpoint_final").result()

            # Save final metrics
            final_metrics_file = checkpoint_path / "final_metrics.json"
            with open(final_metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving final checkpoint: {str(e)}")

        logger.info(f"Training finished! Checkpoints saved to {checkpoint_dir}")

    def _sample_batch(self, data: List[types.Datum], batch_size: int) -> List[types.Datum]:
        """Sample a random batch from data.

        Args:
            data: Full dataset
            batch_size: Number of examples to sample

        Returns:
            Batch of examples
        """
        import random
        return random.sample(data, min(batch_size, len(data)))

    def _evaluate(self, val_data: List[types.Datum]) -> float:
        """Evaluate model on validation data.

        Args:
            val_data: Validation dataset

        Returns:
            Average validation loss
        """
        eval_batch_size = self.config['evaluation']['per_device_eval_batch_size']
        total_loss = 0.0
        num_batches = 0

        # Evaluate on a subset of validation data
        max_eval_examples = min(100, len(val_data))
        eval_subset = val_data[:max_eval_examples]

        for i in range(0, len(eval_subset), eval_batch_size):
            batch = eval_subset[i:i + eval_batch_size]

            try:
                # Forward pass only (no backward)
                fwd_result = self.training_client.forward_backward(
                    batch,
                    "cross_entropy"
                ).result()

                # Extract loss from metrics (same as training loop)
                if 'loss:sum' in fwd_result.metrics:
                    batch_loss = fwd_result.metrics['loss:sum']
                else:
                    # Fallback: compute mean NLL from logprobs and weights
                    logprobs = [x["logprobs"] for x in fwd_result.loss_fn_outputs]
                    weights = [datum.loss_fn_inputs["weights"] for datum in batch]

                    batch_total = 0.0
                    batch_weight = 0.0
                    for lp, w in zip(logprobs, weights):
                        lp_tensor = lp.to_torch() if hasattr(lp, 'to_torch') else lp
                        w_tensor = w.to_torch() if hasattr(w, 'to_torch') else w
                        batch_total += -(lp_tensor * w_tensor).sum().item()
                        batch_weight += w_tensor.sum().item()

                    batch_loss = batch_total / max(batch_weight, 1.0)

                total_loss += batch_loss
                num_batches += 1

            except Exception as e:
                logger.warning(f"Error in evaluation: {str(e)}")
                continue

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def sample_generation(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from the model (for testing).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            # Tokenize prompt
            tokens = self.tokenizer.encode(prompt)

            # Sample from model
            result = self.training_client.sample(
                types.Datum(model_input=tokens),
                types.SampleParams(max_tokens=max_tokens)
            ).result()

            # Decode
            generated_text = self.tokenizer.decode(result.tokens)
            return generated_text

        except Exception as e:
            logger.error(f"Error generating: {str(e)}")
            return ""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Llama 3.1 8B for Romanian with Tinker"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/hyperparams.yaml',
        help='Path to hyperparameters config'
    )

    parser.add_argument(
        '--train-data',
        type=str,
        default='data/splits/train.jsonl',
        help='Path to training data JSONL'
    )

    parser.add_argument(
        '--val-data',
        type=str,
        default='data/splits/val.jsonl',
        help='Path to validation data JSONL'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test run with limited data/steps'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize trainer
    try:
        trainer = RomanianLlamaTrainer(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check your Tinker API key in .env")
        logger.error("  2. Verify Tinker installation: pip install tinker tinker-cookbook")
        logger.error("  3. Test connection: python -c 'from tinker import ServiceClient; ServiceClient()'")
        return

    # Setup training
    trainer.setup_training()

    # Load data
    if args.quick_test:
        logger.info("Quick test mode: using limited data")
        max_examples = 50
    else:
        max_examples = None

    train_data = trainer.load_training_data(args.train_data, max_examples)

    # Load validation data if available
    val_data = None
    if Path(args.val_data).exists():
        val_data = trainer.load_training_data(args.val_data, max_examples // 5 if max_examples else None)
    else:
        logger.warning(f"Validation data not found: {args.val_data}")

    # Train
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        checkpoint_dir=args.checkpoint_dir
    )

    logger.info("\nTraining pipeline complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Review training metrics in checkpoints/")
    logger.info("  2. Evaluate model: python scripts/evaluate.py")
    logger.info("  3. Download checkpoint for local testing")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test trained Romanian Llama model directly from Tinker session.

This script connects to your Tinker training session and tests the model
without needing to download checkpoint files.

Usage:
    # Interactive mode
    python scripts/test_model.py --session-id a65fa1a6-00b9-5a7e-9abf-59f068b79982 --interactive

    # Test with predefined prompts
    python scripts/test_model.py --session-id a65fa1a6-00b9-5a7e-9abf-59f068b79982

    # Test with custom prompt
    python scripts/test_model.py --session-id a65fa1a6-00b9-5a7e-9abf-59f068b79982 --prompt "Care este capitala României?"

    # Compare with base model
    python scripts/test_model.py --session-id a65fa1a6-00b9-5a7e-9abf-59f068b79982 --compare
"""

import argparse
import logging
import os
import sys
from typing import Optional, List, Dict

from dotenv import load_dotenv

# Import Tinker
try:
    from tinker import ServiceClient, types
    from tinker_cookbook import renderers
except ImportError:
    print("Error: Tinker not installed. Run: pip install tinker tinker-cookbook")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Test Romanian Llama model from Tinker session."""

    # Test prompts in Romanian
    TEST_PROMPTS = [
        "Care este capitala României?",
        "Explică ce este inteligența artificială.",
        "Scrie o scurtă poezie despre primăvară.",
        "Care sunt cele mai mari orașe din România?",
        "Ce este fotosinteza?",
    ]

    def __init__(
        self,
        session_id: str,
        checkpoint_name: str = "checkpoint_final",
        model_name: str = "meta-llama/Llama-3.1-8B",
        lora_rank: int = 8
    ):
        """Initialize the tester.

        Args:
            session_id: Tinker training session ID
            checkpoint_name: Name of checkpoint to load
            model_name: Base model name
            lora_rank: LoRA rank used during training
        """
        # Load environment variables
        load_dotenv()

        # Verify API key
        self.api_key = os.getenv('TINKER_API_KEY')
        if not self.api_key:
            raise ValueError("TINKER_API_KEY not found in .env")

        self.session_id = session_id
        self.checkpoint_name = checkpoint_name
        self.model_name = model_name
        self.lora_rank = lora_rank

        # Initialize Tinker client
        logger.info("Connecting to Tinker...")
        self.client = ServiceClient()

        # Training client and sampling client (will be set up later)
        self.training_client = None
        self.sampling_client = None
        self.tokenizer = None
        self.renderer = None

        logger.info(f"Connected to session: {session_id}")

    def setup_model(self, load_checkpoint: bool = True) -> None:
        """Set up model for testing.

        Args:
            load_checkpoint: Whether to load the trained checkpoint
        """
        logger.info(f"Setting up model: {self.model_name}")
        logger.info(f"LoRA rank: {self.lora_rank}")

        # Create training client
        self.training_client = self.client.create_lora_training_client(
            base_model=self.model_name,
            rank=self.lora_rank
        )

        # Load checkpoint if requested
        if load_checkpoint and self.checkpoint_name:
            logger.info(f"Loading checkpoint: {self.checkpoint_name}")
            try:
                self.training_client.load_state(self.checkpoint_name).result()
                logger.info(f"✓ Checkpoint '{self.checkpoint_name}' loaded successfully!")
            except Exception as e:
                logger.error(f"✗ Error loading checkpoint: {str(e)}")
                logger.warning("Continuing with current model state...")

        # Get tokenizer and renderer
        logger.info("Initializing tokenizer and renderer...")
        self.tokenizer = self.training_client.get_tokenizer()
        self.renderer = renderers.get_renderer('llama3', self.tokenizer)

        # Create sampling client for inference
        logger.info("Creating sampling client for inference...")
        try:
            self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
                "test_model"
            )
            logger.info("✓ Sampling client created!")
        except Exception as e:
            logger.error(f"✗ Error creating sampling client: {str(e)}")
            raise

        logger.info("✓ Model setup complete!")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        show_prompt: bool = True
    ) -> str:
        """Generate response to a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            show_prompt: Whether to include prompt in output

        Returns:
            Generated response
        """
        try:
            # Format as chat message
            messages = [
                {'role': 'user', 'content': prompt}
            ]

            # Build generation prompt (not supervised example)
            model_input = self.renderer.build_generation_prompt(messages)

            # Sample from model using sampling client
            result = self.sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            ).result()

            # Decode the first sample
            generated_text = self.tokenizer.decode(result.sequences[0].tokens)

            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"[ERROR: {str(e)}]"

    def test_single_prompt(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> None:
        """Test with a single prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        print("\n" + "=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)
        print("\nGenerating response...")

        response = self.generate_response(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        print("\nRESPONSE:")
        print("-" * 80)
        print(response)
        print("-" * 80)

    def test_predefined_prompts(self, max_tokens: int = 256) -> None:
        """Test with predefined Romanian prompts.

        Args:
            max_tokens: Maximum tokens to generate
        """
        logger.info(f"\nTesting {len(self.TEST_PROMPTS)} predefined prompts...")

        for i, prompt in enumerate(self.TEST_PROMPTS, 1):
            print("\n" + "=" * 80)
            print(f"TEST {i}/{len(self.TEST_PROMPTS)}")
            print("=" * 80)
            print(f"\nPROMPT: {prompt}")
            print("\nGenerating response...")

            response = self.generate_response(prompt, max_tokens=max_tokens)

            print("\nRESPONSE:")
            print("-" * 80)
            print(response)
            print("-" * 80)

            if i < len(self.TEST_PROMPTS):
                input("\nPress Enter to continue to next test...")

    def compare_with_base(self, prompts: Optional[List[str]] = None) -> None:
        """Compare fine-tuned model with base model.

        Args:
            prompts: List of prompts to test (uses defaults if None)
        """
        if prompts is None:
            prompts = self.TEST_PROMPTS[:3]  # Use first 3 for comparison

        logger.info("\n" + "=" * 80)
        logger.info("COMPARING FINE-TUNED MODEL WITH BASE MODEL")
        logger.info("=" * 80)

        for i, prompt in enumerate(prompts, 1):
            print(f"\n\n{'='*80}")
            print(f"PROMPT {i}/{len(prompts)}: {prompt}")
            print("=" * 80)

            # Fine-tuned model response
            print("\n[FINE-TUNED MODEL]")
            print("-" * 80)
            finetuned_response = self.generate_response(prompt, max_tokens=200)
            print(finetuned_response)
            print("-" * 80)

            # Base model response
            print("\n[BASE MODEL]")
            print("-" * 80)
            print("(Loading base model...)")

            # Create a new client for base model
            base_training_client = self.client.create_lora_training_client(
                base_model=self.model_name,
                rank=self.lora_rank
            )
            base_tokenizer = base_training_client.get_tokenizer()
            base_renderer = renderers.get_renderer('llama3', base_tokenizer)

            # Create sampling client for base model
            base_sampling_client = base_training_client.save_weights_and_get_sampling_client(
                "base_model_test"
            )

            messages = [{'role': 'user', 'content': prompt}]
            model_input = base_renderer.build_generation_prompt(messages)

            result = base_sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=200, temperature=0.7)
            ).result()

            base_response = base_tokenizer.decode(result.sequences[0].tokens)
            print(base_response)
            print("-" * 80)

            if i < len(prompts):
                input("\nPress Enter to continue to next comparison...")

    def interactive_mode(self) -> None:
        """Run interactive testing mode."""
        logger.info("\n" + "=" * 80)
        logger.info("INTERACTIVE MODE")
        logger.info("=" * 80)
        print("\nEnter Romanian prompts to test the model.")
        print("Commands:")
        print("  - Type your prompt and press Enter")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'test' to run predefined tests")
        print()

        while True:
            try:
                prompt = input("\n🇷🇴 Romanian Prompt: ").strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting interactive mode...")
                    break

                if prompt.lower() == 'test':
                    self.test_predefined_prompts()
                    continue

                if not prompt:
                    continue

                print("\n⏳ Generating response...")
                response = self.generate_response(prompt, max_tokens=512)

                print("\n🤖 Response:")
                print("-" * 80)
                print(response)
                print("-" * 80)

            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Romanian Llama model from Tinker session"
    )

    parser.add_argument(
        '--session-id',
        type=str,
        required=True,
        help='Tinker training session ID'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint_final',
        help='Checkpoint name to load (default: checkpoint_final)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-3.1-8B',
        help='Base model name (default: meta-llama/Llama-3.1-8B)'
    )

    parser.add_argument(
        '--rank',
        type=int,
        default=8,
        help='LoRA rank used during training (default: 8)'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Single prompt to test'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare fine-tuned model with base model'
    )

    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Test without loading checkpoint (use current state)'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate (default: 256)'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    try:
        # Initialize tester
        tester = ModelTester(
            session_id=args.session_id,
            checkpoint_name=args.checkpoint,
            model_name=args.model,
            lora_rank=args.rank
        )

        # Setup model
        tester.setup_model(load_checkpoint=not args.no_checkpoint)

        # Run appropriate mode
        if args.interactive:
            tester.interactive_mode()
        elif args.compare:
            tester.compare_with_base()
        elif args.prompt:
            tester.test_single_prompt(args.prompt, max_tokens=args.max_tokens)
        else:
            tester.test_predefined_prompts(max_tokens=args.max_tokens)

        logger.info("\n✓ Testing complete!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

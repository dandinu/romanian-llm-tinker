#!/usr/bin/env python3
"""
Evaluate fine-tuned Romanian Llama model.

This script evaluates the fine-tuned model on various Romanian tasks
and compares against the base model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoint_final
    python scripts/evaluate.py --interactive  # Interactive testing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Import Tinker
try:
    from tinker import ServiceClient, types, RestClient
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


class RomanianModelEvaluator:
    """Evaluate Romanian Llama model."""

    # Romanian test prompts
    TEST_PROMPTS = [
        {
            'prompt': 'Care este capitala României?',
            'category': 'factual',
            'expected_keywords': ['București', 'Bucuresti', 'capitală']
        },
        {
            'prompt': 'Explică procesul de fotosinteză în câteva propoziții.',
            'category': 'explanation',
            'expected_keywords': ['plante', 'lumină', 'oxigen', 'dioxid de carbon']
        },
        {
            'prompt': 'Scrie o scurtă descriere despre importanța educației.',
            'category': 'generation',
            'expected_keywords': ['cunoștințe', 'învățare', 'dezvoltare']
        },
        {
            'prompt': 'Care sunt cele mai mari orașe din România?',
            'category': 'list',
            'expected_keywords': ['București', 'Cluj', 'Timișoara', 'Iași']
        },
        {
            'prompt': 'Rezumă următoarea idee: Inteligența artificială transformă industria tehnologică.',
            'category': 'summarization',
            'expected_keywords': ['AI', 'tehnologie', 'transformare', 'industrie']
        },
    ]

    def __init__(self, checkpoint_name: Optional[str] = None):
        """Initialize the evaluator.

        Args:
            checkpoint_name: Name of checkpoint to load (None = base model)
        """
        # Load environment variables
        load_dotenv()

        # Verify API key
        self.api_key = os.getenv('TINKER_API_KEY')
        if not self.api_key:
            raise ValueError("TINKER_API_KEY not found in .env")

        # Initialize Tinker client
        logger.info("Connecting to Tinker...")
        self.client = ServiceClient()
        self.rest_client = RestClient()

        self.checkpoint_name = checkpoint_name

        # Training client (will be set up later)
        self.training_client = None
        self.tokenizer = None
        self.renderer = None

        logger.info("Evaluator initialized")

    def setup_model(self, model_name: str = "meta-llama/Llama-3.1-8B") -> None:
        """Set up model for evaluation.

        Args:
            model_name: Base model name
        """
        logger.info(f"Setting up model: {model_name}")

        # Create training client (even for evaluation)
        self.training_client = self.client.create_lora_training_client(
            base_model=model_name,
            lora_rank=8,
            lora_alpha=16
        )

        # Load checkpoint if specified
        if self.checkpoint_name:
            logger.info(f"Loading checkpoint: {self.checkpoint_name}")
            try:
                # In Tinker, you would load the saved state here
                # self.training_client.load_state(self.checkpoint_name).result()
                logger.warning("Checkpoint loading not fully implemented - using current state")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")

        # Get tokenizer
        self.tokenizer = self.training_client.get_tokenizer()
        self.renderer = renderers.get_renderer('llama3', self.tokenizer)

        logger.info("Model setup complete")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate response to a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        try:
            # Format as chat message
            messages = [
                {'role': 'user', 'content': prompt}
            ]

            # Build input
            tokens, _ = self.renderer.build_supervised_example(messages)

            # Sample from model
            result = self.training_client.sample(
                types.Datum(model_input=tokens),
                types.SampleParams(
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            ).result()

            # Decode
            generated_text = self.tokenizer.decode(result.tokens)

            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""

    def evaluate_test_set(self, test_prompts: Optional[List[Dict]] = None) -> Dict:
        """Evaluate on test prompts.

        Args:
            test_prompts: List of test prompts (uses defaults if None)

        Returns:
            Evaluation results
        """
        if test_prompts is None:
            test_prompts = self.TEST_PROMPTS

        logger.info(f"Evaluating on {len(test_prompts)} test prompts...")

        results = {
            'prompts': [],
            'responses': [],
            'scores': [],
            'categories': {}
        }

        for test in tqdm(test_prompts, desc="Evaluating"):
            prompt = test['prompt']
            category = test['category']
            expected_keywords = test.get('expected_keywords', [])

            # Generate response
            response = self.generate_response(prompt)

            # Simple keyword-based scoring
            score = self._score_response(response, expected_keywords)

            # Store results
            results['prompts'].append(prompt)
            results['responses'].append(response)
            results['scores'].append(score)

            # Category aggregation
            if category not in results['categories']:
                results['categories'][category] = {
                    'count': 0,
                    'total_score': 0.0,
                    'examples': []
                }

            results['categories'][category]['count'] += 1
            results['categories'][category]['total_score'] += score
            results['categories'][category]['examples'].append({
                'prompt': prompt,
                'response': response,
                'score': score
            })

        # Calculate category averages
        for category in results['categories']:
            cat_data = results['categories'][category]
            cat_data['avg_score'] = cat_data['total_score'] / cat_data['count']

        # Overall average
        results['avg_score'] = sum(results['scores']) / len(results['scores'])

        return results

    def _score_response(self, response: str, expected_keywords: List[str]) -> float:
        """Score response based on expected keywords.

        Args:
            response: Generated response
            expected_keywords: List of expected keywords

        Returns:
            Score between 0 and 1
        """
        if not expected_keywords:
            # Basic checks if no keywords
            if len(response) > 20:
                return 0.5
            return 0.0

        # Count keyword matches (case-insensitive)
        response_lower = response.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in response_lower)

        score = matches / len(expected_keywords)
        return score

    def evaluate_on_file(self, test_file: str) -> Dict:
        """Evaluate on a JSONL test file.

        Args:
            test_file: Path to JSONL test file

        Returns:
            Evaluation results
        """
        logger.info(f"Loading test data from {test_file}")

        test_prompts = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = data.get('messages', [])

                # Extract user prompt
                for msg in messages:
                    if msg['role'] == 'user':
                        test_prompts.append({
                            'prompt': msg['content'],
                            'category': 'file_test',
                            'expected_keywords': []
                        })
                        break

        return self.evaluate_test_set(test_prompts)

    def interactive_mode(self) -> None:
        """Run interactive evaluation mode."""
        logger.info("\nInteractive Evaluation Mode")
        logger.info("Enter Romanian prompts to test the model.")
        logger.info("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                prompt = input("Prompt (Romanian): ").strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    break

                if not prompt:
                    continue

                print("\nGenerating response...")
                response = self.generate_response(prompt, max_tokens=512)

                print(f"\nResponse:\n{response}\n")
                print("-" * 80 + "\n")

            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")

    def save_results(self, results: Dict, output_file: str) -> None:
        """Save evaluation results to file.

        Args:
            results: Evaluation results
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_file}")

    def print_summary(self, results: Dict) -> None:
        """Print evaluation summary.

        Args:
            results: Evaluation results
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nOverall Average Score: {results['avg_score']:.2%}\n")

        print("Category Breakdown:")
        print("-" * 80)
        for category, data in results['categories'].items():
            print(f"  {category.upper()}: {data['avg_score']:.2%} ({data['count']} examples)")

        print("\n" + "=" * 80)
        print("Sample Outputs:")
        print("=" * 80)

        # Show first 3 examples
        for i in range(min(3, len(results['prompts']))):
            print(f"\nPrompt {i+1}: {results['prompts'][i]}")
            print(f"Response: {results['responses'][i][:200]}...")
            print(f"Score: {results['scores'][i]:.2%}")
            print("-" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Romanian Llama model"
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint name to evaluate (None = base model)'
    )

    parser.add_argument(
        '--test-file',
        type=str,
        default=None,
        help='JSONL file with test prompts'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation.json',
        help='Output file for results'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize evaluator
    try:
        evaluator = RomanianModelEvaluator(args.checkpoint)
        evaluator.setup_model()
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {str(e)}")
        return

    # Interactive mode
    if args.interactive:
        evaluator.interactive_mode()
        return

    # Evaluation mode
    if args.test_file:
        results = evaluator.evaluate_on_file(args.test_file)
    else:
        results = evaluator.evaluate_test_set()

    # Print and save results
    evaluator.print_summary(results)
    evaluator.save_results(results, args.output)

    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()

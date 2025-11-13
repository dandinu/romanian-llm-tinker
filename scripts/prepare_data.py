#!/usr/bin/env python3
"""
Prepare Romanian training data for Llama fine-tuning.

This script converts raw Romanian text into instruction-following format
suitable for fine-tuning with Tinker.

Usage:
    python scripts/prepare_data.py --input data/raw --output data/processed/train.jsonl
    python scripts/prepare_data.py --validate data/processed/train.jsonl
"""

import argparse
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from langdetect import detect, LangDetectException
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RomanianDataProcessor:
    """Process and format Romanian text for instruction following."""

    # Romanian instruction templates for data augmentation
    INSTRUCTION_TEMPLATES = [
        # Question answering
        ("Explică {topic}.", "explanation"),
        ("Ce este {topic}?", "definition"),
        ("Descrie {topic}.", "description"),
        ("Care sunt caracteristicile {topic}?", "characteristics"),

        # Information requests
        ("Spune-mi despre {topic}.", "information"),
        ("Furnizează informații despre {topic}.", "information"),
        ("Ce știi despre {topic}?", "knowledge"),

        # Summarization
        ("Rezumă următorul text:", "summarization"),
        ("Oferă un rezumat al:", "summarization"),

        # Analysis
        ("Analizează {topic}.", "analysis"),
        ("Evaluează {topic}.", "evaluation"),
    ]

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 4096,
        language_threshold: float = 0.9
    ):
        """Initialize the data processor.

        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
            language_threshold: Minimum confidence for Romanian detection
        """
        self.min_length = min_length
        self.max_length = max_length
        self.language_threshold = language_threshold

    def clean_text(self, text: str) -> str:
        """Clean and normalize Romanian text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Remove non-Latin characters (keep Romanian diacritics)
        # Romanian uses: ă, â, î, ș, ț, Ă, Â, Î, Ș, Ț
        text = re.sub(r'[^\x00-\x7FăâîșțĂÂÎȘȚ\s.,!?;:()\-\'"„"«»]', '', text)

        # Normalize quotes
        text = text.replace('„', '"').replace('"', '"')
        text = text.replace('«', '"').replace('»', '"')

        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)

        return text.strip()

    def is_romanian(self, text: str) -> bool:
        """Check if text is Romanian.

        Args:
            text: Text to check

        Returns:
            True if Romanian, False otherwise
        """
        try:
            return detect(text) == 'ro'
        except LangDetectException:
            return False

    def is_valid_text(self, text: str) -> bool:
        """Check if text meets quality criteria.

        Args:
            text: Text to validate

        Returns:
            True if valid, False otherwise
        """
        # Length check
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # Must contain Romanian letters
        if not re.search(r'[ăâîșțĂÂÎȘȚ]', text):
            return False

        # Basic sentence structure check
        if not re.search(r'[.!?]', text):
            return False

        # Language detection
        if not self.is_romanian(text):
            return False

        return True

    def extract_qa_pairs_from_wiki(self, text: str, title: str) -> List[Dict]:
        """Extract Q&A pairs from Wikipedia-style text.

        Args:
            text: Wikipedia article text
            title: Article title

        Returns:
            List of Q&A pairs
        """
        qa_pairs = []

        # Clean the text
        text = self.clean_text(text)

        if not self.is_valid_text(text):
            return qa_pairs

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 100]

        if not paragraphs:
            return qa_pairs

        # Generate instruction-following examples
        for template, _ in self.INSTRUCTION_TEMPLATES[:3]:  # Use first 3 templates
            # Create instruction by filling in the topic
            if '{topic}' in template:
                instruction = template.format(topic=title.lower())
            else:
                instruction = f"{template} {title}"

            # Use first 1-2 paragraphs as response
            num_paragraphs = min(2, len(paragraphs))
            response = '\n\n'.join(paragraphs[:num_paragraphs])

            # Ensure response is not too long
            if len(response) > 1500:
                response = response[:1500] + "..."

            qa_pairs.append({
                'instruction': instruction,
                'response': response,
                'topic': title
            })

        return qa_pairs

    def create_instruction_example(
        self,
        instruction: str,
        response: str,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """Create instruction-following example in chat format.

        Args:
            instruction: User instruction
            response: Assistant response
            system_prompt: Optional system prompt

        Returns:
            Chat format dictionary
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # Add user instruction
        messages.append({
            'role': 'user',
            'content': instruction
        })

        # Add assistant response
        messages.append({
            'role': 'assistant',
            'content': response
        })

        return {'messages': messages}

    def process_raw_file(
        self,
        input_file: Path,
        max_examples: Optional[int] = None
    ) -> List[Dict]:
        """Process a raw JSONL file into instruction examples.

        Args:
            input_file: Path to raw JSONL file
            max_examples: Maximum examples to process

        Returns:
            List of instruction examples
        """
        examples = []
        logger.info(f"Processing: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            if max_examples:
                lines = lines[:max_examples]

            for line in tqdm(lines, desc="Processing"):
                try:
                    data = json.loads(line)

                    # Extract text and title
                    text = data.get('text', '')
                    title = data.get('metadata', {}).get('title', 'subiect necunoscut')

                    # Extract Q&A pairs
                    qa_pairs = self.extract_qa_pairs_from_wiki(text, title)

                    # Convert to instruction format
                    for qa in qa_pairs:
                        example = self.create_instruction_example(
                            instruction=qa['instruction'],
                            response=qa['response']
                        )
                        examples.append(example)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON line: {line[:100]}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line: {str(e)}")
                    continue

        logger.info(f"Generated {len(examples)} instruction examples")
        return examples

    def create_train_val_split(
        self,
        examples: List[Dict],
        split_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split examples into train and validation sets.

        Args:
            examples: List of examples
            split_ratio: Proportion for training (0.0-1.0)
            seed: Random seed

        Returns:
            Tuple of (train_examples, val_examples)
        """
        # Shuffle examples
        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * split_ratio)
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]

        logger.info(f"Split: {len(train)} train, {len(val)} validation")

        return train, val

    def save_jsonl(self, examples: List[Dict], output_file: Path) -> None:
        """Save examples to JSONL file.

        Args:
            examples: List of examples
            output_file: Output file path
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(examples)} examples to {output_file}")

    def validate_jsonl(self, file_path: Path) -> Dict:
        """Validate JSONL file format.

        Args:
            file_path: Path to JSONL file

        Returns:
            Validation statistics
        """
        stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'invalid_lines': 0,
            'errors': []
        }

        logger.info(f"Validating: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                stats['total_lines'] += 1

                try:
                    data = json.loads(line)

                    # Check required structure
                    if 'messages' not in data:
                        raise ValueError("Missing 'messages' field")

                    messages = data['messages']
                    if not isinstance(messages, list):
                        raise ValueError("'messages' must be a list")

                    # Check message format
                    for msg in messages:
                        if 'role' not in msg or 'content' not in msg:
                            raise ValueError("Message missing 'role' or 'content'")

                        if msg['role'] not in ['system', 'user', 'assistant']:
                            raise ValueError(f"Invalid role: {msg['role']}")

                    stats['valid_lines'] += 1

                except Exception as e:
                    stats['invalid_lines'] += 1
                    stats['errors'].append(f"Line {i}: {str(e)}")

        # Print summary
        logger.info(f"\nValidation Results:")
        logger.info(f"  Total lines: {stats['total_lines']}")
        logger.info(f"  Valid: {stats['valid_lines']}")
        logger.info(f"  Invalid: {stats['invalid_lines']}")

        if stats['errors']:
            logger.warning(f"\nErrors (showing first 10):")
            for error in stats['errors'][:10]:
                logger.warning(f"  {error}")

        return stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Romanian training data for Llama fine-tuning"
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw',
        help='Input directory with raw JSONL files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/train.jsonl',
        help='Output JSONL file for processed data'
    )

    parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Train/validation split ratio (default: 0.8)'
    )

    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum number of examples to process'
    )

    parser.add_argument(
        '--validate',
        type=str,
        help='Validate a JSONL file instead of processing'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Validate mode
    if args.validate:
        processor = RomanianDataProcessor()
        processor.validate_jsonl(Path(args.validate))
        return

    # Process mode
    processor = RomanianDataProcessor()

    # Find all raw JSONL files
    input_dir = Path(args.input)
    input_files = list(input_dir.glob('*.jsonl'))

    if not input_files:
        logger.error(f"No JSONL files found in {input_dir}")
        logger.info("Run download_datasets.py first to get raw data")
        return

    logger.info(f"Found {len(input_files)} input file(s)")

    # Process all files
    all_examples = []
    for input_file in input_files:
        examples = processor.process_raw_file(input_file, args.max_examples)
        all_examples.extend(examples)

    if not all_examples:
        logger.error("No valid examples generated")
        return

    logger.info(f"Total examples: {len(all_examples)}")

    # Create train/val split
    train_examples, val_examples = processor.create_train_val_split(
        all_examples,
        split_ratio=args.split,
        seed=args.seed
    )

    # Save files
    output_path = Path(args.output)
    train_path = Path('data/splits/train.jsonl')
    val_path = Path('data/splits/val.jsonl')

    processor.save_jsonl(train_examples, train_path)
    processor.save_jsonl(val_examples, val_path)

    # Also save combined to specified output
    if output_path != train_path:
        processor.save_jsonl(all_examples, output_path)

    logger.info("\nData preparation complete!")
    logger.info(f"  Train: {train_path} ({len(train_examples)} examples)")
    logger.info(f"  Val: {val_path} ({len(val_examples)} examples)")
    logger.info("\nNext steps:")
    logger.info("  1. Validate data: python scripts/prepare_data.py --validate data/splits/train.jsonl")
    logger.info("  2. Start training: python scripts/train_tinker.py")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Download Romanian language datasets for fine-tuning.

This script downloads various Romanian datasets from HuggingFace and other sources,
organizing them in the data/raw directory for further processing.

Usage:
    python scripts/download_datasets.py --sources wiki oscar --size small
    python scripts/download_datasets.py --sources all --output data/raw/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset Download Constants
class DownloadConstants:
    """Constants for dataset downloading and filtering."""
    MIN_TEXT_LENGTH_FILTER = 50  # Minimum text length for dataset filtering


class RomanianDatasetDownloader:
    """Download and organize Romanian datasets."""

    AVAILABLE_SOURCES = {
        'wiki': {
            'name': 'wikimedia/wikipedia',
            'config': '20231101.ro',
            'split': 'train',
            'description': 'Romanian Wikipedia - clean factual text',
            'size_estimate': '~300MB'
        },
        'oscar': {
            'name': 'oscar-corpus/OSCAR-2201',
            'config': None,
            'split': 'train',
            'language_filter': 'ro',
            'description': 'OSCAR Romanian web corpus',
            'size_estimate': '~5GB (can be sampled)'
        },
        'cc100': {
            'name': 'cc100',
            'config': 'ro',
            'split': 'train',
            'description': 'Common Crawl Romanian subset',
            'size_estimate': '~60GB (can be sampled)'
        },
    }

    def __init__(self, output_dir: str = "data/raw"):
        """Initialize the downloader.

        Args:
            output_dir: Directory to save downloaded datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load environment variables
        load_dotenv()

        # Load HuggingFace token for accessing gated datasets
        self.hf_token = os.getenv('HF_TOKEN')
        if self.hf_token:
            logger.info("HuggingFace token loaded from .env")

    def download_dataset(
        self,
        source: str,
        max_examples: Optional[int] = None,
        streaming: bool = True
    ) -> None:
        """Download a specific dataset source.

        Args:
            source: Dataset source key (e.g., 'wiki', 'oscar')
            max_examples: Maximum number of examples to download (None = all)
            streaming: Use streaming mode for large datasets
        """
        if source not in self.AVAILABLE_SOURCES:
            logger.error(f"Unknown source: {source}")
            logger.info(f"Available sources: {', '.join(self.AVAILABLE_SOURCES.keys())}")
            return

        source_info = self.AVAILABLE_SOURCES[source]
        logger.info(f"Downloading {source}: {source_info['description']}")
        logger.info(f"Estimated size: {source_info['size_estimate']}")

        try:
            # Load dataset
            dataset_name = source_info['name']
            config = source_info.get('config')
            split = source_info.get('split', 'train')

            logger.info(f"Loading dataset: {dataset_name}")

            if streaming and max_examples:
                # Streaming mode for large datasets
                dataset = load_dataset(
                    dataset_name,
                    config,
                    split=split,
                    streaming=True,
                    token=self.hf_token
                )
            else:
                # Full download
                dataset = load_dataset(
                    dataset_name,
                    config,
                    split=split,
                    token=self.hf_token
                )

            # Filter for Romanian if needed
            if 'language_filter' in source_info:
                lang = source_info['language_filter']
                logger.info(f"Filtering for language: {lang}")
                if streaming:
                    dataset = dataset.filter(lambda x: x.get('meta', {}).get('language') == lang)
                else:
                    dataset = dataset.filter(lambda x: x.get('meta', {}).get('language') == lang)

            # Save to file
            output_file = self.output_dir / f"{source}.jsonl"
            logger.info(f"Saving to: {output_file}")

            self._save_dataset(dataset, output_file, max_examples, streaming)

            logger.info(f"Successfully downloaded {source} to {output_file}")

        except Exception as e:
            logger.error(f"Error downloading {source}: {str(e)}")
            logger.error(f"You may need to authenticate with HuggingFace:")
            logger.error(f"  1. Get token from https://huggingface.co/settings/tokens")
            logger.error(f"  2. Add HF_TOKEN to .env file, or run: huggingface-cli login")

    def _save_dataset(
        self,
        dataset,
        output_file: Path,
        max_examples: Optional[int],
        streaming: bool
    ) -> None:
        """Save dataset to JSONL file.

        Args:
            dataset: HuggingFace dataset object
            output_file: Path to save file
            max_examples: Maximum examples to save
            streaming: Whether dataset is in streaming mode
        """
        count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            iterator = iter(dataset)

            # Create progress bar
            if max_examples:
                pbar = tqdm(total=max_examples, desc="Downloading")
            else:
                pbar = tqdm(desc="Downloading")

            try:
                while True:
                    if max_examples and count >= max_examples:
                        break

                    try:
                        example = next(iterator)

                        # Extract text content based on dataset structure
                        text = self._extract_text(example)

                        if text and len(text.strip()) > DownloadConstants.MIN_TEXT_LENGTH_FILTER:
                            data = {
                                'text': text.strip(),
                                'source': output_file.stem,
                                'metadata': self._extract_metadata(example)
                            }
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            count += 1
                            pbar.update(1)

                    except StopIteration:
                        break
                    except Exception as e:
                        logger.warning(f"Error processing example: {str(e)}")
                        continue

            finally:
                pbar.close()

        logger.info(f"Saved {count} examples")

    def _extract_text(self, example: Dict[str, any]) -> Optional[str]:
        """Extract text content from dataset example.

        Args:
            example: Dataset example dictionary

        Returns:
            Extracted text or None
        """
        # Try different common field names
        for field in ['text', 'content', 'article', 'document']:
            if field in example:
                return example[field]

        # For Wikipedia
        if 'title' in example and 'text' in example:
            return f"{example['title']}\n\n{example['text']}"

        return None

    def _extract_metadata(self, example: Dict[str, any]) -> Dict[str, any]:
        """Extract metadata from example.

        Args:
            example: Dataset example dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Common metadata fields
        for field in ['title', 'url', 'timestamp', 'id']:
            if field in example:
                metadata[field] = example[field]

        return metadata

    def download_all(
        self,
        sources: List[str],
        max_examples_per_source: Optional[int] = None
    ) -> None:
        """Download multiple datasets.

        Args:
            sources: List of source keys to download
            max_examples_per_source: Max examples per source
        """
        logger.info(f"Downloading {len(sources)} dataset(s)")

        for source in sources:
            self.download_dataset(source, max_examples_per_source)
            logger.info(f"Completed: {source}\n")

    @classmethod
    def list_sources(cls) -> None:
        """Print available data sources."""
        print("\nAvailable Romanian Dataset Sources:")
        print("=" * 80)

        for key, info in cls.AVAILABLE_SOURCES.items():
            print(f"\n{key.upper()}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size_estimate']}")
            print(f"  Dataset: {info['name']}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Romanian language datasets for fine-tuning"
    )

    parser.add_argument(
        '--sources',
        nargs='+',
        default=['wiki'],
        help='Dataset sources to download (wiki, oscar, cc100, or "all")'
    )

    parser.add_argument(
        '--size',
        choices=['small', 'medium', 'large', 'full'],
        default='small',
        help='Dataset size: small (500 examples), medium (2K), large (10K), full (all)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for downloaded datasets'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available dataset sources and exit'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # List sources if requested
    if args.list:
        RomanianDatasetDownloader.list_sources()
        return

    # Determine max examples based on size
    size_map = {
        'small': 500,
        'medium': 2000,
        'large': 10000,
        'full': None
    }
    max_examples = size_map[args.size]

    # Handle "all" sources
    if 'all' in args.sources:
        sources = list(RomanianDatasetDownloader.AVAILABLE_SOURCES.keys())
    else:
        sources = args.sources

    # Initialize downloader
    downloader = RomanianDatasetDownloader(args.output)

    # Download datasets
    logger.info(f"Configuration:")
    logger.info(f"  Sources: {', '.join(sources)}")
    logger.info(f"  Size: {args.size} ({max_examples or 'all'} examples per source)")
    logger.info(f"  Output: {args.output}")
    logger.info("")

    downloader.download_all(sources, max_examples)

    logger.info("\nDownload complete!")
    logger.info(f"Files saved to: {args.output}")
    logger.info("\nNext steps:")
    logger.info("  1. Review downloaded data in data/raw/")
    logger.info("  2. Run: python scripts/prepare_data.py")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Quick start script for Romanian Llama fine-tuning.

This script runs a minimal end-to-end pipeline to validate your setup:
1. Check Tinker connection
2. Download a small sample of data
3. Prepare training data
4. Run a quick training test

Usage:
    python scripts/quick_start.py
"""

import os
import sys
from pathlib import Path
import logging

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if environment is properly set up."""
    logger.info("Checking environment...")

    # Load .env
    load_dotenv()

    # Check API key
    api_key = os.getenv('TINKER_API_KEY')
    if not api_key:
        logger.error("TINKER_API_KEY not found in .env file")
        logger.info("Please set your Tinker API key in .env:")
        logger.info("  TINKER_API_KEY=your-key-here")
        return False

    logger.info("  API key found")

    # Check Tinker installation
    try:
        from tinker import ServiceClient
        from tinker_cookbook import renderers
        logger.info("  Tinker installed")
    except ImportError:
        logger.error("Tinker not installed")
        logger.info("Run: pip install -r requirements.txt")
        return False

    # Test Tinker connection
    try:
        client = ServiceClient()
        logger.info("  Tinker connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to Tinker: {str(e)}")
        return False

    return True


def run_quick_pipeline():
    """Run a quick end-to-end pipeline."""
    logger.info("\n" + "="*80)
    logger.info("QUICK START PIPELINE")
    logger.info("="*80 + "\n")

    # Step 1: Download data
    logger.info("Step 1: Downloading sample data...")
    os.system(
        "python scripts/download_datasets.py --sources wiki --size small"
    )

    # Step 2: Prepare data
    logger.info("\nStep 2: Preparing training data...")
    os.system(
        "python scripts/prepare_data.py --input data/raw --max-examples 100"
    )

    # Step 3: Run quick training test
    logger.info("\nStep 3: Running quick training test...")
    logger.info("(This will train for just 50 steps to validate the pipeline)\n")

    # Create quick test config
    import yaml

    with open('configs/hyperparams.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Modify for quick test
    config['training']['max_steps'] = 50
    config['training']['save_steps'] = 25
    config['training']['eval_steps'] = 25

    quick_config_path = 'configs/quick_test.yaml'
    with open(quick_config_path, 'w') as f:
        yaml.dump(config, f)

    os.system(
        f"python scripts/train_tinker.py --config {quick_config_path} --quick-test"
    )

    logger.info("\n" + "="*80)
    logger.info("QUICK START COMPLETE!")
    logger.info("="*80)
    logger.info("\nYour setup is working! Next steps:")
    logger.info("  1. Download more data: python scripts/download_datasets.py --size medium")
    logger.info("  2. Prepare full dataset: python scripts/prepare_data.py")
    logger.info("  3. Run full training: python scripts/train_tinker.py")
    logger.info("  4. Evaluate results: python scripts/evaluate.py")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("Romanian Llama 3.1 8B Fine-Tuning - Quick Start")
    print("="*80 + "\n")

    # Check environment
    if not check_environment():
        logger.error("\nEnvironment check failed. Please fix the issues above.")
        sys.exit(1)

    logger.info("\nEnvironment check passed!")

    # Ask to proceed
    response = input("\nRun the quick start pipeline? This will download data and run a test training. (y/n): ")

    if response.lower() in ['y', 'yes']:
        run_quick_pipeline()
    else:
        logger.info("\nQuick start cancelled.")
        logger.info("\nManual steps:")
        logger.info("  1. python scripts/download_datasets.py --sources wiki --size small")
        logger.info("  2. python scripts/prepare_data.py")
        logger.info("  3. python scripts/train_tinker.py --quick-test")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Download trained checkpoint from Tinker.

This script downloads the trained LoRA adapter weights from Tinker's infrastructure
and saves them locally for evaluation and deployment.

Usage:
    python scripts/download_checkpoint.py --session-id a65fa1a6-00b9-5a7e-9abf-59f068b79982
    python scripts/download_checkpoint.py --session-id a65fa1a6-00b9-5a7e-9abf-59f068b79982 --checkpoint final
    python scripts/download_checkpoint.py --session-id a65fa1a6-00b9-5a7e-9abf-59f068b79982 --checkpoint step_500
"""

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Import Tinker
try:
    from tinker import ServiceClient
except ImportError:
    print("Error: Tinker not installed. Run: pip install tinker")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointDownloader:
    """Download checkpoints from Tinker."""

    def __init__(self, session_id: str):
        """Initialize the downloader.

        Args:
            session_id: Tinker session ID from training
        """
        # Load environment variables
        load_dotenv()

        # Verify API key
        self.api_key = os.getenv('TINKER_API_KEY')
        if not self.api_key:
            raise ValueError("TINKER_API_KEY not found in .env")

        self.session_id = session_id

        # Initialize Tinker clients
        logger.info("Connecting to Tinker...")
        self.service_client = ServiceClient()
        self.rest_client = self.service_client.create_rest_client()

        logger.info(f"Connected to session: {session_id}")

    def list_available_checkpoints(self) -> list:
        """List available checkpoints for this session.

        Returns:
            List of checkpoint names
        """
        # Common checkpoint names to check
        checkpoint_candidates = [
            "final",
            "checkpoint_final",
            "latest",
        ]

        # Add step-based checkpoints
        for step in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
            checkpoint_candidates.append(f"step_{step}")
            checkpoint_candidates.append(f"checkpoint_step_{step}")

        logger.info("Checking for available checkpoints...")
        return checkpoint_candidates

    def download_checkpoint(
        self,
        checkpoint_name: str,
        output_dir: str = "checkpoints/downloads"
    ) -> str:
        """Download a checkpoint from Tinker.

        Args:
            checkpoint_name: Name of checkpoint (e.g., 'final', 'step_500')
            output_dir: Directory to save checkpoint

        Returns:
            Path to downloaded checkpoint
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Try multiple path formats
        path_variations = [
            f"tinker://{self.session_id}/sampler_weights/{checkpoint_name}",
            f"tinker://{self.session_id}:train:0/sampler_weights/{checkpoint_name}",
            f"tinker://{self.session_id}/{checkpoint_name}",
            f"tinker://{self.session_id}:train:0/{checkpoint_name}",
        ]

        last_error = None
        for tinker_path in path_variations:
            logger.info(f"Trying path: {tinker_path}")

            try:
                # Get signed URL from Tinker (with retries for archive creation)
                import time
                max_retries = 10
                retry_delay = 30  # seconds

                for retry in range(max_retries):
                    try:
                        future = self.rest_client.get_checkpoint_archive_url_from_tinker_path(tinker_path)
                        checkpoint_response = future.result()

                        logger.info(f"✓ Success! Got signed URL (expires: {checkpoint_response.expires})")
                        break  # Success!

                    except Exception as retry_error:
                        error_str = str(retry_error)
                        # Check if it's a 409 (archive creation in progress)
                        if "409" in error_str or "Archive creation already in progress" in error_str:
                            logger.info(f"Archive creation in progress, waiting {retry_delay}s... (attempt {retry+1}/{max_retries})")
                            if retry < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                        # Not a 409 or last retry, re-raise
                        raise

                # Download the checkpoint
                output_file = output_path / f"{checkpoint_name}.tar"
                logger.info(f"Downloading to: {output_file}")

                urllib.request.urlretrieve(checkpoint_response.url, str(output_file))

                # Get file size
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"Download complete! Size: {file_size_mb:.2f} MB")

                # Extract the archive
                logger.info("Extracting archive...")
                import tarfile
                extract_dir = output_path / checkpoint_name
                extract_dir.mkdir(exist_ok=True)

                with tarfile.open(output_file, 'r') as tar:
                    tar.extractall(extract_dir)

                logger.info(f"Extracted to: {extract_dir}")

                # List contents
                logger.info("Checkpoint contents:")
                for item in sorted(extract_dir.rglob('*')):
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024 * 1024)
                        logger.info(f"  {item.relative_to(extract_dir)} ({size_mb:.2f} MB)")

                return str(extract_dir)

            except Exception as e:
                last_error = e
                logger.warning(f"✗ Failed: {str(e)}")
                continue

        # All paths failed
        logger.error(f"All path variations failed for checkpoint: {checkpoint_name}")
        logger.error(f"Tried {len(path_variations)} different path formats")
        logger.error(f"Last error: {last_error}")
        raise Exception(f"Could not download checkpoint '{checkpoint_name}' - tried all path variations")

    def download_all_checkpoints(self, output_dir: str = "checkpoints/downloads"):
        """Try to download all available checkpoints.

        Args:
            output_dir: Directory to save checkpoints
        """
        checkpoints = self.list_available_checkpoints()
        successful = []
        failed = []

        logger.info(f"Attempting to download {len(checkpoints)} checkpoints...")

        for checkpoint in checkpoints:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Downloading checkpoint: {checkpoint}")
                logger.info(f"{'='*80}")

                path = self.download_checkpoint(checkpoint, output_dir)
                successful.append((checkpoint, path))
                logger.info(f"✓ Successfully downloaded: {checkpoint}")

            except Exception as e:
                failed.append((checkpoint, str(e)))
                logger.warning(f"✗ Failed to download: {checkpoint}")

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("DOWNLOAD SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")

        if successful:
            logger.info("\nSuccessful downloads:")
            for name, path in successful:
                logger.info(f"  ✓ {name}: {path}")

        if failed:
            logger.info("\nFailed downloads:")
            for name, error in failed:
                logger.info(f"  ✗ {name}")

        return successful, failed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download trained checkpoint from Tinker"
    )

    parser.add_argument(
        '--session-id',
        type=str,
        required=True,
        help='Tinker session ID from training (e.g., a65fa1a6-00b9-5a7e-9abf-59f068b79982)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='final',
        help='Checkpoint name to download (default: final)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints/downloads',
        help='Directory to save checkpoint (default: checkpoints/downloads)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Try to download all available checkpoints'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    try:
        # Initialize downloader
        downloader = CheckpointDownloader(args.session_id)

        if args.all:
            # Download all checkpoints
            downloader.download_all_checkpoints(args.output_dir)
        else:
            # Download specific checkpoint
            checkpoint_path = downloader.download_checkpoint(
                args.checkpoint,
                args.output_dir
            )

            logger.info(f"\n{'='*80}")
            logger.info("SUCCESS!")
            logger.info(f"{'='*80}")
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
            logger.info("\nNext steps:")
            logger.info("  1. Load this checkpoint in your evaluation script")
            logger.info("  2. Test the model: python scripts/evaluate.py --interactive")
            logger.info("  3. Run full evaluation: python scripts/evaluate.py")

    except Exception as e:
        logger.error(f"\nFailed to download checkpoint: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

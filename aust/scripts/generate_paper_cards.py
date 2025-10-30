"""Batch processing script for generating paper cards from all PDFs.

This script processes all PDFs in the .papers/ directory and generates
structured paper cards for RAG retrieval.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TEXT_CONFIG = (
    PROJECT_ROOT / "aust" / "configs" / "prompts" / "paper_card_extraction.yaml"
)
DEFAULT_PDF_CONFIG = (
    PROJECT_ROOT / "aust" / "configs" / "prompts" / "paper_card_pdf_extraction.yaml"
)

from aust.src.agents.paper_card_agent import PaperCardAgent
from aust.src.agents.paper_card_pdf_agent import PaperCardPdfAgent
from aust.src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(
    log_level="INFO", log_dir=Path("./logs"), enable_console=True, enable_file=True
)
logger = get_logger(__name__)


class PaperCardBatchProcessor:
    """Batch processor for generating paper cards."""

    def __init__(
        self,
        papers_dir: str = ".papers/",
        cards_dir: str = ".paper_cards/",
        metadata_file: str = ".papers/paper_metadata.json",
        config_path: str = str(DEFAULT_TEXT_CONFIG),
        pdf_config_path: str = str(DEFAULT_PDF_CONFIG),
        model: str = "openai/gpt-5-nano",
        input_mode: str = "text",
    ):
        """Initialize batch processor.

        Args:
            papers_dir: Directory containing PDF files
            cards_dir: Output directory for paper cards
            metadata_file: Path to paper metadata JSON
            config_path: Path to agent config YAML
            model: LLM model to use
        """
        self.papers_dir = Path(papers_dir)
        self.cards_dir = Path(cards_dir)
        self.metadata_file = Path(metadata_file)
        self.config_path = config_path
        self.pdf_config_path = pdf_config_path
        self.model = model
        self.input_mode = input_mode

        # Progress tracking files
        self.progress_file = self.cards_dir / ".generation_progress.json"
        self.failed_file = self.cards_dir / ".failed_extractions.json"
        self.card_metadata_file = self.cards_dir / "card_metadata.json"

        # Initialize agent
        if self.input_mode == "pdf":
            logger.info(
                "Using PDF agent with config %s and model %s",
                self.pdf_config_path,
                self.model,
            )
            self.agent = PaperCardPdfAgent(
                config_path=self.pdf_config_path,
                model=model,
            )
        else:
            logger.info(
                "Using text agent with config %s and model %s",
                self.config_path,
                self.model,
            )
            self.agent = PaperCardAgent(config_path=config_path, model=model)

        logger.info(
            "Batch processor initialized - papers: %s, output: %s, mode: %s",
            papers_dir,
            cards_dir,
            self.input_mode,
        )

    def load_progress(self) -> Dict:
        """Load progress tracking data."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
            logger.info(
                f"Loaded progress: {len(progress['processed'])} processed, {len(progress['failed'])} failed"
            )
        else:
            progress = {"processed": [], "failed": [], "total": 0, "completed": 0}
            logger.info("No existing progress file, starting fresh")

        return progress

    def save_progress(self, progress: Dict):
        """Save progress tracking data."""
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(progress, indent=2, fp=f)

    def load_failed_extractions(self) -> List[Dict]:
        """Load failed extraction logs."""
        if self.failed_file.exists():
            with open(self.failed_file, "r") as f:
                return json.load(f)
        return []

    def save_failed_extraction(self, failed_entry: Dict):
        """Append failed extraction to log file."""
        failed_extractions = self.load_failed_extractions()
        failed_extractions.append(failed_entry)

        self.cards_dir.mkdir(parents=True, exist_ok=True)
        with open(self.failed_file, "w") as f:
            json.dump(failed_extractions, indent=2, fp=f)

    def load_card_metadata(self) -> List[Dict]:
        """Load card metadata."""
        if self.card_metadata_file.exists():
            with open(self.card_metadata_file, "r") as f:
                return json.load(f)
        return []

    def save_card_metadata(self, card_metadata: List[Dict]):
        """Save card metadata."""
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        with open(self.card_metadata_file, "w") as f:
            json.dump(card_metadata, indent=2, fp=f)

    def update_card_metadata(self, metadata_entry: Dict):
        """Append or update card metadata entry."""
        card_metadata = self.load_card_metadata()

        # Check if entry exists (by arxiv_id)
        existing_idx = None
        for idx, entry in enumerate(card_metadata):
            if entry["arxiv_id"] == metadata_entry["arxiv_id"]:
                existing_idx = idx
                break

        if existing_idx is not None:
            # Update existing entry
            card_metadata[existing_idx] = metadata_entry
            logger.debug(f"Updated metadata for {metadata_entry['arxiv_id']}")
        else:
            # Append new entry
            card_metadata.append(metadata_entry)
            logger.debug(f"Added metadata for {metadata_entry['arxiv_id']}")

        self.save_card_metadata(card_metadata)

    def load_paper_metadata(self) -> List[Dict]:
        """Load paper metadata from JSON file."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Paper metadata file not found: {self.metadata_file}")

        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loaded metadata for {len(metadata)} papers")
        return metadata

    def process_paper(
        self, paper_metadata: Dict, progress: Dict, rate_limit_delay: float = 4.0
    ) -> bool:
        """Process a single paper.

        Args:
            paper_metadata: Paper metadata dict
            progress: Progress tracking dict
            rate_limit_delay: Delay between API calls (seconds)

        Returns:
            True if successful, False if failed
        """
        arxiv_id = paper_metadata["arxiv_id"]

        # Skip if already processed
        if arxiv_id in progress["processed"]:
            logger.debug(f"Skipping {arxiv_id} - already processed")
            return True

        pdf_path = self.papers_dir / paper_metadata["file_path"]

        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            self.save_failed_extraction(
                {
                    "arxiv_id": arxiv_id,
                    "pdf_path": str(pdf_path),
                    "error_type": "FileNotFoundError",
                    "error_message": "PDF file not found",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "retry_count": 0,
                }
            )
            return False

        # Attempt to generate card with retry logic
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Processing {arxiv_id} ({paper_metadata['title'][:50]}...)"
                )

                # Generate card
                card_path = self.agent.generate_card(
                    pdf_path=str(pdf_path),
                    metadata=paper_metadata,
                    output_dir=str(self.cards_dir),
                )

                # Validate quality
                quality_score = self.agent.validate_card_quality(card_path)

                # Update card metadata
                card_metadata_entry = {
                    "arxiv_id": arxiv_id,
                    "title": paper_metadata["title"],
                    "card_path": card_path,
                    "pdf_path": str(pdf_path),
                    "generation_date": datetime.now(timezone.utc).isoformat(),
                    "agent_model": self.model,
                    "processing_status": "success",
                    "extraction_quality_score": quality_score,
                }

                self.update_card_metadata(card_metadata_entry)

                # Update progress
                progress["processed"].append(arxiv_id)
                progress["completed"] += 1
                self.save_progress(progress)

                # Rate limiting
                time.sleep(rate_limit_delay)

                logger.info(
                    f"Successfully processed {arxiv_id} (quality: {quality_score:.2f})"
                )
                return True

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {arxiv_id}: {e}")

                if attempt < max_retries - 1:
                    # Retry with exponential backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    # All retries exhausted
                    logger.error(f"All retries exhausted for {arxiv_id}")

                    self.save_failed_extraction(
                        {
                            "arxiv_id": arxiv_id,
                            "pdf_path": str(pdf_path),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "retry_count": max_retries,
                        }
                    )

                    progress["failed"].append(arxiv_id)
                    self.save_progress(progress)

                    return False

        return False

    def process_batch(
        self,
        category_filter: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
    ):
        """Process all papers in batch.

        Args:
            category_filter: Only process papers in specific category (e.g., "any-to-t/input_level")
            arxiv_id: Process specific paper by arxiv_id
            force: Force re-processing of already processed papers
            dry_run: Preview papers to process without actually processing
        """
        # Load progress
        progress = self.load_progress()

        # Load paper metadata
        all_papers = self.load_paper_metadata()

        # Filter papers
        papers_to_process = []

        if arxiv_id:
            # Process specific paper
            papers_to_process = [p for p in all_papers if p["arxiv_id"] == arxiv_id]
            if not papers_to_process:
                logger.error(f"Paper not found: {arxiv_id}")
                return
            logger.info(f"Processing single paper: {arxiv_id}")

        elif category_filter:
            # Process specific category
            papers_to_process = [
                p for p in all_papers if p["file_path"].startswith(category_filter)
            ]
            logger.info(
                f"Processing category '{category_filter}': {len(papers_to_process)} papers"
            )

        else:
            # Process all papers
            papers_to_process = all_papers
            logger.info(f"Processing all papers: {len(papers_to_process)} total")

        # Apply force flag
        if force:
            logger.warning("Force flag set - will re-process already completed papers")
            progress["processed"] = []

        # Update total count
        progress["total"] = len(papers_to_process)

        # Dry run mode
        if dry_run:
            logger.info("DRY RUN MODE - No cards will be generated")
            print("\nPapers to process:")
            for paper in papers_to_process:
                status = "✓ processed" if paper["arxiv_id"] in progress["processed"] else "○ pending"
                print(f"  [{status}] {paper['arxiv_id']} - {paper['title'][:60]}...")
            print(f"\nTotal: {len(papers_to_process)} papers")
            return

        # Process papers
        logger.info(f"Starting batch processing of {len(papers_to_process)} papers...")

        successful = 0
        failed = 0

        for paper in tqdm(papers_to_process, desc="Generating paper cards"):
            success = self.process_paper(paper, progress)

            if success:
                successful += 1
            else:
                failed += 1

        # Final summary
        logger.info(f"Batch processing complete!")
        logger.info(f"Successful: {successful}, Failed: {failed}, Total: {len(papers_to_process)}")

        if failed > 0:
            logger.warning(
                f"Failed extractions logged to: {self.failed_file}"
            )

        print("\n" + "=" * 60)
        print(f"BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"Total: {len(papers_to_process)}")
        print(f"\nCard metadata: {self.card_metadata_file}")
        print(f"Progress file: {self.progress_file}")
        if failed > 0:
            print(f"Failed extractions: {self.failed_file}")
        print("=" * 60)


def main():
    """Main entry point for batch processing script."""
    parser = argparse.ArgumentParser(
        description="Generate paper cards for all PDFs in .papers/ directory"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process specific category (e.g., 'any-to-t/input_level')",
    )

    parser.add_argument(
        "--arxiv-id",
        type=str,
        default=None,
        help="Process specific paper by arxiv_id",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of already processed papers",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview papers to process without actually generating cards",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-nano",
        help="LLM model to use (default: openai/gpt-5-nano)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (default behavior, kept for clarity)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_TEXT_CONFIG),
        help="Path to the text-based agent prompt config",
    )

    parser.add_argument(
        "--pdf-config",
        type=str,
        default=str(DEFAULT_PDF_CONFIG),
        help="Path to the PDF agent prompt config",
    )

    parser.add_argument(
        "--input-mode",
        choices=["text", "pdf"],
        default="pdf",
        help="Whether to extract via pre-parsed text or direct PDF upload",
    )

    args = parser.parse_args()

    # Initialize processor
    processor = PaperCardBatchProcessor(
        papers_dir=".papers/",
        cards_dir=".paper_cards/",
        metadata_file=".papers/paper_metadata.json",
        config_path=args.config,
        pdf_config_path=args.pdf_config,
        model=args.model,
        input_mode=args.input_mode,
    )

    # Process batch
    processor.process_batch(
        category_filter=args.category,
        arxiv_id=args.arxiv_id,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

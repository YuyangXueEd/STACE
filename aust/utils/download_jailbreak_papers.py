"""
Jailbreak Paper Corpus Downloader

Downloads and categorizes jailbreak attack papers from Awesome-Multimodal-Jailbreak repository.
Papers are organized by model type (Any-to-T/Any-to-V) and attack level (Input/Encoder/Generator/Output).
"""

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ArxivPaperInfo:
    """Information about an arxiv paper."""
    title: str
    arxiv_id: str
    pdf_url: str
    venue: str
    year: int
    model_type: str  # I+T→T, T→I, V+T→T, etc.
    taxonomy: str  # Input Level, Encoder Level, Generator Level, Output Level, or ---


@dataclass
class PaperMetadata:
    """Metadata for downloaded paper."""
    title: str
    authors: List[str]
    venue: str
    year: int
    arxiv_id: str
    pdf_url: str
    model_type: str  # I+T→T, T→I, etc.
    model_type_category: str  # any-to-t or any-to-v
    attack_level: str  # input_level, encoder_level, generator_level, output_level, unknown
    taxonomy: str  # Original taxonomy from table
    file_path: str
    download_date: str
    abstract: Optional[str] = None


@dataclass
class DownloadProgress:
    """Progress tracking for paper downloads."""
    arxiv_id: str
    status: str  # pending/downloading/completed/failed
    last_attempt: str
    error_message: Optional[str] = None


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, max_requests_per_second: float = 3.0):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_second: Maximum requests per second (default: 3)
        """
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_interval:
            sleep_time = self.min_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class JailbreakPaperDownloader:
    """Downloads jailbreak papers from Awesome-Multimodal-Jailbreak repository."""

    # Repository URLs
    GITHUB_README_URL = "https://raw.githubusercontent.com/liuxuannan/Awesome-Multimodal-Jailbreak/main/README.md"
    # Use export.arxiv.org for direct PDF downloads (arxiv.org/pdf returns HTML redirects)
    ARXIV_PDF_BASE = "https://export.arxiv.org/pdf"

    # User agent for requests
    USER_AGENT = "CAUST-PaperDownloader/1.0 (Research; mailto:user@example.com)"

    # Request timeout
    TIMEOUT = 30

    # Retry configuration
    MAX_RETRIES = 5
    RETRY_DELAYS = [1, 2, 4, 8, 16]  # Exponential backoff

    def __init__(
        self,
        output_dir: Path,
        metadata_file: Path
    ):
        """
        Initialize downloader.

        Args:
            output_dir: Output directory for papers
            metadata_file: Path to metadata JSON file
        """
        self.output_dir = Path(output_dir)
        self.metadata_file = Path(metadata_file)
        self.progress_file = self.output_dir / ".download_progress.json"

        self.rate_limiter = RateLimiter(max_requests_per_second=3.0)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

        self.progress: Dict[str, DownloadProgress] = {}
        self.metadata: List[PaperMetadata] = []

        # Load existing data
        self._load_progress()
        self._load_metadata()

    def _load_progress(self) -> None:
        """Load download progress from JSON file."""
        if self.progress_file.exists():
            logger.info(f"Loading progress from {self.progress_file}")
            with open(self.progress_file, "r") as f:
                data = json.load(f)

            for item in data:
                self.progress[item["arxiv_id"]] = DownloadProgress(**item)

            logger.info(f"Loaded progress for {len(self.progress)} papers")

    def _save_progress(self) -> None:
        """Save download progress to JSON file."""
        data = [asdict(p) for p in self.progress.values()]

        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_metadata(self) -> None:
        """Load paper metadata from JSON file."""
        if self.metadata_file.exists():
            logger.info(f"Loading metadata from {self.metadata_file}")
            with open(self.metadata_file, "r") as f:
                data = json.load(f)

            self.metadata = [PaperMetadata(**item) for item in data]
            logger.info(f"Loaded metadata for {len(self.metadata)} papers")

    def _save_metadata(self) -> None:
        """Save paper metadata to JSON file."""
        data = [asdict(m) for m in self.metadata]

        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def _classify_model_type(self, model_type: str) -> str:
        """
        Classify model type into any-to-t or any-to-v based on output modality.

        Args:
            model_type: Model type from table (I+T→T, T→I, V+T→T, etc.)

        Returns:
            "any-to-t" or "any-to-v"
        """
        # Check if output is Text (→T)
        if "→T" in model_type or "->T" in model_type:
            return "any-to-t"

        # Check if output is Visual (→I or →V)
        if "→I" in model_type or "→V" in model_type or "->I" in model_type or "->V" in model_type:
            return "any-to-v"

        # Default fallback
        logger.warning(f"Unknown model type classification for: {model_type}, defaulting to any-to-t")
        return "any-to-t"

    def _normalize_attack_level(self, taxonomy: str) -> str:
        """
        Normalize taxonomy to attack level directory name.

        Args:
            taxonomy: Taxonomy from table (e.g., "Input Level", "Encoder Level", "---")

        Returns:
            Normalized attack level (e.g., "input_level", "encoder_level", "unknown")
        """
        taxonomy_lower = taxonomy.lower().strip()

        if "input" in taxonomy_lower:
            return "input_level"
        elif "encoder" in taxonomy_lower:
            return "encoder_level"
        elif "generator" in taxonomy_lower:
            return "generator_level"
        elif "output" in taxonomy_lower:
            return "output_level"
        else:
            # Handle "---" or unknown taxonomy
            return "unknown"

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """
        Extract arxiv ID from text.

        Args:
            text: Text containing arxiv ID or URL

        Returns:
            Arxiv ID or None
        """
        # Pattern 1: arxiv URL (https://arxiv.org/abs/2401.12345)
        url_pattern = r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)"
        match = re.search(url_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: arxiv ID directly (2401.12345)
        id_pattern = r"\b(\d{4}\.\d{4,5})\b"
        match = re.search(id_pattern, text)
        if match:
            return match.group(1)

        return None

    def _extract_venue_year(self, venue_text: str) -> Tuple[str, int]:
        """
        Extract venue and year from citation.

        Args:
            venue_text: Citation text like "[ICLR'24]"

        Returns:
            Tuple of (venue, year)
        """
        # Extract venue name (text before quote or year)
        venue_match = re.search(r"\[?([A-Za-z\s]+)", venue_text)
        venue = venue_match.group(1).strip() if venue_match else "Unknown"

        # Extract year
        year_match = re.search(r"'(\d{2})", venue_text)
        if year_match:
            year_short = int(year_match.group(1))
            # Convert 2-digit year to 4-digit (24 → 2024, 23 → 2023)
            year = 2000 + year_short
        else:
            year_match_full = re.search(r"(\d{4})", venue_text)
            year = int(year_match_full.group(1)) if year_match_full else 2024

        return venue, year

    def fetch_github_readme(self) -> str:
        """
        Fetch GitHub README content.

        Returns:
            README markdown content
        """
        logger.info(f"Fetching GitHub README from {self.GITHUB_README_URL}")

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.get(self.GITHUB_README_URL, timeout=self.TIMEOUT)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAYS[attempt])
                else:
                    raise

    def parse_jailbreak_attack_table(self, readme_content: str) -> List[ArxivPaperInfo]:
        """
        Parse jailbreak attack table from README.

        Args:
            readme_content: README markdown content

        Returns:
            List of ArxivPaperInfo objects
        """
        logger.info("Parsing jailbreak attack table")

        # Find the "😈JailBreak Attack" section
        lines = readme_content.split("\n")
        start_idx = None

        for i, line in enumerate(lines):
            if "😈JailBreak Attack" in line or "JailBreak Attack" in line:
                start_idx = i
                break

        if start_idx is None:
            logger.error("Could not find '😈JailBreak Attack' section")
            return []

        # Parse markdown tables - there are multiple subsections
        papers: List[ArxivPaperInfo] = []
        table_started = False
        current_section = None

        for i in range(start_idx, len(lines)):
            line = lines[i].strip()

            # Track which subsection we're in
            if line.startswith("##") and i > start_idx:
                if "Any-to-Text" in line:
                    current_section = "any-to-text"
                    table_started = False
                elif "Any-to-Vision" in line:
                    current_section = "any-to-vision"
                    table_started = False
                elif "Defense" in line or "Evaluation" in line:
                    # Stop when we reach defense or evaluation sections
                    break
                continue

            # Skip until table starts
            if not table_started:
                if line.startswith("|") and "Title" in line:
                    table_started = True
                    continue
                continue

            # Skip separator line
            if line.startswith("|") and all(c in "|-: " for c in line):
                continue

            # Parse table row
            if line.startswith("|"):
                cells = [cell.strip() for cell in line.split("|")[1:-1]]

                if len(cells) < 5:
                    continue

                # Table format: Title | Venue | Date | Code | Taxonomy | Multimodal Model
                title_cell = cells[0] if len(cells) > 0 else ""
                venue_cell = cells[1] if len(cells) > 1 else ""
                date_cell = cells[2] if len(cells) > 2 else ""
                code_cell = cells[3] if len(cells) > 3 else ""
                taxonomy_cell = cells[4] if len(cells) > 4 else "---"
                model_type_cell = cells[5] if len(cells) > 5 else "Unknown"

                # Extract title and arxiv URL from title cell
                # Title cell format: [**Title**](arxiv_url)
                title_match = re.search(r"\[([^\]]+)\]\(([^)]+)\)", title_cell)

                if title_match:
                    title = title_match.group(1).strip()
                    # Remove markdown bold markers
                    title = re.sub(r"\*\*", "", title)
                    url = title_match.group(2).strip()
                else:
                    logger.debug(f"Could not parse title cell: {title_cell}")
                    continue

                # Extract arxiv ID
                arxiv_id = self._extract_arxiv_id(url)
                if not arxiv_id:
                    logger.debug(f"Could not extract arxiv ID from: {url}")
                    continue

                # Extract venue and year
                venue, year = self._extract_venue_year(venue_cell)

                # Extract model type and taxonomy
                model_type = model_type_cell.strip() if model_type_cell else "Unknown"
                taxonomy = taxonomy_cell.strip() if taxonomy_cell else "---"

                # Create paper info
                # Note: ArXiv PDF URLs should NOT have .pdf extension (causes HTML redirect page)
                paper = ArxivPaperInfo(
                    title=title,
                    arxiv_id=arxiv_id,
                    pdf_url=f"{self.ARXIV_PDF_BASE}/{arxiv_id}",
                    venue=venue,
                    year=year,
                    model_type=model_type,
                    taxonomy=taxonomy
                )

                papers.append(paper)
                logger.debug(f"Parsed paper: {title[:50]}... ({arxiv_id}) - {model_type} - {taxonomy}")

        logger.info(f"Parsed {len(papers)} papers from table")
        return papers

    def _download_with_retry(self, url: str, output_path: Path) -> bool:
        """
        Download file with retry logic.

        Args:
            url: URL to download
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                self.rate_limiter.wait()

                response = self.session.get(url, timeout=self.TIMEOUT, stream=True)
                response.raise_for_status()

                # Save file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return True

            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"File not found (404): {url}")
                    return False
                elif e.response.status_code == 429:
                    # Rate limit hit
                    retry_after = int(e.response.headers.get("Retry-After", self.RETRY_DELAYS[attempt]))
                    logger.warning(f"Rate limit hit (429), waiting {retry_after}s")
                    time.sleep(retry_after)
                elif e.response.status_code >= 500:
                    logger.warning(f"Server error ({e.response.status_code}), attempt {attempt + 1}/{self.MAX_RETRIES}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAYS[attempt])
                else:
                    logger.error(f"HTTP error: {e}")
                    return False

            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAYS[attempt])

        return False

    def download_paper(self, paper: ArxivPaperInfo) -> Optional[PaperMetadata]:
        """
        Download a single paper.

        Args:
            paper: Paper information

        Returns:
            PaperMetadata if successful, None otherwise
        """
        # Check if already downloaded
        progress = self.progress.get(paper.arxiv_id)
        if progress and progress.status == "completed":
            logger.debug(f"Skipping already downloaded paper: {paper.arxiv_id}")
            return None

        # Update progress
        self.progress[paper.arxiv_id] = DownloadProgress(
            arxiv_id=paper.arxiv_id,
            status="downloading",
            last_attempt=datetime.now(timezone.utc).isoformat()
        )
        self._save_progress()

        try:
            # Classify model type and normalize attack level
            model_type_category = self._classify_model_type(paper.model_type)
            attack_level = self._normalize_attack_level(paper.taxonomy)

            # Construct file path: {model_type_category}/{attack_level}/{arxiv_id}.pdf
            relative_path = Path(
                model_type_category,
                attack_level,
                f"{paper.arxiv_id}.pdf"
            )
            output_path = self.output_dir / relative_path

            # Download PDF
            logger.info(f"Downloading {paper.arxiv_id} ({paper.model_type} / {paper.taxonomy}) to {relative_path}")
            success = self._download_with_retry(paper.pdf_url, output_path)

            if not success:
                self.progress[paper.arxiv_id].status = "failed"
                self.progress[paper.arxiv_id].error_message = "Download failed"
                self._save_progress()
                return None

            # Create metadata
            metadata = PaperMetadata(
                title=paper.title,
                authors=[],  # Will be filled from arxiv API if implemented
                venue=paper.venue,
                year=paper.year,
                arxiv_id=paper.arxiv_id,
                pdf_url=paper.pdf_url,
                model_type=paper.model_type,
                model_type_category=model_type_category,
                attack_level=attack_level,
                taxonomy=paper.taxonomy,
                file_path=str(relative_path),
                download_date=datetime.now(timezone.utc).isoformat()
            )

            # Update progress
            self.progress[paper.arxiv_id].status = "completed"
            self._save_progress()

            return metadata

        except Exception as e:
            logger.error(f"Error downloading {paper.arxiv_id}: {e}")
            self.progress[paper.arxiv_id].status = "failed"
            self.progress[paper.arxiv_id].error_message = str(e)
            self._save_progress()
            return None

    def download_all_papers(self, papers: List[ArxivPaperInfo], force_redownload: bool = False) -> None:
        """
        Download all papers with progress tracking.

        Args:
            papers: List of papers to download
            force_redownload: If True, redownload all papers
        """
        if force_redownload:
            logger.info("Force redownload enabled, clearing progress")
            self.progress = {}

        logger.info(f"Starting download of {len(papers)} papers")

        # Create directory structure
        for model_type_cat in ["any-to-t", "any-to-v"]:
            for attack_level in ["input_level", "encoder_level", "generator_level", "output_level", "unknown"]:
                (self.output_dir / model_type_cat / attack_level).mkdir(parents=True, exist_ok=True)

        # Download papers with progress bar
        new_metadata = []
        with tqdm(total=len(papers), desc="Downloading papers") as pbar:
            for paper in papers:
                metadata = self.download_paper(paper)
                if metadata:
                    new_metadata.append(metadata)
                pbar.update(1)

        # Update metadata file
        if new_metadata:
            self.metadata.extend(new_metadata)
            self._save_metadata()
            logger.info(f"Downloaded {len(new_metadata)} new papers")

        # Summary
        completed = sum(1 for p in self.progress.values() if p.status == "completed")
        failed = sum(1 for p in self.progress.values() if p.status == "failed")
        logger.info(f"Download summary: {completed} completed, {failed} failed")

    def run(self, force_redownload: bool = False) -> None:
        """
        Run the full download pipeline.

        Args:
            force_redownload: If True, redownload all papers
        """
        try:
            # Fetch and parse GitHub README
            readme_content = self.fetch_github_readme()
            papers = self.parse_jailbreak_attack_table(readme_content)

            if not papers:
                logger.error("No papers found in README")
                return

            # Download papers
            self.download_all_papers(papers, force_redownload=force_redownload)

            logger.info("Download pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download jailbreak papers from Awesome-Multimodal-Jailbreak repository"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".papers"),
        help="Output directory for papers (default: .papers/)"
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path(".papers/paper_metadata.json"),
        help="Path to metadata JSON file (default: .papers/paper_metadata.json)"
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force redownload of all papers"
    )

    args = parser.parse_args()

    # Create downloader
    downloader = JailbreakPaperDownloader(
        output_dir=args.output_dir,
        metadata_file=args.metadata_file
    )

    # Run pipeline
    downloader.run(force_redownload=args.force_redownload)


if __name__ == "__main__":
    main()

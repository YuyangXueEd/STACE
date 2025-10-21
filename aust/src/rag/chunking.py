"""Paper card chunking module for semantic section extraction."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a semantic chunk from a paper card.

    Attributes:
        arxiv_id: ArXiv identifier (e.g., "2505.11842")
        section: Section type (METHODOLOGY, EXPERIMENTS, RESULTS, RELEVANCE)
        task_type: Task taxonomy (any-to-t, any-to-v)
        attack_level: Attack taxonomy level
        paper_title: Full paper title
        card_path: Relative path to paper card markdown file
        text: Full chunk text with section prefix
    """

    arxiv_id: str
    section: str
    task_type: str
    attack_level: str
    paper_title: str
    card_path: str
    text: str


class PaperCardChunker:
    """Chunks paper cards into semantic sections for vector embedding.

    Parses markdown paper cards and extracts key sections (Methodology,
    Experiments, Key Results, Relevance) with metadata for RAG indexing.
    """

    # Target sections to extract for chunking (section heading -> normalized name)
    # Note: We use startswith matching for sections with parentheticals
    TARGET_SECTIONS = {
        "Methodology": "METHODOLOGY",
        "Experiment Design": "EXPERIMENTS",
        "Key Results (Summary)": "RESULTS",  # Actual heading in paper cards
        "Relevance to Our Work": "RELEVANCE",
    }

    MIN_CHUNK_LENGTH = 50  # Skip chunks shorter than 50 characters

    def __init__(self, paper_cards_dir: Path):
        """Initialize chunker with paper cards directory.

        Args:
            paper_cards_dir: Path to .paper_cards/ directory
        """
        self.paper_cards_dir = Path(paper_cards_dir)
        if not self.paper_cards_dir.exists():
            raise ValueError(f"Paper cards directory not found: {paper_cards_dir}")

    def chunk_card(self, card_path: Path) -> List[Chunk]:
        """Parse and chunk a single paper card.

        Args:
            card_path: Path to paper card markdown file

        Returns:
            List of Chunk objects extracted from the card

        Raises:
            ValueError: If card parsing fails
        """
        try:
            with open(card_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract metadata
            metadata = self._extract_metadata(content, card_path)

            # Extract paper title (first H1 heading)
            paper_title = self._extract_title(content)

            # Extract sections
            sections = self._extract_sections(content)

            # Build chunks
            chunks = []
            for section_heading, section_text in sections.items():
                if section_heading in self.TARGET_SECTIONS:
                    section_type = self.TARGET_SECTIONS[section_heading]

                    # Skip short chunks
                    if len(section_text.strip()) < self.MIN_CHUNK_LENGTH:
                        logger.debug(
                            f"Skipping short section {section_type} in {card_path.name}"
                        )
                        continue

                    # Format chunk text with section prefix
                    chunk_text = f"[{section_type}] {paper_title}\n\n{section_text.strip()}"

                    chunk = Chunk(
                        arxiv_id=metadata["arxiv_id"],
                        section=section_type,
                        task_type=metadata["task_type"],
                        attack_level=metadata["attack_level"],
                        paper_title=paper_title,
                        card_path=str(card_path.relative_to(self.paper_cards_dir.parent)),
                        text=chunk_text,
                    )
                    chunks.append(chunk)

            logger.debug(f"Extracted {len(chunks)} chunks from {card_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk card {card_path}: {e}")
            raise ValueError(f"Card parsing failed for {card_path}: {e}")

    def chunk_all_cards(self) -> List[Chunk]:
        """Recursively chunk all paper cards in the directory.

        Returns:
            List of all chunks from all paper cards
        """
        all_chunks = []
        failed_cards = []

        # Find all markdown files recursively
        card_files = list(self.paper_cards_dir.rglob("*.md"))

        # Skip TEMPLATE.md and README.md
        card_files = [
            f for f in card_files
            if f.name not in ("TEMPLATE.md", "README.md")
        ]

        logger.info(f"Found {len(card_files)} paper cards to process")

        for card_path in card_files:
            try:
                chunks = self.chunk_card(card_path)
                all_chunks.extend(chunks)
            except Exception as e:
                failed_cards.append((card_path, str(e)))
                logger.warning(f"Skipping failed card {card_path.name}: {e}")

        if failed_cards:
            logger.warning(
                f"Failed to parse {len(failed_cards)} cards: "
                f"{[str(p) for p, _ in failed_cards]}"
            )

        logger.info(
            f"Successfully chunked {len(card_files) - len(failed_cards)} cards "
            f"into {len(all_chunks)} chunks"
        )

        return all_chunks

    def _extract_metadata(self, content: str, card_path: Path) -> Dict[str, str]:
        """Extract metadata from card's Metadata section.

        Args:
            content: Full markdown content
            card_path: Path to card file (used for fallback taxonomy inference)

        Returns:
            Dictionary with arxiv_id, task_type, attack_level
        """
        metadata = {}

        # Extract ArXiv ID
        arxiv_match = re.search(r"ArXiv ID[:\s]+(\d+\.\d+)", content, re.IGNORECASE)
        if arxiv_match:
            metadata["arxiv_id"] = arxiv_match.group(1)
        else:
            # Fallback: try to extract from filename
            filename_match = re.match(r"(\d+\.\d+)\.md", card_path.name)
            if filename_match:
                metadata["arxiv_id"] = filename_match.group(1)
            else:
                raise ValueError("ArXiv ID not found in metadata or filename")

        # Extract Attack Level
        attack_match = re.search(
            r"Attack Level[:\s]+([\w_]+)", content, re.IGNORECASE
        )
        if attack_match:
            metadata["attack_level"] = attack_match.group(1)
        else:
            # Fallback: infer from directory structure
            # Expected: .paper_cards/any-to-t/input_level/2505.11842.md
            parts = card_path.parts
            if len(parts) >= 2:
                metadata["attack_level"] = parts[-2]  # e.g., "input_level"
            else:
                metadata["attack_level"] = "unknown"

        # Infer task_type from directory structure
        # Expected: .paper_cards/any-to-t/... or .paper_cards/any-to-v/...
        parts = card_path.parts
        for part in parts:
            if part in ("any-to-t", "any-to-v"):
                metadata["task_type"] = part
                break
        else:
            metadata["task_type"] = "unknown"

        return metadata

    def _extract_title(self, content: str) -> str:
        """Extract paper title from first H1 heading.

        Args:
            content: Full markdown content

        Returns:
            Paper title string
        """
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        return "Unknown Title"

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections by H2 headings.

        Combines all content under each H2 heading (including subsections)
        until the next H2 heading.

        Args:
            content: Full markdown content

        Returns:
            Dictionary mapping section heading to section text
        """
        sections = {}

        # Split by H2 headings (## Section Name)
        h2_pattern = r"^##\s+(.+)$"
        matches = list(re.finditer(h2_pattern, content, re.MULTILINE))

        for i, match in enumerate(matches):
            section_name = match.group(1).strip()
            start_pos = match.end()

            # Find end position (next H2 or end of file)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)

            section_text = content[start_pos:end_pos].strip()
            sections[section_name] = section_text

        return sections

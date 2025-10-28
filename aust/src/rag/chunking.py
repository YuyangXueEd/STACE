"""Paper card chunking module for semantic section extraction."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a complete paper as a single chunk with prioritized sections.

    Attributes:
        arxiv_id: ArXiv identifier (e.g., "2505.11842")
        section: Always "FULL_PAPER" - kept for backward compatibility
        task_type: Task taxonomy (any-to-t, any-to-v)
        attack_level: Attack taxonomy level
        model_type: Model modality tag from metadata (e.g., "T->I")
        paper_title: Full paper title
        card_path: Relative path to paper card markdown file
        text: Combined text from important sections (metadata, summary, methodology, etc.)
    """

    arxiv_id: str
    section: str  # Now "FULL_PAPER" instead of individual sections
    task_type: str
    attack_level: str
    model_type: str
    paper_title: str
    card_path: str
    text: str


@dataclass
class SectionBlock:
    """Represents a markdown section extracted from a paper card."""

    heading: str
    level: int
    parent_heading: Optional[str]
    content: str


class PaperCardChunker:
    """Chunks paper cards into semantic sections for vector embedding.

    Parses markdown paper cards and extracts key sections (Methodology,
    Experiments, Key Results, Relevance) with metadata for RAG indexing.
    """

    # Target sections to extract for chunking (section heading -> normalized name)
    # Note: We use startswith matching for sections with parentheticals
    SECTION_RULES = [
        {"name": "METADATA", "level": 2, "aliases": ("metadata",)},
        {"name": "SUMMARY", "level": 2, "aliases": ("quick summary",)},
        {"name": "METHODOLOGY", "level": 2, "aliases": ("methodology",)},
        {
            "name": "CORE_METHOD",
            "level": 3,
            "aliases": ("core method",),
            "parent_aliases": ("methodology",),
        },
        {
            "name": "ALGORITHM_APPROACH",
            "level": 3,
            "aliases": ("algorithm/approach", "algorithm approach", "key techniques"),
            "parent_aliases": ("methodology",),
        },
        {"name": "EXPERIMENTS", "level": 2, "aliases": ("experiment design",)},
        {"name": "RESULTS", "level": 2, "aliases": ("key results",)},
        {"name": "RELEVANCE", "level": 2, "aliases": ("relevance to our work",)},
        {"name": "IMPLEMENTATION", "level": 2, "aliases": ("implementation details",)},
        {
            "name": "ATTACK_METHODS",
            "level": 2,
            "aliases": ("potential attack methods",),
        },
    ]

    MIN_CHUNK_LENGTH = 50  # Skip chunks shorter than 50 characters
    ALLOW_SHORT_SECTIONS = {"METADATA", "SUMMARY", "CORE_METHOD"}

    def __init__(self, paper_cards_dir: Path):
        """Initialize chunker with paper cards directory.

        Args:
            paper_cards_dir: Path to .paper_cards/ directory
        """
        self.paper_cards_dir = Path(paper_cards_dir)
        if not self.paper_cards_dir.exists():
            raise ValueError(f"Paper cards directory not found: {paper_cards_dir}")

    def chunk_card(self, card_path: Path) -> List[Chunk]:
        """Parse a paper card into ONE chunk combining important sections.

        Args:
            card_path: Path to paper card markdown file

        Returns:
            List with single Chunk combining all important sections

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

            # Important sections to embed (in priority order)
            PRIORITY_SECTIONS = [
                "METADATA",
                "SUMMARY",
                "CORE_METHOD",
                "ALGORITHM_APPROACH",
                "IMPLEMENTATION",
                "ATTACK_METHODS",
            ]

            # Combine important sections into one text
            combined_parts = [f"# {paper_title}"]

            for block in sections:
                section_type = self._resolve_section_name(block)
                if not section_type or section_type not in PRIORITY_SECTIONS:
                    continue

                section_text = block.content.strip()
                if not section_text:
                    continue

                # Add section with header
                combined_parts.append(f"\n## [{section_type}]\n{section_text}")

            # Create single chunk for the entire paper
            combined_text = "\n".join(combined_parts)

            chunk = Chunk(
                arxiv_id=metadata["arxiv_id"],
                section="FULL_PAPER",  # Changed from individual sections
                task_type=metadata["task_type"],
                attack_level=metadata["attack_level"],
                model_type=metadata["model_type"],
                paper_title=paper_title,
                card_path=str(card_path.relative_to(self.paper_cards_dir.parent)),
                text=combined_text,
            )

            logger.debug(f"Created 1 combined chunk from {card_path.name}")
            return [chunk]  # Return list with single chunk

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
            Dictionary with arxiv_id, task_type, attack_level, model_type
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

        # Extract Model Type (e.g., "T->I")
        # Handle markdown bold: **Model Type**: value
        model_type_match = re.search(
            r"\*?\*?Model Type\*?\*?[:\s]+([^\n]+)", content, re.IGNORECASE
        )
        if model_type_match:
            metadata["model_type"] = model_type_match.group(1).strip()
        else:
            metadata["model_type"] = "unknown"

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

    def _extract_sections(self, content: str) -> List[SectionBlock]:
        """Extract H2/H3 sections along with their parent headings."""
        heading_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(content))

        sections: List[SectionBlock] = []
        current_h2: Optional[str] = None

        for idx, match in enumerate(matches):
            hashes, heading = match.group(1), match.group(2).strip()
            level = len(hashes)

            if level not in (2, 3):
                continue

            start_pos = match.end()
            end_pos = len(content)
            for next_match in matches[idx + 1 :]:
                next_level = len(next_match.group(1))
                if next_level <= level:
                    end_pos = next_match.start()
                    break

            section_text = content[start_pos:end_pos].strip()

            if level == 2:
                current_h2 = heading
                parent_heading = None
            else:
                parent_heading = current_h2

            sections.append(
                SectionBlock(
                    heading=heading,
                    level=level,
                    parent_heading=parent_heading,
                    content=section_text,
                )
            )

        return sections

    @staticmethod
    def _normalize_heading_key(text: Optional[str]) -> Optional[str]:
        """Normalize heading text for alias matching."""
        if text is None:
            return None
        cleaned = re.sub(r"\s*\(.*?\)", "", text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.lower()

    def _resolve_section_name(self, block: SectionBlock) -> Optional[str]:
        """Resolve normalized section name for a given section block."""
        heading_key = self._normalize_heading_key(block.heading)
        parent_key = self._normalize_heading_key(block.parent_heading)

        if heading_key is None:
            return None

        for rule in self.SECTION_RULES:
            if rule["level"] != block.level:
                continue

            if heading_key not in rule["aliases"]:
                continue

            parent_aliases = rule.get("parent_aliases")
            if parent_aliases:
                if parent_key not in parent_aliases:
                    continue

            return rule["name"]

        return None

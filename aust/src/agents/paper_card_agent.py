"""Paper Card Agent for extracting structured information from research papers.

This agent uses CAMEL-AI's ChatAgent with PDF loaders to generate structured
paper cards from academic papers for RAG retrieval.
"""

import os
import re
import json
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.loaders import UnstructuredIO, create_file_from_raw_bytes

from aust.src.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(
    log_level="INFO", log_dir=Path("aust/logs"), enable_console=True, enable_file=True
)
logger = get_logger(__name__)

load_dotenv()


class PaperCardAgent:
    """Agent for generating structured paper cards from PDF research papers."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """Initialize Paper Card Agent.

        Args:
            config_path: Path to prompt configuration YAML file
            model: Override model from config (e.g., "openai/gpt-5-nano")
            temperature: Override temperature from config
        """
        if config_path is None:
            self.config_path = (
                Path(__file__).resolve().parent.parent
                / "configs"
                / "prompts"
                / "paper_card_extraction.yaml"
            )
        else:
            self.config_path = Path(config_path)
        self.config = self._load_config()

        # Override config with parameters if provided
        if model:
            self.config["model"] = model
        if temperature is not None:
            self.config["temperature"] = temperature

        logger.info(f"Initializing PaperCardAgent with model: {self.config['model']}")

        # Create CAMEL model
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.config["model"],
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={
                "temperature": self.config.get("temperature", 0.3),
                "max_tokens": self.config.get("max_tokens", 4000),
            },
        )

        # Create ChatAgent
        self.agent = ChatAgent(
            system_message=self.config["system_prompt"], model=self.model
        )

        logger.info("PaperCardAgent initialized successfully")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.debug(f"Loaded config from {self.config_path}")
        return config

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using CAMEL loaders.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text from PDF

        Raises:
            Exception: If PDF extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting text from PDF: {pdf_path.name}")

        try:
            # Method 1: UnstructuredIO (preferred for academic papers)
            logger.debug("Attempting extraction with UnstructuredIO")
            uio = UnstructuredIO()
            elements = uio.parse_file_or_url(str(pdf_path))

            # Extract text from elements
            text = "\n\n".join([str(el) for el in elements])

            # Clean text
            pdf_config = self.config.get("pdf_config", {})
            clean_options = [
                ("replace_unicode_quotes", {}),
                ("clean_dashes", {}),
                ("clean_non_ascii_chars", {}),
                ("clean_extra_whitespace", {}),
            ]
            text = uio.clean_text_data(text=text, clean_options=clean_options)

            logger.info(
                f"Successfully extracted {len(text)} characters with UnstructuredIO"
            )

        except Exception as e:
            logger.warning(f"UnstructuredIO failed: {e}. Trying fallback method...")

            # Fallback: Base IO method
            try:
                with open(pdf_path, "rb") as file:
                    file_content = file.read()

                file_obj = create_file_from_raw_bytes(file_content, pdf_path.name)

                # Extract text from all pages
                text = "\n\n".join([doc["page_content"] for doc in file_obj.docs])

                logger.info(
                    f"Successfully extracted {len(text)} characters with Base IO fallback"
                )

            except Exception as fallback_error:
                logger.error(f"Base IO fallback also failed: {fallback_error}")
                raise Exception(
                    f"Failed to extract PDF text: {e}. Fallback also failed: {fallback_error}"
                )

        # Remove sections we do not need (e.g., references, appendices)
        pdf_config = self.config.get("pdf_config", {})
        exclude_patterns = pdf_config.get("exclude_patterns", [])
        if exclude_patterns:
            for pattern in exclude_patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    match = compiled.search(text)
                    if match:
                        logger.debug(
                            "Excluding section starting with pattern '%s' at position %d",
                            pattern,
                            match.start(),
                        )
                        text = text[: match.start()]
                except re.error as regex_error:
                    logger.warning(
                        "Invalid exclude pattern '%s': %s", pattern, regex_error
                    )

        # Truncate if too long
        max_length = pdf_config.get("max_text_length", 50000)
        if len(text) > max_length:
            logger.warning(
                f"Text length {len(text)} exceeds max {max_length}, truncating"
            )
            text = text[:max_length] + "\n\n[... text truncated due to length ...]"

        return text

    def extract_github_link(self, pdf_text: str) -> str:
        """Extract GitHub repository URL from PDF text.

        Args:
            pdf_text: Extracted PDF text

        Returns:
            GitHub URL if found, otherwise "Not provided"
        """
        # Regex pattern for GitHub URLs
        github_pattern = r"https?://github\.com/[\w\-]+/[\w\-\.]+"

        matches = re.findall(github_pattern, pdf_text)

        if matches:
            # Return first match (usually main repo)
            github_url = matches[0]
            logger.debug(f"Found GitHub URL: {github_url}")
            return github_url

        logger.debug("No GitHub URL found in PDF text")
        return "Not provided"

    def generate_card(
        self,
        pdf_path: str,
        metadata: Dict[str, Any],
        output_dir: str = ".paper_cards/",
    ) -> str:
        """Generate paper card for a single PDF.

        Args:
            pdf_path: Path to PDF file
            metadata: Paper metadata (title, authors, venue, year, arxiv_id, etc.)
            output_dir: Output directory for paper cards

        Returns:
            Path to generated paper card markdown file

        Raises:
            Exception: If card generation fails
        """
        arxiv_id = metadata.get("arxiv_id")
        logger.info(f"Generating paper card for {arxiv_id}")

        # Extract PDF text
        try:
            pdf_text = self.extract_pdf_text(pdf_path)
        except Exception as e:
            logger.error(f"PDF extraction failed for {arxiv_id}: {e}")
            raise

        # Extract GitHub link
        github_link = self.extract_github_link(pdf_text)

        # Format metadata for prompt
        prompt_metadata = {
            "title": metadata.get("title", "Unknown Title"),
            "authors": ", ".join(metadata.get("authors", [])) or "Unknown Authors",
            "venue": metadata.get("venue", "Unknown Venue"),
            "year": metadata.get("year", "Unknown Year"),
            "arxiv_id": arxiv_id,
            "model_type": metadata.get("model_type", "Unknown"),
            "attack_level": metadata.get("attack_level", "Unknown"),
            "file_path": metadata.get("file_path", ""),
            "pdf_text": pdf_text,
        }

        # Generate card using LLM
        logger.info(f"Calling LLM for structured extraction ({self.config['model']})")

        # Use string replace instead of format to avoid issues with curly braces in template
        user_prompt = self.config["user_prompt_template"]
        for key, value in prompt_metadata.items():
            user_prompt = user_prompt.replace(f"{{{key}}}", str(value))

        try:
            response = self.agent.step(user_prompt)
            card_content = response.msgs[0].content

            logger.info(
                f"LLM extraction complete - generated {len(card_content)} characters"
            )

        except Exception as e:
            logger.error(f"LLM extraction failed for {arxiv_id}: {e}")
            raise Exception(f"LLM extraction failed: {e}")

        # Post-process card content (fix metadata placeholders)
        card_content = self._post_process_card(
            card_content, metadata, github_link
        )

        # Save card to output directory
        output_path = self._save_card(card_content, metadata, output_dir)

        logger.info(f"Paper card saved to: {output_path}")
        return output_path

    def _post_process_card(
        self, card_content: str, metadata: Dict[str, Any], github_link: str
    ) -> str:
        """Post-process card content to fix placeholders and add metadata.

        Args:
            card_content: Raw card content from LLM
            metadata: Paper metadata
            github_link: Extracted GitHub link

        Returns:
            Post-processed card content
        """
        # Add generation timestamp
        generation_date = datetime.now(timezone.utc).isoformat()

        # Replace timestamp placeholder if present
        if "[Current timestamp in ISO format]" in card_content:
            card_content = card_content.replace(
                "[Current timestamp in ISO format]", generation_date
            )

        # Ensure metadata footer exists
        if "**Card Generated**:" not in card_content:
            card_content += f"\n\n---\n\n**Card Generated**: {generation_date}\n**Agent Model**: {self.config['model']}\n"

        return card_content

    def _save_card(
        self, card_content: str, metadata: Dict[str, Any], output_dir: str
    ) -> str:
        """Save paper card to output directory.

        Args:
            card_content: Generated card content
            metadata: Paper metadata
            output_dir: Output directory

        Returns:
            Path to saved card file
        """
        arxiv_id = metadata.get("arxiv_id")
        file_path = metadata.get("file_path", "")

        # Construct output path matching .papers/ structure
        output_path = Path(output_dir) / file_path.replace(".pdf", ".md")

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write card
        with open(output_path, "w") as f:
            f.write(card_content)

        logger.debug(f"Card written to: {output_path}")
        return str(output_path)

    def validate_card_quality(self, card_path: str) -> float:
        """Calculate quality score for generated paper card.

        Args:
            card_path: Path to paper card markdown file

        Returns:
            Quality score (0.0 - 1.0)
        """
        with open(card_path, "r") as f:
            card_content = f.read()

        score = 0.0
        thresholds = self.config.get("quality_thresholds", {})

        # Check core method length
        core_method_match = re.search(
            r"### Core Method\n(.+?)(?=\n###|\n##|$)", card_content, re.DOTALL
        )
        if core_method_match:
            core_method = core_method_match.group(1).strip()
            if len(core_method) > thresholds.get("min_core_method_length", 50):
                score += 0.3
                logger.debug(f"Core method check passed: {len(core_method)} chars")

        # Check key techniques count
        key_techniques_match = re.search(
            r"### Key Techniques\n(.+?)(?=\n###|\n##|$)", card_content, re.DOTALL
        )
        if key_techniques_match:
            key_techniques = key_techniques_match.group(1).strip()
            technique_count = len(re.findall(r"^- ", key_techniques, re.MULTILINE))
            if technique_count >= thresholds.get("min_key_techniques", 3):
                score += 0.2
                logger.debug(f"Key techniques check passed: {technique_count} items")

        # Check datasets
        datasets_match = re.search(
            r"### Datasets\n(.+?)(?=\n###|\n##|$)", card_content, re.DOTALL
        )
        if datasets_match:
            datasets = datasets_match.group(1).strip()
            if (
                len(datasets) > 10
                and "Not provided" not in datasets
                and "Not mentioned" not in datasets
            ):
                score += 0.2
                logger.debug("Datasets check passed")

        # Check main findings length
        findings_match = re.search(
            r"### Main Findings\n(.+?)(?=\n###|\n##|$)", card_content, re.DOTALL
        )
        if findings_match:
            findings = findings_match.group(1).strip()
            if len(findings) > thresholds.get("min_main_findings_length", 100):
                score += 0.2
                logger.debug(f"Main findings check passed: {len(findings)} chars")

        # Check relevance analysis length
        relevance_match = re.search(
            r"### Application to Data-Based Unlearning\n(.+?)(?=\n###|\n##|$)",
            card_content,
            re.DOTALL,
        )
        if relevance_match:
            relevance = relevance_match.group(1).strip()
            if len(relevance) > thresholds.get("min_relevance_length", 50):
                score += 0.1
                logger.debug(f"Relevance check passed: {len(relevance)} chars")

        logger.info(f"Card quality score: {score:.2f}")
        return score


def main():
    """Example usage of Paper Card Agent."""
    # Initialize agent
    agent = PaperCardAgent(
        config_path="configs/prompts/paper_card_extraction.yaml",
        model="openai/gpt-5-nano",
    )

    # Load paper metadata
    with open(".papers/paper_metadata.json", "r") as f:
        all_metadata = json.load(f)

    # Test with first paper
    test_metadata = all_metadata[0]
    pdf_path = f".papers/{test_metadata['file_path']}"

    logger.info(f"Testing with paper: {test_metadata['title']}")

    # Generate card
    card_path = agent.generate_card(
        pdf_path=pdf_path, metadata=test_metadata, output_dir=".paper_cards/"
    )

    # Validate quality
    quality_score = agent.validate_card_quality(card_path)

    print(f"\nPaper card generated: {card_path}")
    print(f"Quality score: {quality_score:.2f}")

    if quality_score < 0.7:
        logger.warning(f"Low quality score: {quality_score:.2f} (threshold: 0.7)")


if __name__ == "__main__":
    main()

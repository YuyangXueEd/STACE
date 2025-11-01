"""
Long-Term Memory Agent for storing and retrieving successful attack discoveries.

This agent manages persistent memory of vulnerability discoveries across runs,
enabling the system to learn from past successes and avoid redundant exploration.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aust.src.data_models import AttackMemoryCard, Hypothesis, IterationResult
from aust.src.utils.logging_config import get_logger
from aust.src.utils.markdown_parser import extract_markdown_sections

logger = get_logger(__name__)


class LongTermMemoryAgent:
    """
    Manages long-term memory of successful vulnerability discoveries.

    Stores attack memory cards as structured markdown files in:
    outputs/memory_store/attacks/{attack_id}.md
    """

    def __init__(self, memory_dir: Optional[Path] = None):
        """
        Initialize Long-Term Memory Agent.

        Args:
            memory_dir: Directory for storing attack memory cards.
                       Defaults to outputs/memory_store/attacks/
        """
        if memory_dir is None:
            # Default to outputs/memory_store/attacks/ at the project root
            project_root = Path(__file__).resolve().parents[3]
            memory_dir = project_root / "outputs" / "memory_store" / "attacks"

        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LongTermMemoryAgent initialized with memory_dir: {self.memory_dir}")

    def create_attack_card(
        self,
        hypothesis: Hypothesis,
        iteration_result: IterationResult,
        task_spec: dict[str, Any],
        attack_trace_path: str,
    ) -> AttackMemoryCard:
        """
        Create an attack memory card from iteration results.

        Args:
            hypothesis: The successful hypothesis
            iteration_result: Results from the successful iteration
            task_spec: Task specification dict
            attack_trace_path: Path to attack trace markdown file

        Returns:
            AttackMemoryCard instance ready for storage
        """
        # Generate attack ID from task_id and iteration
        task_id = task_spec.get("task_id", "unknown")
        attack_id = f"{task_id}_iter{iteration_result.iteration_number}"

        # Extract experiment parameters
        experiment_results = iteration_result.experiment_results or {}
        experiment_parameters = {
            "prompts": experiment_results.get("prompts", []),
            "seeds": experiment_results.get("seeds", []),
            "num_images": experiment_results.get("num_images", 0),
            "guidance_scale": experiment_results.get("guidance_scale"),
            "num_steps": experiment_results.get("num_steps"),
        }

        # Parse successful prompts from evaluator feedback
        successful_prompts = self._extract_successful_prompts(
            iteration_result.evaluator_feedback or ""
        )

        # Extract key findings from evaluator feedback
        key_findings = self._extract_key_findings(
            iteration_result.evaluator_feedback or ""
        )

        card = AttackMemoryCard(
            attack_id=attack_id,
            task_type=task_spec.get("task_type", "unknown"),
            unlearned_target=task_spec.get("unlearned_target", "unknown"),
            unlearning_method=task_spec.get("unlearning_method"),
            model_name=task_spec.get("model_name"),
            hypothesis_attack_type=hypothesis.attack_type,
            hypothesis_summary=hypothesis.description,  # Use description field
            hypothesis_reasoning=hypothesis.experiment_design,  # Use experiment_design as reasoning
            hypothesis_full=hypothesis.model_dump(),
            experiment_parameters=experiment_parameters,
            detection_rate=iteration_result.vulnerability_confidence,  # Using as proxy
            max_confidence=iteration_result.vulnerability_confidence,
            vulnerability_confidence=iteration_result.vulnerability_confidence,
            successful_prompts=successful_prompts,
            key_findings=key_findings,
            attack_trace_path=attack_trace_path,
            iteration_number=iteration_result.iteration_number,
            discovered_at=datetime.now(timezone.utc),
        )

        logger.debug(f"Created attack memory card: {attack_id}")
        return card

    def _extract_successful_prompts(self, evaluator_feedback: str) -> list[dict[str, Any]]:
        """Extract successful prompts from evaluator feedback."""
        successful_prompts = []

        # Look for "Successful Prompts" section in feedback
        prompts_match = re.search(
            r"Successful Prompts:.*?\n(.*?)(?=\n\n|\Z)",
            evaluator_feedback,
            re.DOTALL,
        )

        if prompts_match:
            prompts_text = prompts_match.group(1)
            # Parse lines like: • 'kitten' (100.0%)
            for line in prompts_text.split("\n"):
                match = re.search(r"['\"](.*?)['\"].*?\((\d+\.?\d*)%\)", line)
                if match:
                    prompt = match.group(1)
                    rate_str = match.group(2)
                    successful_prompts.append(
                        {
                            "prompt": prompt,
                            "detection_rate": float(rate_str) / 100.0,
                        }
                    )

        return successful_prompts

    def _extract_key_findings(self, evaluator_feedback: str) -> list[str]:
        """Extract key findings from evaluator feedback."""
        findings = []

        # Extract vulnerability detected message as primary finding
        if "Vulnerability Detected:" in evaluator_feedback:
            # Extract the vulnerability message
            msg_match = re.search(
                r"⚠️ Vulnerability Detected: (.+?)(?=\n\n|\Z)",
                evaluator_feedback,
                re.DOTALL,
            )
            if msg_match:
                findings.append(msg_match.group(1).strip())

        # Add detection rate as finding
        rate_match = re.search(r"Concept Detected: \d+/\d+ \((\d+\.?\d*)%\)", evaluator_feedback)
        if rate_match:
            detection_rate = rate_match.group(1)
            findings.append(
                f"Detection rate: {detection_rate}% of generated images contained the concept"
            )

        # Add max confidence as finding
        conf_match = re.search(r"Max Confidence: (\d+\.?\d+)", evaluator_feedback)
        if conf_match:
            max_conf = conf_match.group(1)
            findings.append(
                f"Maximum detection confidence: {max_conf} (high confidence indicates clear leakage)"
            )

        return findings

    def store_attack(self, card: AttackMemoryCard) -> Path:
        """
        Store attack memory card to disk.

        Args:
            card: AttackMemoryCard to store

        Returns:
            Path to stored markdown file
        """
        filepath = self.memory_dir / f"{card.attack_id}.md"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(card.to_markdown())

            logger.info(f"Stored attack memory card: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to store attack memory card {card.attack_id}: {e}")
            raise

    def get_attacks_by_task_type(
        self,
        task_type: str,
        min_confidence: float = 0.0,
        max_results: Optional[int] = None,
    ) -> list[AttackMemoryCard]:
        """
        Retrieve attack memory cards filtered by task type.

        Args:
            task_type: Filter by task type (concept_erasure, data_based_unlearning)
            min_confidence: Minimum vulnerability confidence (0.0-1.0)
            max_results: Maximum number of results to return (most recent first)

        Returns:
            List of AttackMemoryCard instances matching criteria
        """
        logger.debug(
            f"Querying memory: task_type={task_type}, min_confidence={min_confidence}"
        )

        attacks = []

        # Scan all markdown files in memory directory
        for filepath in self.memory_dir.glob("*.md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

                # Parse attack card
                attack_id = filepath.stem
                card = AttackMemoryCard.from_markdown(markdown_content, attack_id)

                # Apply filters
                if card.task_type == task_type and card.vulnerability_confidence >= min_confidence:
                    attacks.append(card)

            except Exception as e:
                logger.warning(f"Failed to parse attack memory card {filepath}: {e}")
                continue

        # Sort by discovery time (most recent first)
        attacks.sort(key=lambda x: x.discovered_at, reverse=True)

        # Limit results
        if max_results is not None:
            attacks = attacks[:max_results]

        logger.info(f"Retrieved {len(attacks)} attack memory cards")
        return attacks

    def get_all_attacks(self) -> list[AttackMemoryCard]:
        """
        Retrieve all stored attack memory cards.

        Returns:
            List of all AttackMemoryCard instances
        """
        return self.get_attacks_by_task_type(
            task_type="",  # Empty string matches nothing, but we override in loop
            min_confidence=0.0,
            max_results=None,
        )

    def get_memory_summary_for_context(
        self,
        task_type: str,
        min_confidence: float = 0.5,
        max_entries: int = 5,
    ) -> list[dict[str, str]]:
        """
        Get summarized memory entries for hypothesis context.

        Extracts only relevant sections (METADATA, HYPOTHESIS, RESULTS, LESSONS_LEARNED)
        to minimize token usage.

        Args:
            task_type: Filter by task type
            min_confidence: Minimum confidence threshold
            max_entries: Maximum number of entries to return

        Returns:
            List of dicts with attack_id and summary (markdown excerpt)
        """
        attacks = self.get_attacks_by_task_type(
            task_type=task_type,
            min_confidence=min_confidence,
            max_results=max_entries,
        )

        memory_entries = []

        for attack in attacks:
            # Generate full markdown
            markdown_content = attack.to_markdown()

            # Extract only relevant sections
            summary = extract_markdown_sections(
                markdown_content,
                sections_to_extract=["METADATA", "HYPOTHESIS", "RESULTS", "LESSONS_LEARNED"],
            )

            memory_entries.append(
                {
                    "attack_id": attack.attack_id,
                    "summary": summary,
                }
            )

        logger.debug(f"Generated {len(memory_entries)} memory summaries for context")
        return memory_entries


__all__ = ["LongTermMemoryAgent"]

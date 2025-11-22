"""Reporter Agent for generating long-form academic vulnerability reports."""

from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import yaml


class LiteralString(str):
    """Signal to YAML dumper to emit literal block style (|)."""


class FoldedString(str):
    """Signal to YAML dumper to emit folded block style (>)."""


class TemplateSafeDumper(yaml.SafeDumper):
    """Custom dumper that supports LiteralString and FoldedString helpers."""


def _literal_str_representer(dumper: yaml.Dumper, data: LiteralString):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _folded_str_representer(dumper: yaml.Dumper, data: FoldedString):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")


TemplateSafeDumper.add_representer(LiteralString, _literal_str_representer)
TemplateSafeDumper.add_representer(FoldedString, _folded_str_representer)

from aust.src.data_models.loop_state import InnerLoopState
from aust.src.data_models.report import (
    AcademicReport,
    NoveltyInfo,
    ReportMetadata,
    ReportSection,
    ReportSectionType,
)
from aust.src.rag.vector_db import PaperRAG
from aust.src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ReporterAgent:
    """
    Agent responsible for generating academic reports from inner loop results.

    The Reporter reads attack traces, inner loop state, and experiment results
    to generate structured academic papers with standard sections:
    - Introduction
    - Methods
    - Experiments
    - Results
    - Discussion
    - Conclusion

    This is Story 4.1 implementation (structure and outlines).
    Story 4.2 will add detailed content generation with citations.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        rag_storage_path: Optional[Path] = None,
    ):
        """
        Initialize Reporter Agent.

        Args:
            output_dir: Base output directory for reports (default: ./outputs)
            rag_storage_path: Path to RAG vector store for novelty calculation
        """
        self.output_dir = output_dir or Path("./outputs")
        self.rag_storage_path = rag_storage_path or (self._get_project_root() / "aust" / "rag_paper_db")
        self._config = self._load_report_config()
        report_cfg = self._config.get("report", {}) if isinstance(self._config, dict) else {}
        self._target_word_counts = (
            report_cfg.get("target_word_counts")
            if isinstance(report_cfg, dict)
            else {}
        )

        logger.info("ReporterAgent initialized (output_dir: %s)", self.output_dir)

    def generate_report(
        self,
        inner_loop_state: InnerLoopState,
        attack_trace_json_path: Path,
        attack_trace_md_path: Path,
        retrieved_papers: Optional[dict[str, dict[str, Any]]] = None,
    ) -> AcademicReport:
        """
        Generate complete academic report from inner loop results.

        Args:
            inner_loop_state: Final inner loop state
            attack_trace_json_path: Path to JSON attack trace
            attack_trace_md_path: Path to Markdown attack trace
            retrieved_papers: Optional mapping of paper IDs to metadata

        Returns:
            AcademicReport with all sections and metadata
        """
        logger.info("Generating report for task %s", inner_loop_state.task_id)

        attack_trace_data = self._load_attack_trace(attack_trace_json_path)
        metadata = self._create_metadata(inner_loop_state, attack_trace_data)
        report = AcademicReport(metadata=metadata, references=retrieved_papers or {})

        context = self._collect_report_context(
            inner_loop_state,
            attack_trace_data,
            attack_trace_md_path,
        )

        reference_sources = retrieved_papers or {}
        if not reference_sources:
            reference_sources = self._collect_papers_from_state(inner_loop_state)
        context["retrieved_papers"] = reference_sources

        citation_map = self._apply_references(report, reference_sources)

        for section in self._build_sections(report.metadata, context, citation_map):
            report.add_section(section)

        logger.info("Report generated: %s", report.metadata.report_id)
        logger.info("  Sections: %s", len(report.sections))
        logger.info("  Total word count: %s", report.total_word_count)
        logger.info("  Complete: %s", report.is_complete)

        return report

    def save_report(self, report: AcademicReport, task_id: str) -> Path:
        """
        Save report to file.

        Args:
            report: AcademicReport to save
            task_id: Task identifier for directory structure

        Returns:
            Path where report was saved
        """
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_filename = f"report_{report.metadata.report_id}.md"
        report_path = reports_dir / report_filename

        report.save_to_file(report_path)
        logger.info("Report saved to %s", report_path)

        return report_path

    # =========================================================================
    # Data preparation
    # =========================================================================

    def _collect_report_context(
        self,
        inner_loop_state: InnerLoopState,
        attack_trace_data: dict[str, Any],
        attack_trace_md_path: Path,
    ) -> dict[str, Any]:
        """Collect contextual information required for section generation."""
        trace_iterations: dict[int, dict[str, Any]] = {}
        if isinstance(attack_trace_data, dict):
            for entry in attack_trace_data.get("iterations", []) or []:
                if isinstance(entry, dict) and "iteration_number" in entry:
                    trace_iterations[entry["iteration_number"]] = entry

        iteration_details: list[dict[str, Any]] = []
        success_count = 0
        executed_count = 0
        confidence_scores: list[float] = []
        rag_queries: set[str] = set()

        for iteration in inner_loop_state.iterations:
            trace_entry = trace_iterations.get(iteration.iteration_number, {})
            evaluation = trace_entry.get("evaluation", {}) if isinstance(trace_entry, dict) else {}

            detected = evaluation.get("vulnerability_detected", iteration.vulnerability_detected)
            confidence = evaluation.get("vulnerability_confidence", iteration.vulnerability_confidence)

            if detected:
                success_count += 1
            if iteration.experiment_executed:
                executed_count += 1
            if confidence is not None:
                confidence_scores.append(confidence)

            rag_queries.update(iteration.rag_queries or [])
            if iteration.debate_session:
                rag_queries.update(iteration.debate_session.rag_queries)

            iteration_details.append(
                {
                    "iteration_number": iteration.iteration_number,
                    "attack_type": iteration.hypothesis.attack_type,
                    "description": iteration.hypothesis.description,
                    "experiment_executed": iteration.experiment_executed,
                    "vulnerability_detected": detected,
                    "vulnerability_confidence": confidence or 0.0,
                    "outcome": evaluation.get("outcome", iteration.outcome),
                    "retrieved_paper_ids": list(iteration.retrieved_paper_ids),
                }
            )

        average_confidence = mean(confidence_scores) if confidence_scores else 0.0

        return {
            "state": inner_loop_state,
            "attack_trace": attack_trace_data,
            "attack_trace_excerpt": self._read_attack_trace_excerpt(attack_trace_md_path),
            "iteration_details": iteration_details,
            "success_count": success_count,
            "executed_count": executed_count,
            "confidence_scores": confidence_scores,
            "average_confidence": average_confidence,
            "rag_queries": sorted(rag_queries),
        }

    def _collect_papers_from_state(self, inner_loop_state: InnerLoopState) -> dict[str, dict[str, Any]]:
        """Aggregate retrieved paper metadata from the inner loop state."""
        aggregated: dict[str, dict[str, Any]] = {}

        for iteration in inner_loop_state.iterations:
            for paper_id in iteration.retrieved_paper_ids or []:
                if paper_id and paper_id not in aggregated:
                    aggregated[paper_id] = {"title": paper_id, "source": "rag_memory"}

            if iteration.debate_session:
                for snippet in iteration.debate_session.retrieved_papers:
                    if not isinstance(snippet, dict):
                        continue
                    citation_key = snippet.get("paper_id") or snippet.get("id")
                    if not citation_key:
                        continue
                    aggregated.setdefault(citation_key, snippet)

        return aggregated

    def _read_attack_trace_excerpt(self, attack_trace_md_path: Path, max_lines: int = 40) -> str:
        """Read the first few lines of the attack trace markdown for context."""
        if not attack_trace_md_path or not attack_trace_md_path.exists():
            return ""

        try:
            lines: list[str] = []
            with attack_trace_md_path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if idx >= max_lines:
                        break
                    lines.append(line.rstrip())
            return "\n".join(lines).strip()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read attack trace markdown %s: %s", attack_trace_md_path, exc)
            return ""

    # =========================================================================
    # Metadata creation
    # =========================================================================

    def _create_metadata(
        self,
        inner_loop_state: InnerLoopState,
        attack_trace_data: dict[str, Any],
    ) -> ReportMetadata:
        """Create report metadata from inner loop state."""
        task_spec = inner_loop_state.task_spec or {}

        # Calculate novelty if we have a final successful hypothesis
        novelty_info = None
        if inner_loop_state.vulnerability_found and inner_loop_state.iterations:
            # Get the final iteration's hypothesis
            final_iteration = inner_loop_state.iterations[-1]
            if final_iteration.hypothesis and final_iteration.hypothesis.description:
                novelty_info = self.calculate_hypothesis_novelty(
                    final_iteration.hypothesis.description
                )

        return ReportMetadata(
            report_id=f"report_{uuid.uuid4().hex[:8]}",
            task_id=inner_loop_state.task_id,
            task_type=inner_loop_state.task_type,
            target_model_name=task_spec.get("model_name"),
            target_model_version=task_spec.get("model_version"),
            unlearning_method=task_spec.get("unlearning_method"),
            unlearned_target=task_spec.get("unlearned_target"),
            total_iterations=len(inner_loop_state.iterations),
            vulnerability_found=inner_loop_state.vulnerability_found,
            highest_confidence=inner_loop_state.highest_vulnerability_confidence,
            generated_at=datetime.now(timezone.utc),
            inner_loop_started_at=inner_loop_state.started_at,
            inner_loop_completed_at=inner_loop_state.completed_at,
            generator_model=attack_trace_data.get("configuration", {}).get("generator_model"),
            critic_model=attack_trace_data.get("configuration", {}).get("critic_model"),
            novelty_info=novelty_info,
        )

    # =========================================================================
    # Section builders
    # =========================================================================

    def _build_sections(
        self,
        metadata: ReportMetadata,
        context: dict[str, Any],
        citation_map: dict[str, dict[str, Any]],
    ) -> list[ReportSection]:
        """Construct all report sections in order."""
        return [
            self._create_introduction_outline(metadata),
            self._create_methods_outline(metadata, context),
            self._create_experiments_outline(metadata, context),
            self._create_results_outline(metadata, context),
            self._create_discussion_outline(metadata, context, citation_map),
            self._create_conclusion_outline(metadata),
        ]

    def _create_introduction_outline(self, metadata: ReportMetadata) -> ReportSection:
        """Generate Introduction section outline."""
        target_model = metadata.target_model_name or "the target model"
        target_version = f" v{metadata.target_model_version}" if metadata.target_model_version else ""
        unlearning_method = metadata.unlearning_method or "the specified unlearning method"
        unlearned_target = metadata.unlearned_target or "the target concept"
        expected_outcome = (
            "a confirmed vulnerability"
            if metadata.vulnerability_found
            else "no high-confidence vulnerability within the explored iterations"
        )

        outline_content = f"""### Problem Statement
- Assess whether {unlearning_method} robustly removes {unlearned_target} from {target_model}{target_version} without collateral failures.

### Objectives
- Measure residual leakage after concept erasure across {metadata.total_iterations or 'the recorded'} iteration(s).
- Evaluate hypothesis generation and debate quality to understand unlearning robustness.
- Produce a structured record suitable for judge agents and external reviewers.

### System Context
- Task ID: {metadata.task_id}
- Target model: {target_model}{target_version}
- Expected outcome: {expected_outcome}.
"""

        return ReportSection(
            section_type=ReportSectionType.INTRODUCTION,
            title="Introduction",
            content=outline_content.strip(),
            order=1,
        )

    def _create_methods_outline(
        self,
        metadata: ReportMetadata,
        context: dict[str, Any],
    ) -> ReportSection:
        """Generate Methods section outline."""
        attack_trace = context.get("attack_trace") or {}
        config = attack_trace.get("configuration", {}) if isinstance(attack_trace, dict) else {}
        generator_model = config.get("generator_model") or metadata.generator_model or "unknown generator"
        critic_model = config.get("critic_model") or metadata.critic_model or "unknown critic"
        debate_enabled = config.get("enable_debate")
        if debate_enabled is None and context.get("state"):
            debate_enabled = context["state"].enable_debate
        debate_flag = "enabled" if debate_enabled else "disabled"
        rag_queries = context.get("rag_queries", [])

        # Add novelty information if available
        novelty_section = ""
        if metadata.novelty_info:
            novelty_score = metadata.novelty_info.novelty_score
            max_sim = metadata.novelty_info.max_similarity
            top_papers = metadata.novelty_info.top_similar_papers

            novelty_section = f"""

### Novelty Assessment
- **Novelty Score**: {novelty_score:.3f} (calculated as 1 - max_similarity)
- **Maximum Similarity**: {max_sim:.3f} (compared against {len(context.get('rag_queries', []))} papers in database)
- **Most Similar Prior Work**:
"""
            for i, paper in enumerate(top_papers, 1):
                novelty_section += f"\n  {i}. {paper['paper_title']} (similarity: {paper['similarity']:.3f})"

            if novelty_score >= 0.7:
                novelty_section += "\n- This hypothesis demonstrates **high novelty** compared to existing literature."
            elif novelty_score >= 0.4:
                novelty_section += "\n- This hypothesis shows **moderate novelty**, building upon existing approaches."
            else:
                novelty_section += "\n- This hypothesis has **low novelty**, closely resembling known methods."

        outline_content = f"""### Hypothesis Generation Process
- Generator model: {generator_model}
- Critic model: {critic_model}
- Debate workflow is {debate_flag}, coordinating hypothesis refinement through structured exchanges.

### Multi-Agent Debate
- Each iteration captures generator proposals and critic assessments before experiments execute.
- Retrieved context informs refinement; {len(rag_queries)} paper retrieval queries were issued across the loop.
- Debate transcripts are persisted for judge review and reproducibility.
{novelty_section}

### Evaluation Workflow
- Experiment executors follow the agreed plan, logging outputs to the attack trace (JSON + Markdown).
- Evaluation agents compute vulnerability confidence scores and trigger safeguards when nudity detectors fire.
- Final results feed into the Reporter along with trace artifacts to build the academic narrative.
"""

        return ReportSection(
            section_type=ReportSectionType.METHODS,
            title="Methods",
            content=outline_content.strip(),
            order=2,
        )

    def _create_experiments_outline(
        self,
        metadata: ReportMetadata,
        context: dict[str, Any],
    ) -> ReportSection:
        """Generate Experiments section outline."""
        iteration_details = context.get("iteration_details", [])
        retrieved_papers = context.get("retrieved_papers") or {}
        executed_count = context.get("executed_count", 0)

        iteration_lines: list[str] = []
        for detail in iteration_details:
            status = "vulnerability confirmed" if detail["vulnerability_detected"] else "no vulnerability detected"
            executed = "executed" if detail["experiment_executed"] else "planned"
            confidence_pct = round(detail["vulnerability_confidence"] * 100)
            iteration_lines.append(
                f"- Iteration {detail['iteration_number']}: {detail['attack_type']} ({executed}) -> {status} "
                f"(confidence {confidence_pct}%)"
            )

        if not iteration_lines:
            iteration_lines.append("- No experiment records were available in the inner loop state.")

        iteration_overview = "\n".join(iteration_lines)
        trace_excerpt = context.get("attack_trace_excerpt") or "Attack trace markdown was unavailable for this run."
        task_description = context.get("state").task_description if context.get("state") else "Not specified"

        outline_content = (
            "### Experiment Setup\n"
            f"- Task description: {task_description}.\n"
            f"- Total iterations recorded: {metadata.total_iterations}\n"
            f"- Experiments executed: {executed_count} / {metadata.total_iterations}\n"
            f"- Retrieved paper references: {len(retrieved_papers)}\n\n"
            "### Iteration Overview\n"
            f"{iteration_overview}\n\n"
            "### Data and Tooling\n"
            f"- Attack trace excerpt:\n{trace_excerpt}\n"
        )

        return ReportSection(
            section_type=ReportSectionType.EXPERIMENTS,
            title="Experiments",
            content=outline_content.strip(),
            order=3,
        )

    def _create_results_outline(
        self,
        metadata: ReportMetadata,
        context: dict[str, Any],
    ) -> ReportSection:
        """Generate Results section outline."""
        success_count = context.get("success_count", 0)
        average_confidence = context.get("average_confidence", 0.0)
        iteration_details = context.get("iteration_details", [])

        iteration_rows: list[str] = []
        for detail in iteration_details:
            confidence_pct = round(detail["vulnerability_confidence"] * 100)
            iteration_rows.append(
                f"- Iteration {detail['iteration_number']}: {detail['attack_type']} -> "
                f"{'success' if detail['vulnerability_detected'] else 'failure'} "
                f"(confidence {confidence_pct}%) - {detail['outcome']}"
            )

        avg_conf_display = f"{average_confidence:.0%}" if average_confidence else "N/A"
        iteration_outcomes = "\n".join(iteration_rows) if iteration_rows else "- No iteration details available."

        summary_block = (
            "### Summary Statistics\n"
            f"- Total iterations: {metadata.total_iterations}\n"
            f"- Successful attacks: {success_count} / {metadata.total_iterations}\n"
            f"- Average vulnerability confidence: {avg_conf_display}\n\n"
            "### Iteration Outcomes\n"
            f"{iteration_outcomes}\n"
        )

        return ReportSection(
            section_type=ReportSectionType.RESULTS,
            title="Results",
            content=summary_block.strip(),
            order=4,
        )

    def _create_discussion_outline(
        self,
        metadata: ReportMetadata,
        context: dict[str, Any],
        citation_map: dict[str, dict[str, Any]],
    ) -> ReportSection:
        """Generate Discussion section outline."""
        average_confidence = context.get("average_confidence", 0.0)
        retrieved_papers = context.get("retrieved_papers") or {}
        success_count = context.get("success_count", 0)
        total_iterations = metadata.total_iterations or 1

        strengths = [
            f"- Multi-agent debate surfaced reproducible hypotheses across {total_iterations} iteration(s).",
            f"- Evidence captured in attack traces enables downstream judge agents to audit findings.",
        ]

        limitations = [
            "- Limited iteration budget may under-represent long-horizon attacks.",
            "- Retrieved paper evidence is synthesized; full citation grounding arrives in Story 4.2.",
        ]

        future_work = [
            "- Expand the experiment executor to quantify collateral damage metrics.",
            "- Integrate judge feedback to refine reporting thresholds and narrative templates.",
        ]

        avg_conf_discussion = f"{average_confidence:.0%}" if average_confidence else "N/A"
        strengths_block = "\n".join(strengths)
        limitations_block = "\n".join(limitations)
        future_work_block = "\n".join(future_work)

        discussion_content = (
            "### Interpretation\n"
            f"- Success rate: {success_count} / {total_iterations} iterations produced confirmed vulnerabilities.\n"
            f"- Average confidence across successful detections: {avg_conf_discussion}.\n"
            f"- References captured: {len(retrieved_papers)} (see bibliography for details).\n\n"
            "### Strengths\n"
            f"{strengths_block}\n\n"
            "### Limitations\n"
            f"{limitations_block}\n\n"
            "### Future Work\n"
            f"{future_work_block}\n"
        )

        return ReportSection(
            section_type=ReportSectionType.DISCUSSION,
            title="Discussion",
            content=discussion_content.strip(),
            order=5,
        )

    def _create_conclusion_outline(self, metadata: ReportMetadata) -> ReportSection:
        """Generate Conclusion section outline."""
        outline_content = f"""### Conclusion Outline

**Summary of Contribution**:
This automated vulnerability assessment of {metadata.target_model_name or 'the target model'} using {metadata.unlearning_method or 'the specified unlearning method'} demonstrates the feasibility of autonomous security testing for machine unlearning.

**Main Conclusions**:
- {'Critical vulnerabilities were discovered' if metadata.vulnerability_found else 'No critical vulnerabilities were found'} through {metadata.total_iterations} iteration(s)
- Automated hypothesis generation and refinement successfully {'identified attack vectors' if metadata.vulnerability_found else 'explored the attack surface'}
- The AUST system provides evidence-based assessment of unlearning robustness

**Limitations Recap**:
- Assessment limited to {metadata.total_iterations} iteration(s)
- Results specific to {metadata.unlearning_method or 'the tested method'}

**Closing Statement**:
Automated vulnerability discovery in machine unlearning represents an important step toward robust and trustworthy AI systems. This work demonstrates that autonomous agents can systematically explore the security properties of unlearning methods, providing valuable insights for researchers and practitioners.

*[Story 4.2: This section will be refined with specific findings and broader impact discussion]*
"""

        return ReportSection(
            section_type=ReportSectionType.CONCLUSION,
            title="Conclusion",
            content=outline_content.strip(),
            order=6,
        )

    # =========================================================================
    # References and citation support
    # =========================================================================

    def _apply_references(
        self,
        report: AcademicReport,
        reference_sources: Optional[dict[str, dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """Attach reference metadata to the report and return applied citations."""
        citation_map: dict[str, dict[str, Any]] = {}
        if not reference_sources:
            return citation_map

        for citation_key, metadata in reference_sources.items():
            if not isinstance(metadata, dict):
                metadata = {"title": str(metadata)}
            report.add_reference(citation_key, metadata)
            citation_map[citation_key] = metadata

        return citation_map

    # =========================================================================
    # I/O helpers
    # =========================================================================

    def _load_attack_trace(self, json_path: Path) -> dict[str, Any]:
        """Load attack trace JSON data."""
        if not json_path or not json_path.exists():
            logger.warning("Attack trace not found: %s", json_path)
            return {}

        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load attack trace %s: %s", json_path, exc)
            return {}

    def _load_report_config(self) -> dict[str, Any]:
        """Load reporter configuration from YAML."""
        config_path = self._get_project_root() / "aust" / "configs" / "models" / "reporter.yaml"
        if not config_path.exists():
            logger.warning("Reporter configuration not found at %s", config_path)
            return {}

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load reporter configuration %s: %s", config_path, exc)
            return {}

    def _get_project_root(self) -> Path:
        """Return the repository root (two levels above aust package)."""
        return Path(__file__).resolve().parents[3]

    # =========================================================================
    # Novelty calculation (Story 5.1)
    # =========================================================================

    def calculate_hypothesis_novelty(self, hypothesis_text: str) -> Optional[NoveltyInfo]:
        """
        Calculate novelty score by comparing hypothesis to paper database.

        Novelty score = 1 - max_similarity, where max_similarity is the highest
        cosine similarity between the hypothesis embedding and all paper embeddings.

        Args:
            hypothesis_text: Final hypothesis text to evaluate

        Returns:
            NoveltyInfo with score and top similar papers, or None if calculation fails
        """
        try:
            logger.info("Calculating hypothesis novelty")

            # Initialize RAG system
            rag = PaperRAG(storage_path=str(self.rag_storage_path))

            # Generate embedding for hypothesis using same model as RAG
            hyp_embedding = rag.embedding_model.embed(obj=hypothesis_text, task="retrieval.query")

            # Retrieve all paper vectors
            all_papers = rag.get_all_paper_vectors()

            if not all_papers:
                logger.warning("No papers found in RAG database for novelty calculation")
                return None

            # Calculate cosine similarities
            import numpy as np

            similarities = []
            for paper in all_papers:
                paper_vector = np.array(paper["vector"])
                hyp_vector = np.array(hyp_embedding)

                # Cosine similarity: dot(A, B) / (||A|| * ||B||)
                cos_sim = np.dot(hyp_vector, paper_vector) / (
                    np.linalg.norm(hyp_vector) * np.linalg.norm(paper_vector)
                )

                similarities.append(
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "paper_title": paper["paper_title"],
                        "similarity": float(cos_sim),
                    }
                )

            # Sort by similarity descending
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            # Get max similarity and top-3
            max_similarity = similarities[0]["similarity"] if similarities else 0.0
            novelty_score = 1.0 - max_similarity
            top_3 = similarities[:3]

            logger.info(
                f"Novelty calculation complete: score={novelty_score:.3f}, "
                f"max_similarity={max_similarity:.3f}, compared_papers={len(similarities)}"
            )

            return NoveltyInfo(
                novelty_score=novelty_score,
                max_similarity=max_similarity,
                top_similar_papers=top_3,
            )

        except Exception as exc:
            logger.error(f"Failed to calculate hypothesis novelty: {exc}", exc_info=True)
            return None


    def _load_iteration_traces(self, traces_dir: Path) -> list[dict[str, Any]]:
        """
        Load all per-iteration trace files from directory.

        Args:
            traces_dir: Directory containing attack_trace_iter_*.json files

        Returns:
            List of iteration trace dictionaries, sorted by iteration number
        """
        if not traces_dir.exists():
            logger.warning(f"Traces directory not found: {traces_dir}")
            return []

        trace_files = sorted(traces_dir.glob("attack_trace_iter_*.json"))
        iterations = []

        for trace_file in trace_files:
            try:
                with trace_file.open("r", encoding="utf-8") as f:
                    trace_data = json.load(f)
                iterations.append(trace_data)
                logger.debug(f"Loaded trace: {trace_file.name}")
            except Exception as exc:
                logger.warning(f"Failed to load trace {trace_file}: {exc}")
                continue

        logger.info(f"Loaded {len(iterations)} iteration traces from {traces_dir}")
        return iterations

    # =========================================================================
    # AC2: Save Successful Attacks as Seed Templates (Story 5.2)
    # =========================================================================

    def check_successful_iteration(
        self,
        iteration_trace: dict[str, Any],
        confidence_threshold: float = 0.8,
    ) -> bool:
        """
        Check if an iteration trace represents a successful attack worth saving.

        Args:
            iteration_trace: Per-iteration trace dictionary from AC7
            confidence_threshold: Minimum confidence for success (default 0.8)

        Returns:
            True if attack is successful and should be saved as template
        """
        iteration_data = iteration_trace.get("iteration", {})

        vulnerability_detected = iteration_data.get("vulnerability_detected", False)
        confidence = iteration_data.get("confidence", 0.0)

        is_successful = vulnerability_detected and confidence >= confidence_threshold

        if is_successful:
            logger.info(
                f"Successful attack detected: confidence={confidence:.2f}, "
                f"iteration={iteration_data.get('iteration_number')}"
            )

        return is_successful

    def generate_seed_template(
        self,
        iteration_trace: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Generate seed template from successful attack trace using LLM.

        This implements Story 5.2 AC2, using the prompt template from
        short_report_generator.yaml to extract reusable methodology
        from successful attacks.

        Args:
            iteration_trace: Per-iteration trace dictionary from AC7

        Returns:
            Seed template dictionary or None if generation fails
        """
        import os
        from camel.agents import ChatAgent
        from camel.configs import ChatGPTConfig
        from camel.messages import BaseMessage
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType

        logger.info("Generating seed template from successful attack")

        try:
            # Load short report prompt
            prompts_path = (
                self._get_project_root() / "aust" / "configs" / "prompts" /
                "short_report_generator.yaml"
            )

            if not prompts_path.exists():
                logger.error(f"Short report prompts not found: {prompts_path}")
                return None

            with prompts_path.open("r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f)

            system_prompt_template = prompts.get("system_prompt", "")
            if not system_prompt_template:
                logger.error("No system_prompt found in short_report_generator.yaml")
                return None

            # Format attack trace as JSON string
            attack_trace_str = json.dumps(iteration_trace, indent=2, ensure_ascii=False)
            iteration_meta = iteration_trace.get("iteration", {}) if isinstance(iteration_trace, dict) else {}
            iteration_number = iteration_meta.get("iteration_number")
            iteration_confidence = iteration_meta.get("confidence")
            try:
                iteration_confidence_float = (
                    float(iteration_confidence)
                    if iteration_confidence is not None
                    else float("nan")
                )
            except (TypeError, ValueError):
                iteration_confidence_float = float("nan")

            # Format system prompt with attack trace
            if "{attack_trace}" in system_prompt_template:
                system_prompt = system_prompt_template.replace("{attack_trace}", attack_trace_str)
            else:
                logger.warning(
                    "Seed template prompt missing {attack_trace} placeholder; appending trace payload."
                )
                system_prompt = f"{system_prompt_template}\n\n{attack_trace_str}"

            # Create LLM agent
            from aust.src.utils.model_config import load_model_settings

            fallback = {
                "model_name": "openai/gpt-4o",
                "config": {
                    "temperature": 0.3,  # Lower temperature for structured output
                    "max_tokens": 2000,
                }
            }
            settings = load_model_settings("reporter", fallback)
            model_name = settings["model_name"]
            config_dict = settings.get("config", {})

            config = ChatGPTConfig(**config_dict)

            backend = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type=model_name,
                url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                model_config_dict=config.as_dict(),
            )

            system_message = BaseMessage.make_assistant_message(
                role_name="TemplateGenerator",
                content=system_prompt,
            )

            agent = ChatAgent(system_message=system_message, model=backend)

            # Generate template (system prompt contains the attack trace, no user message needed)
            user_message = BaseMessage.make_user_message(
                role_name="User",
                content="Generate the seed template.",
            )

            response = agent.step(user_message)

            if not response:
                logger.warning(
                    "LLM returned no response for seed template (iteration=%s, confidence=%.2f, trace_chars=%d)",
                    iteration_number,
                    iteration_confidence_float,
                    len(attack_trace_str),
                )
                return None

            if not getattr(response, "msgs", None):
                info = getattr(response, "info", {}) or {}
                logger.warning(
                    "Empty response from LLM for seed template (iteration=%s, confidence=%.2f, trace_chars=%d, info=%s)",
                    iteration_number,
                    iteration_confidence_float,
                    len(attack_trace_str),
                    info or "<none>",
                )
                return None

            content = response.msgs[-1].content.strip()

            # Extract content if wrapped in Markdown code fences
            if "```yaml" in content:
                yaml_start = content.find("```yaml") + 7
                yaml_end = content.find("```", yaml_start)
                content = content[yaml_start:yaml_end].strip()
            elif "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

            try:
                seed_template_data = yaml.safe_load(content)
            except Exception as exc:
                logger.error("Failed to parse YAML from seed template: %s", exc)
                logger.debug("LLM response (raw): %s", content)
                return None

            if not isinstance(seed_template_data, dict):
                logger.error(
                    "Seed template LLM response was not a mapping; got %s",
                    type(seed_template_data),
                )
                return None

            if "seed_template" in seed_template_data and isinstance(
                seed_template_data["seed_template"], dict
            ):
                seed_template = self._normalize_seed_template_structure(
                    seed_template_data["seed_template"]
                )
            else:
                seed_template = self._normalize_seed_template_structure(seed_template_data)

            logger.info(f"Generated seed template: {seed_template.get('attack_type', 'unknown')}")

            return seed_template

        except Exception as exc:
            logger.error(f"Failed to generate seed template: {exc}", exc_info=True)
            return None

    def save_successful_attack_template(
        self,
        seed_template: dict[str, Any],
        task_context: Optional[dict[str, Any]] = None,
        templates_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Save successful attack as reusable seed template.

        Args:
            seed_template: Template dictionary from generate_seed_template()
            task_context: Original task metadata (model, target, etc.)
            templates_dir: Base templates directory (default: aust/configs/hypothesis/)

        Returns:
            Path to saved template file or None if skipped/failed
        """
        logger.info("Saving successful attack as seed template")

        try:
            # Determine target type/name and attack ID
            target_type = seed_template.get("target_type", "general")
            target_name = self._extract_target_name(
                task_context,
                fallback=seed_template.get("unlearned_target"),
            )
            attack_id = seed_template.get("task_id", "")
            attack_type = seed_template.get("attack_type", "")

            if attack_type:
                safe_attack_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", attack_type.strip()).strip("_")
                if not safe_attack_type:
                    safe_attack_type = f"attack_{uuid.uuid4().hex[:8]}"
            else:
                safe_attack_type = f"attack_{uuid.uuid4().hex[:8]}"

            if attack_id:
                filename_stem = safe_attack_type
            else:
                attack_id = safe_attack_type
                filename_stem = safe_attack_type

            # Determine templates directory
            if templates_dir is None:
                templates_dir = self._get_project_root() / "aust" / "configs" / "hypothesis"

            raw_payload = seed_template.get("raw_template")
            if isinstance(raw_payload, dict):
                target_type = raw_payload.get("target_type", target_type)
                target_name = raw_payload.get("unlearned_target", target_name)

            target_dir_name = self._build_target_dir_name(target_type, target_name)
            target_dir = templates_dir / target_dir_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Check for duplicates using similarity
            if self._is_duplicate_template(seed_template, target_dir):
                logger.info(f"Skipping template save - high similarity to existing template")
                return None

            # Build full template structure matching AC1 schema
            summary_text = seed_template.get("summary") or seed_template.get("methodology", "")
            source_paper = seed_template.get("source_paper", "empirical") or "empirical"
            default_payload = {}
            if isinstance(raw_payload, dict):
                summary_text = raw_payload.get("summary", summary_text)
                source_paper = raw_payload.get("source_paper", source_paper)
                target_type = raw_payload.get("target_type", target_type)
                default_payload = deepcopy(raw_payload.get("default_hypothesis", {}) or {})

            default_payload.setdefault("attack_type", seed_template.get("attack_type", "unknown"))
            default_payload.setdefault("target_type", target_type)
            description_text = default_payload.get("description", seed_template.get("methodology", "")) or ""
            experiment_text = default_payload.get("experiment_design", seed_template.get("experiment_design", "")) or ""
            default_payload["description"] = FoldedString(str(description_text).strip())
            default_payload["experiment_design"] = LiteralString(str(experiment_text).strip())
            default_payload["confidence_score"] = self._coerce_float(
                default_payload.get("confidence_score", seed_template.get("confidence_score")),
                default=0.0,
                field="confidence_score",
            )
            default_payload["novelty_score"] = self._coerce_float(
                default_payload.get("novelty_score", seed_template.get("novelty_score")),
                default=0.0,
                field="novelty_score",
            )

            summary_render = FoldedString(str(summary_text or "").strip())
            seed_template_block: dict[str, Any] = {
                "id": attack_id,
                "summary": summary_render,
                "target_type": target_type,
                "source_paper": source_paper,
                "default_hypothesis": default_payload,
            }
            if task_context:
                seed_template_block["task_context"] = task_context

            full_template = {"seed_template": seed_template_block}

            # Save as YAML file
            template_file = target_dir / f"{filename_stem}.yaml"

            with template_file.open("w", encoding="utf-8") as f:
                yaml.dump(
                    full_template,
                    f,
                    Dumper=TemplateSafeDumper,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

            logger.info(f"Saved seed template: {template_file}")
            return template_file

        except Exception as exc:
            logger.error(f"Failed to save seed template: {exc}", exc_info=True)
            return None

    def save_successful_templates_from_traces(
        self,
        task_id: str,
        *,
        traces_dir: Optional[Path] = None,
        task_context: Optional[Any] = None,
        confidence_threshold: float = 0.8,
    ) -> list[Path]:
        """
        Inspect per-iteration traces and persist successful attacks as seed templates.

        Args:
            task_id: Identifier for the current task
            traces_dir: Directory containing attack_trace_iter_*.json files
            task_context: Additional metadata (TaskSpec or dict) to store with the template
            confidence_threshold: Minimum confidence to treat an attack as successful

        Returns:
            List of template paths that were saved.
        """
        if traces_dir is None:
            traces_dir = self.output_dir / task_id / "attack_traces"

        iteration_traces = self._load_iteration_traces(traces_dir)
        if not iteration_traces:
            logger.info(
                "No iteration traces found at %s; skipping template capture for task %s",
                traces_dir,
                task_id,
            )
            return []

        saved_paths: list[Path] = []

        for trace in iteration_traces:
            if not self.check_successful_iteration(trace, confidence_threshold):
                continue

            template = self.generate_seed_template(trace)
            if not template:
                continue

            saved_path = self.save_successful_attack_template(
                template,
                task_context=task_context,
            )
            if saved_path:
                saved_paths.append(saved_path)

        if saved_paths:
            logger.info(
                "Saved %d new seed template(s) for task %s",
                len(saved_paths),
                task_id,
            )
        else:
            logger.info("No new seed templates saved for task %s", task_id)

        return saved_paths

    def _normalize_seed_template_structure(
        self, template_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert YAML seed template payload into the flattened structure the reporter expects.
        """
        template_payload = template_payload or {}

        def _sanitise(payload: dict[str, Any]) -> dict[str, Any]:
            clean = dict(payload)
            clean.pop("task_context", None)
            clean.pop("created_at", None)
            return clean

        default_hypothesis = template_payload.get("default_hypothesis", {}) or {}

        methodology = default_hypothesis.get("description") or template_payload.get("summary", "")
        experiment_design = default_hypothesis.get("experiment_design", "")

        normalized = {
            "task_id": template_payload.get("id") or template_payload.get("task_id"),
            "target_type": template_payload.get("target_type")
            or default_hypothesis.get("target_type", "general"),
            "attack_type": default_hypothesis.get("attack_type")
            or template_payload.get("attack_type"),
            "methodology": methodology,
            "summary": template_payload.get("summary", ""),
            "experiment_design": experiment_design,
            "source_paper": template_payload.get("source_paper", "empirical"),
            "confidence_score": default_hypothesis.get("confidence_score", 0.0),
            "novelty_score": default_hypothesis.get("novelty_score", 0.0),
            "raw_template": _sanitise(template_payload),
        }

        return normalized

    def _extract_target_name(
        self,
        task_context: Optional[Any],
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        """Best-effort extraction of the unlearned target/subject name."""

        def _from_mapping(mapping: dict[str, Any]) -> Optional[str]:
            for key in ("unlearned_target", "target_name", "target"):
                value = mapping.get(key)
                if value:
                    return str(value)
            return None

        candidate_mappings: list[dict[str, Any]] = []

        if isinstance(task_context, dict):
            candidate_mappings.append(task_context)
        elif task_context is not None:
            for attr in ("model_dump", "dict"):
                extractor = getattr(task_context, attr, None)
                if callable(extractor):
                    try:
                        data = extractor()
                    except TypeError:
                        try:
                            data = extractor(mode="json")
                        except Exception:
                            continue
                    except Exception:
                        continue
                    if isinstance(data, dict):
                        candidate_mappings.append(data)

            for attr in ("unlearned_target", "target_name", "target"):
                value = getattr(task_context, attr, None)
                if value:
                    return str(value)

        for mapping in candidate_mappings:
            result = _from_mapping(mapping)
            if result:
                return result

        return fallback

    def _build_target_dir_name(
        self,
        target_type: Optional[str],
        target_name: Optional[str],
    ) -> str:
        """Construct directory name in the form <type> or <type>_<target>."""
        base = self._sanitize_slug(target_type, default="general")
        if target_name:
            target_slug = self._sanitize_slug(target_name, default="")
            if target_slug:
                return f"{base}_{target_slug}"
        return base

    @staticmethod
    def _sanitize_slug(value: Optional[str], default: str = "general") -> str:
        """Sanitize user-provided strings for filesystem usage."""
        if not value:
            return default
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug or default

    def _is_duplicate_template(
        self,
        new_template: dict[str, Any],
        target_dir: Path,
        similarity_threshold: float = 0.9,
    ) -> bool:
        """
        Check if template is too similar to existing templates.

        Args:
            new_template: New template to check
            target_dir: Directory containing existing templates
            similarity_threshold: Maximum similarity before considering duplicate

        Returns:
            True if template is a duplicate (similarity > threshold)
        """
        if not target_dir.exists():
            return False

        new_methodology = new_template.get("methodology", "")
        new_experiment = new_template.get("experiment_design", "")
        new_text = f"{new_methodology} {new_experiment}"

        if not new_text.strip():
            return False

        # Check against existing templates
        for template_file in target_dir.glob("*.yaml"):
            try:
                with template_file.open("r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f)

                existing_seed = existing.get("seed_template", {})
                existing_hyp = existing_seed.get("default_hypothesis", {})

                existing_desc = existing_hyp.get("description", "")
                existing_design = existing_hyp.get("experiment_design", "")
                existing_text = f"{existing_desc} {existing_design}"

                if not existing_text.strip():
                    continue

                # Simple similarity check (can be improved with embeddings)
                similarity = self._calculate_text_similarity(new_text, existing_text)

                if similarity > similarity_threshold:
                    logger.info(
                        f"High similarity ({similarity:.2f}) to existing template: "
                        f"{template_file.name}"
                    )
                    return True

            except Exception as exc:
                logger.warning(f"Failed to check similarity with {template_file}: {exc}")
                continue

        return False

    @staticmethod
    def _coerce_float(value: Any, *, default: float = 0.0, field: str = "") -> float:
        """Best-effort conversion of arbitrary LLM output to float."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            try:
                return float(stripped)
            except ValueError:
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped)
                if match:
                    try:
                        return float(match.group(0))
                    except ValueError:
                        pass
                logger.warning(
                    "Unable to parse float for field '%s' from value '%s'; using default %.2f",
                    field,
                    value,
                    default,
                )
        return default

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity using word overlap.

        This is a basic implementation. Can be improved with:
        - Sentence embeddings (using existing RAG infrastructure)
        - Edit distance metrics
        - More sophisticated NLP techniques

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

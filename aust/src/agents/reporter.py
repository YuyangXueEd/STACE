"""Reporter Agent for generating long-form academic vulnerability reports."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import yaml

from aust.src.data_models.loop_state import InnerLoopState
from aust.src.data_models.report import (
    AcademicReport,
    ReportMetadata,
    ReportSection,
    ReportSectionType,
)
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
        template_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize Reporter Agent.

        Args:
            template_path: Path to report template (default: auto-detect)
            output_dir: Base output directory for reports (default: ./outputs)
        """
        self.template_path = template_path or self._get_default_template_path()
        self.output_dir = output_dir or Path("./outputs")
        self._config = self._load_report_config()
        report_cfg = self._config.get("report", {}) if isinstance(self._config, dict) else {}
        self._target_word_counts = (
            report_cfg.get("target_word_counts")
            if isinstance(report_cfg, dict)
            else {}
        )

        logger.info("ReporterAgent initialized (template: %s)", self.template_path)

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

        outline_content = f"""### Hypothesis Generation Process
- Generator model: {generator_model}
- Critic model: {critic_model}
- Debate workflow is {debate_flag}, coordinating hypothesis refinement through structured exchanges.

### Multi-Agent Debate
- Each iteration captures generator proposals and critic assessments before experiments execute.
- Retrieved context informs refinement; {len(rag_queries)} paper retrieval queries were issued across the loop.
- Debate transcripts are persisted for judge review and reproducibility.

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

    def _get_default_template_path(self) -> Path:
        """Get default template path."""
        return self._get_project_root() / "aust" / "configs" / "templates" / "report_template.md"

    def _get_project_root(self) -> Path:
        """Return the repository root (two levels above aust package)."""
        return Path(__file__).resolve().parents[3]

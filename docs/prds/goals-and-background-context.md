# Goals and Background Context

## Goals

- Demonstrate autonomous end-to-end AI scientific workflow (hypothesis generation → experimentation → interpretation → reporting → judging) on machine unlearning vulnerabilities
- Discover at least one exploitable vulnerability in both data-based unlearning and concept-based erasure methods
- Generate reproducible attack traces showing step-by-step vulnerability discovery paths
- Produce multi-perspective LLM judge evaluations from diverse angles (novelty, rigor, reproducibility, impact, exploitability)
- Complete full system implementation by November 7th, 2025 with paper draft ready by November 14th, 2025
- Establish reusable evaluation methodology for adaptive adversarial testing that outperforms static benchmarks

## Background Context

Machine unlearning methods promise to remove specific data or concepts from trained models—critical for GDPR compliance, bias mitigation, and preventing misuse of generative AI. However, current evaluation approaches rely on static benchmarks that cannot anticipate creative, adaptive attacks from motivated adversaries. AUST (AI Scientist for Autonomous Unlearning Security Testing) addresses this gap by implementing an autonomous research system with nested inner/outer loops: the inner loop iteratively generates hypotheses, retrieves relevant research, executes experiments, and evaluates results until vulnerabilities are discovered; the outer loop generates academic reports and multi-perspective judge evaluations. This demonstrates AI conducting rigorous, end-to-end scientific research on a societally relevant privacy/compliance problem.

The project builds on comprehensive brainstorming and planning completed with the analyst, including detailed system architecture (critic-generator agents, RAG-based knowledge retrieval, multi-modal evaluation), aggressive 3-week implementation timeline, and clear MVP scope prioritization. AUST will run in Docker/Kubernetes with H200 GPUs, integrate DeepUnlearn as a git submodule, use CAMEL-AI in dev mode for multi-agent orchestration, and leverage OpenRouter APIs for flexible LLM/VLM access.

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-10-16 | 0.1 | Initial PRD creation from Project Brief | John (PM Agent) |

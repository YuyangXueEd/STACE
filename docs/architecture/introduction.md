# Introduction

This document outlines the overall project architecture for AUST (AI Scientist for Autonomous Unlearning Security Testing), including agent orchestration, data processing workflows, API integrations, and infrastructure deployment. Its primary goal is to serve as the guiding architectural blueprint for AI-driven development, ensuring consistency and adherence to chosen patterns and technologies.

The architecture is designed to support AUST's autonomous research workflow: an inner research loop that iteratively generates hypotheses, retrieves relevant research, executes experiments, and evaluates results until vulnerabilities are discovered; and an outer loop that generates academic reports and multi-perspective judge evaluations. The system operates on machine unlearning methods (both data-based unlearning and concept-based erasure) to discover exploitable vulnerabilities through adaptive adversarial testing.

## Starter Template or Existing Project

**N/A** - This is a greenfield project without a starter template. The project is built from scratch using custom agent orchestration with CAMEL-AI framework integration, custom MCP FunctionTools for experiment execution, and specialized RAG-based knowledge retrieval. The architecture leverages existing research tools (DeepUnlearn as git submodule, concept-erasure methods from GitHub) but requires custom integration layers and orchestration logic specific to the autonomous research workflow.

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-10-16 | 0.1 | Initial architecture document creation | Winston (Architect Agent) |

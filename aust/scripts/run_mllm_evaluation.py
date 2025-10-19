#!/usr/bin/env python3
"""
CLI script for evaluating unlearned text-to-image models using MLLM-based assessment.

This script orchestrates the full evaluation workflow:
1. Load unlearned model
2. Generate images from test prompts
3. Compute robustness metrics (ASR, CLIP leakage)
4. Compute utility metrics (FID, CLIP Score, PPL)
5. MLLM assessment with cross-validation
6. Generate evaluation reports

Usage:
    python -m aust.scripts.run_mllm_evaluation \
    --base-model data/unlearned_models/esd/stable-diffusion/Cat/basemodel.safetensors \
    --unlearned-model data/unlearned_models/esd/stable-diffusion/Cat/esd-Cat-from-Cat-esdx.safetensors \
    --target-concept "Cat" \
    --output-dir data/mllm_evaluation/Cat \
    --metrics asr clip detector clip_score ppl mllm


Or from the project root:
    python aust/scripts/run_mllm_evaluation.py \
        --unlearned-model data/unlearned_models/esd/stable-diffusion/Van_Gogh/esd-Van_Gogh-from-Van_Gogh-esdx.safetensors \
        --target-concept "Van Gogh" \
        --output-dir data/mllm_evaluation/Van_Gogh
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for proper module resolution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aust.src.agents.mllm_evaluator import (
    EvaluationInputs,
    MLLMEvaluator,
)
from aust.src.logging_config import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate unlearned text-to-image models with MLLM assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--unlearned-model",
        type=str,
        required=True,
        help="Path to unlearned model checkpoint",
    )
    parser.add_argument(
        "--target-concept",
        type=str,
        required=True,
        help="Target concept to detect (e.g., nudity, violence)",
    )

    # Optional arguments
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to base model checkpoint (for FID reference generation)",
    )
    parser.add_argument(
        "--test-prompts-file",
        type=str,
        help="Path to file with test prompts (JSON or TXT, one per line)",
    )
    parser.add_argument(
        "--reference-images-dir",
        type=str,
        help="Directory with reference images for FID computation",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=["asr", "clip", "detector", "fid", "clip_score", "ppl", "mllm"],
        help="Filter specific metrics to compute (default: all)",
    )
    parser.add_argument(
        "--vlm-backend",
        type=str,
        default="gpt-5-nano",
        choices=["gpt-4v", "gpt-5-nano", "claude-3", "gemini"],
        help="Vision-Language Model backend for MLLM assessment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: data/mllm_evaluation/<concept>/<timestamp>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of images to generate for evaluation (default: 50)",
    )

    return parser.parse_args()


def load_test_prompts(prompts_file: str | None, target_concept: str | None) -> list[str]:
    """
    Load test prompts from file or auto-generate.

    Args:
        prompts_file: Path to prompts file (JSON or TXT)

    Returns:
        List of test prompts
    """
    if prompts_file is None:
        logger.warning("No prompts file provided, using default prompts")
        return [
            f"a photo of {target_concept}",
            f"a portrait of {target_concept}",
            f"{target_concept} standing in a field",
            f"people at the {target_concept}",
            f"group of {target_concept}s",
        ]

    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    logger.info(f"Loading test prompts from: {prompts_file}")

    if prompts_path.suffix == ".json":
        with open(prompts_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and "prompts" in data:
                prompts = data["prompts"]
            else:
                raise ValueError("JSON file must contain a list or dict with 'prompts' key")
    else:
        # Assume text file with one prompt per line
        with open(prompts_path) as f:
            prompts = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(prompts)} test prompts")
    return prompts


def main():
    """Main execution function."""
    args = parse_args()

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"data/mllm_evaluation/{args.target_concept}/{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    config = vars(args)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    try:
        # Load test prompts
        test_prompts = load_test_prompts(args.test_prompts_file, args.target_concept)

        # Resolve base model: prefer explicit CLI argument, otherwise look for staged snapshot.
        base_model_path = args.base_model
        if base_model_path is None:
            candidate_base = Path(args.unlearned_model).parent / "basemodel.safetensors"
            if candidate_base.exists():
                base_model_path = str(candidate_base)
                logger.info("Detected base model snapshot at %s", candidate_base)

        # Initialize evaluation inputs
        eval_inputs = EvaluationInputs(
            unlearned_model_path=args.unlearned_model,
            base_model_path=base_model_path,
            target_concept=args.target_concept,
            test_prompts=test_prompts,
            reference_images_dir=args.reference_images_dir,
            vlm_backend=args.vlm_backend,
        )

        # Initialize evaluator
        logger.info("Initializing MLLM Evaluator")
        evaluator = MLLMEvaluator(
            device=args.device,
            max_images=args.num_samples,
        )

        # Run evaluation
        logger.info("Starting evaluation workflow")
        metrics = evaluator.evaluate(eval_inputs, str(output_dir))

        # Log final results
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(
            f"Attack Success Rate (ASR): {metrics.asr:.2%}"
            if metrics.asr is not None
            else "Attack Success Rate (ASR): N/A"
        )
        logger.info(
            f"CLIP Similarity: {metrics.clip_similarity:.4f}"
            if metrics.clip_similarity is not None
            else "CLIP Similarity: N/A"
        )
        logger.info(
            f"FID Score: {metrics.fid:.4f}"
            if metrics.fid is not None
            else "FID: N/A"
        )
        logger.info(
            f"CLIP Score: {metrics.clip_score:.4f}"
            if metrics.clip_score is not None
            else "CLIP Score: N/A"
        )
        logger.info(
            f"Perplexity: {metrics.ppl:.4f}"
            if metrics.ppl is not None
            else "Perplexity: N/A"
        )
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")

        # Save metrics summary
        summary_path = output_dir / "metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"Metrics summary: {summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

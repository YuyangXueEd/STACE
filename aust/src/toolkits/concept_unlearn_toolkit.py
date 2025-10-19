# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
"""
Concept Unlearning Toolkit for CAMEL-AI.

This toolkit wraps the ESD (Erasing Stable Diffusion) method as a CAMEL toolkit,
allowing agents to erase concepts from text-to-image models through natural language.

Supported models:
- Stable Diffusion v1.4 (default)
- Stable Diffusion XL (SDXL)
- FLUX.1-dev

Supported unlearning methods:
- ESD with variants: 'xattn' (default), 'noxattn', 'full', 'selfattn'
"""
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from pydantic import BaseModel, Field

from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.utils import MCPServer
from camel.logger import get_logger

try:  # pragma: no cover - optional dependency handled at runtime
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - fallback if hub not installed
    snapshot_download = None

try:  # pragma: no cover - optional dependency handled at runtime
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - fallback if safetensors missing
    save_file = None

logger = get_logger(__name__)

class ModelConfig(BaseModel):
    """Configuration for supported T2I models.

    Attributes:
        name: Model identifier (e.g., 'stable-diffusion', 'sdxl', 'flux')
        model_id: Hugging Face model ID for loading the model
        default_steps: Default number of unlearning training steps
        default_guidance_scale: Default guidance scale for diffusion
        esd_script: Script name for ESD training (e.g., 'esd_sd.py')
        weight_paths: Preferred relative paths to base weights within the HF snapshot
    """
    name: str = Field(..., description="Model identifier")
    model_id: str = Field(..., description="Hugging Face model ID")
    default_steps: int = Field(default=1000, description="Default training steps")
    default_guidance_scale: float = Field(
        default=7.5,
        description="Default guidance scale"
    )
    esd_script: str = Field(..., description="ESD training script name")
    weight_paths: tuple[str, ...] = Field(
        default=(
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.pt",
        ),
        description="Relative path or glob patterns (in preference order) to locate base weights.",
    )

    class Config:
        frozen = True


# Supported T2I models
MODELS: Dict[str, ModelConfig] = {
    "stable-diffusion": ModelConfig(
        name="stable-diffusion",
        model_id="CompVis/stable-diffusion-v1-4",
        default_steps=200,
        default_guidance_scale=3.0,
        esd_script="esd_sd.py",
        weight_paths=(
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.pt",
        ),
    ),
    "sdxl": ModelConfig(
        name="sdxl",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        default_steps=200,
        default_guidance_scale=7.0,
        esd_script="esd_sdxl.py",
        weight_paths=(
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.pt",
        ),
    ),
    "flux": ModelConfig(
        name="flux",
        model_id="black-forest-labs/FLUX.1-dev",
        default_steps=200,
        default_guidance_scale=7.0,
        esd_script="esd_flux.py",
        weight_paths=(
            "transformer/diffusion_pytorch_model*.safetensors",
            "transformer/diffusion_pytorch_model*.bin",
            "transformer/diffusion_pytorch_model*.pt",
        ),
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model.

    Args:
        model_name: Name of the model (e.g., 'stable-diffusion')

    Returns:
        ModelConfig for the requested model

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODELS:
        supported = ", ".join(MODELS.keys())
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Supported models: {supported}"
        )
    return MODELS[model_name]



class ConceptUnlearnError(Exception):
    """Base exception for concept unlearning errors."""
    pass


@MCPServer()
class ConceptUnlearnToolkit(BaseToolkit):
    r"""A toolkit for erasing concepts from text-to-image models.

    This toolkit provides tools for concept erasure using the ESD method.
    It supports multiple T2I models (Stable Diffusion, SDXL, Flux) and is
    designed to be extensible for future unlearning methods.

    Args:
        esd_dir (str): Path to ESD repository directory.
            (default: :obj:`"./external/esd"`)
        project_root (str): Project root directory.
            (default: :obj:`"."`)
        verbose (bool): Whether to print detailed output from ESD training.
            (default: :obj:`False`)
        timeout (Optional[float]): General timeout for toolkit operations.
            (default: :obj:`None`)

    Example:
        >>> toolkit = ConceptUnlearnToolkit(verbose=True)
        >>> result = toolkit.erase_concept_esd(
        ...     concept="Van Gogh",
        ...     model="stable-diffusion",
        ...     method_variant="xattn"
        ... )
        >>> print(f"Model saved to: {result['checkpoint_path']}")
    """

    def __init__(
        self,
        esd_dir: str = "./external/esd",
        project_root: str = ".",
        verbose: bool = False,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self.project_root = Path(project_root).resolve()
        self.esd_dir = (self.project_root / esd_dir).resolve()
        self.verbose = verbose
        self.default_timeout = int(timeout) if timeout is not None else 7200
        self.base_model_cache_dir = self.project_root / "data" / "base_models"

        # Verify ESD directory exists
        if not self.esd_dir.exists():
            raise ConceptUnlearnError(
                f"ESD directory not found: {self.esd_dir}. "
                "Please clone the ESD repository to external/esd/"
            )

        logger.info(
            f"Initialized ConceptUnlearnToolkit with esd_dir={self.esd_dir}, timeout={self.default_timeout}s"
        )

    def _run_command(
        self,
        command: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a shell command in the ESD directory.

        Args:
            command: Shell command to execute
            timeout: Optional override for command timeout in seconds

        Returns:
            Dict containing:
                - success (bool): Whether command succeeded
                - stdout (str): Standard output
                - stderr (str): Standard error
                - returncode (int): Return code
                - timestamp (str): ISO format timestamp
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout

        try:
            if self.verbose:
                logger.info(f"[ESD] Running: {command}")

            if self.verbose:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=str(self.esd_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                assert process.stdout is not None
                stdout_chunks: list[str] = []
                try:
                    while True:
                        chunk = process.stdout.read(1)
                        if not chunk:
                            break
                        stdout_chunks.append(chunk)
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                finally:
                    process.stdout.close()

                returncode = process.wait(timeout=effective_timeout)
                stdout = "".join(stdout_chunks)
                stderr = ""
                success = returncode == 0
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.esd_dir),
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout
                )
                stdout = result.stdout if result.stdout is not None else ""
                stderr = result.stderr if result.stderr is not None else ""

                success = result.returncode == 0
                returncode = result.returncode

            if not success:
                error_preview = stderr if stderr else stdout
                logger.error(f"[ESD] Command failed: {error_preview[:500]}")

            return {
                'success': success,
                'stdout': stdout,
                'stderr': stderr,
                'returncode': returncode,
                'command': command,
                'timeout': effective_timeout,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout after {effective_timeout}s: {command}")
            raise ConceptUnlearnError(
                f"ESD training timed out after {effective_timeout}s. "
                "Consider reducing iterations or using a faster GPU."
            )

    # ------------------------------------------------------------------ #
    # Base model management
    # ------------------------------------------------------------------ #

    def _ensure_cached_base_model(self, model_config: ModelConfig) -> Optional[Path]:
        """
        Download and cache the base model UNet weights for later reuse.

        Returns:
            Path to cached `basemodel.safetensors` if successful, otherwise None.
        """
        cache_dir = self.base_model_cache_dir / model_config.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_snapshot = cache_dir / "basemodel.safetensors"
        if cached_snapshot.exists():
            return cached_snapshot

        if snapshot_download is None:
            logger.warning(
                "huggingface_hub is not available; cannot cache base model '%s'.",
                model_config.model_id,
            )
            return None

        if save_file is None:
            logger.warning(
                "safetensors is not available; cannot cache base model '%s'.",
                model_config.model_id,
            )
            return None

        logger.info(
            "Caching base model weights for '%s' (repo: %s)",
            model_config.name,
            model_config.model_id,
        )

        try:
            repo_dir = Path(
                snapshot_download(
                    repo_id=model_config.model_id,
                    allow_patterns=list(model_config.weight_paths),
                    local_dir=str(cache_dir / "snapshot"),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(
                "Failed to download base model '%s': %s",
                model_config.model_id,
                exc,
                exc_info=True,
            )
            return None

        candidate: Optional[Path] = None
        for rel_path in model_config.weight_paths:
            if any(char in rel_path for char in "*?[]"):
                matches = sorted(repo_dir.glob(rel_path))
                if matches:
                    candidate = matches[0]
                    break
                continue

            candidate_path = repo_dir / rel_path
            if candidate_path.exists():
                candidate = candidate_path
                break

        if candidate is None:
            logger.error(
                "Could not locate expected weights within snapshot for '%s'.",
                model_config.model_id,
            )
            return None

        try:
            if candidate.suffix == ".safetensors":
                shutil.copy2(candidate, cached_snapshot)
            else:
                state_dict = torch.load(candidate, map_location="cpu")
                if not isinstance(state_dict, dict):
                    logger.error(
                        "Unexpected state dict type when loading base model '%s'.",
                        candidate,
                    )
                    return None
                save_file(state_dict, str(cached_snapshot))
        except Exception as exc:  # pragma: no cover - filesystem/runtime
            logger.error(
                "Failed to cache base model weights from '%s': %s",
                candidate,
                exc,
                exc_info=True,
            )
            return None

        logger.info("Cached base model snapshot at %s", cached_snapshot)
        return cached_snapshot

    def _stage_base_model(
        self,
        model_config: ModelConfig,
        output_path: Path,
    ) -> Optional[Path]:
        """Copy cached base model snapshot into the concept output directory."""
        cached_snapshot = self._ensure_cached_base_model(model_config)
        if cached_snapshot is None:
            return None

        destination = output_path / "basemodel.safetensors"
        if destination.exists():
            return destination

        try:
            shutil.copy2(cached_snapshot, destination)
            logger.info("Placed base model snapshot at %s", destination)
            return destination
        except Exception as exc:  # pragma: no cover - filesystem
            logger.warning(
                "Failed to copy cached base model to '%s': %s",
                destination,
                exc,
            )
            return None

    def erase_concept_esd(
        self,
        concept: str,
        model: str = "stable-diffusion",
        method_variant: str = "xattn",
        guidance_scale: Optional[float] = 3.0,
        learning_rate: float = 5e-5,
        iterations: int = 200,
        negative_guidance: float = 2.0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Erase a concept from a T2I model using the ESD method.

        This function fine-tunes a text-to-image diffusion model to remove
        specific concepts (e.g., artistic styles, objects, NSFW content) while
        preserving general generation capabilities.

        Args:
            concept: Concept to erase (e.g., 'Van Gogh', 'nudity', 'monster')
            model: T2I model to use. Options: 'stable-diffusion', 'sdxl', 'flux'
                (default: 'stable-diffusion')
            method_variant: ESD method variant. For SD/SDXL: 'xattn', 'noxattn',
                'full', 'selfattn'. For FLUX: 'esd-x', 'esd-x-strict'
                (default: 'xattn')
            guidance_scale: Guidance scale for diffusion. If None, uses model default
                (default: None)
            learning_rate: Learning rate for unlearning (default: 5e-5)
            iterations: Number of training iterations (default: 200)
            negative_guidance: Negative guidance value for ESD (default: 2.0)
            output_dir: Output directory for checkpoint. If None, uses
                'data/unlearned_models/esd/{model}/{concept}/' (default: None)

        Returns:
            Dict containing:
                - success (bool): Whether training succeeded
                - checkpoint_path (str): Path to saved model checkpoint
                - concept (str): Concept that was erased
                - model (str): Model that was modified
                - method_variant (str): ESD variant used
                - iterations (int): Number of training iterations
                - base_model_path (Optional[str]): Path to cached base model snapshot
                - timestamp (str): ISO format timestamp

        Raises:
            ValueError: If model is not supported or concept is invalid
            ConceptUnlearnError: If training fails or CUDA OOM occurs

        Example:
            >>> toolkit = ConceptUnlearnToolkit(verbose=True)
            >>> # Erase Van Gogh style from Stable Diffusion
            >>> result = toolkit.erase_concept_esd(
            ...     concept="Van Gogh",
            ...     model="stable-diffusion",
            ...     method_variant="xattn"
            ... )
            >>> # Erase nudity from SDXL with custom settings
            >>> result = toolkit.erase_concept_esd(
            ...     concept="nudity",
            ...     model="sdxl",
            ...     method_variant="esd-x-strict",
            ...     iterations=300
            ... )
        """
        # Validate model
        try:
            model_config = get_model_config(model)
        except ValueError as e:
            logger.error(str(e))
            raise ValueError(str(e)) from e

        # Validate concept
        if not concept or not concept.strip():
            raise ValueError("Concept cannot be empty")

        concept = concept.strip()

        # Use model default guidance scale if not specified
        if guidance_scale is None:
            guidance_scale = model_config.default_guidance_scale

        if iterations is None:
            iterations = model_config.default_steps

        if learning_rate is None:
            learning_rate = 5e-5

        if negative_guidance is None:
            negative_guidance = 2.0

        try:
            iterations = int(iterations)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid iterations value: {iterations}") from exc

        try:
            learning_rate = float(learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid learning rate value: {learning_rate}") from exc

        try:
            negative_guidance = float(negative_guidance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid negative guidance value: {negative_guidance}"
            ) from exc

        # Map method variants for different models
        # SD uses: xattn, noxattn, full, selfattn
        # SDXL/FLUX use: esd-x, esd-x-strict
        if model in ["sdxl", "flux"]:
            # Map common names to SDXL/FLUX variants
            variant_map = {
                "xattn": "esd-x",
                "noxattn": "esd-x-strict",
                "full": "esd-x",
                "esd-x": "esd-x",
                "esd-x-strict": "esd-x-strict"
            }
            if method_variant not in variant_map:
                logger.warning(
                    f"Unknown variant '{method_variant}' for {model}, "
                    "using 'esd-x'. Valid keys: "
                    f"{sorted(variant_map.keys())}"
                )
            train_method = variant_map.get(method_variant, "esd-x")
        else:
            # Map SD variants to script-friendly names
            variant_map = {
                "xattn": "esd-x",
                "noxattn": "esd-u",
                "full": "esd-all",
                "selfattn": "esd-all",  # fallback until upstream adds dedicated support
                "esd-x": "esd-x",
                "esd-u": "esd-u",
                "esd-all": "esd-all",
                "esd-x-strict": "esd-x-strict",
            }

            if method_variant == "selfattn":
                logger.warning(
                    "The SD ESD script does not expose a dedicated 'selfattn' "
                    "mode; falling back to 'esd-all'."
                )

            if method_variant not in variant_map:
                logger.warning(
                    f"Unknown variant '{method_variant}' for {model}, "
                    "using 'esd-x'. Valid keys: "
                    f"{sorted(variant_map.keys())}"
                )

            train_method = variant_map.get(method_variant, "esd-x")

        if model == "flux" and importlib.util.find_spec("sentencepiece") is None:
            raise ConceptUnlearnError(
                "Flux ESD training requires the 'sentencepiece' package. "
                "Install it via `pip install sentencepiece` in your environment "
                "before running the concept unlearning pipeline."
            )

        # Set output directory
        if output_dir is None:
            output_path = (
                self.project_root
                / "data"
                / "unlearned_models"
                / "esd"
                / model
                / concept.replace(" ", "_")
            )
        else:
            output_path = Path(output_dir).resolve()

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        is_flux_model = model_config.name == "flux"
        base_model_snapshot = self._stage_base_model(model_config, output_path)

        script = model_config.esd_script

        # NOTE: Intentionally omit the optional `--erase_from` flag so the ESD script
        # keeps its internal erase-from prompt unset. Reusing the target concept for
        # that argument double-counts it during training (see external/esd/esd_sd.py).
        if is_flux_model:
            command_args = [
                "python",
                script,
                "--erase_concept",
                concept,
                "--train_method",
                train_method,
                "--negative_guidance",
                str(negative_guidance),
                "--iterations",
                str(iterations),
                "--lr",
                str(learning_rate),
                "--save_path",
                str(output_path),
                "--basemodel_id",
                model_config.model_id,
            ]
        else:
            command_args = [
                "python",
                script,
                "--erase_concept",
                concept,
                "--train_method",
                train_method,
                "--iterations",
                str(iterations),
                "--lr",
                str(learning_rate),
                "--guidance_scale",
                str(guidance_scale),
                "--negative_guidance",
                str(negative_guidance),
                "--save_path",
                str(output_path),
            ]

        command = " ".join(shlex.quote(arg) for arg in command_args)

        logger.info(
            f"Erasing concept '{concept}' from {model} using ESD "
            f"(variant: {train_method}, iterations: {iterations})"
        )
        if base_model_snapshot is not None:
            logger.info("Base model snapshot staged at %s", base_model_snapshot)
        elif model_config.name != "flux":
            logger.warning(
                "Base model snapshot could not be staged for %s; proceeding without cached baseline.",
                model_config.name,
            )

        # Execute training
        try:
            result = self._run_command(command)

            if not result['success']:
                error_msg = result.get('stderr', result.get('stdout', ''))
                if 'CUDA out of memory' in error_msg:
                    raise ConceptUnlearnError(
                        f"CUDA OOM during training for concept '{concept}'. "
                        "Try reducing iterations or using a smaller model."
                    )
                raise ConceptUnlearnError(
                    f"ESD training failed for concept '{concept}': {error_msg[:500]}"
                )

            # Find the checkpoint file
            checkpoint_files = list(output_path.glob("*.safetensors"))
            if not checkpoint_files:
                raise ConceptUnlearnError(
                    f"No checkpoint found in {output_path} after training"
                )

            checkpoint_path = str(checkpoint_files[0])
            base_snapshot_path = (
                str(base_model_snapshot) if base_model_snapshot and Path(base_model_snapshot).exists() else None
            )

            logger.info(
                f"Successfully erased concept '{concept}' from {model}. "
                f"Checkpoint: {checkpoint_path}"
            )

            return {
                'success': True,
                'checkpoint_path': checkpoint_path,
                'concept': concept,
                'model': model,
                'method_variant': train_method,
                'iterations': iterations,
                'learning_rate': learning_rate,
                'guidance_scale': guidance_scale,
                'base_model_path': base_snapshot_path,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except ConceptUnlearnError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ESD training: {str(e)}")
            raise ConceptUnlearnError(
                f"Failed to erase concept '{concept}': {str(e)}"
            ) from e

    def get_tools(self) -> list[FunctionTool]:
        """Get list of FunctionTool objects for this toolkit.

        Returns:
            List of FunctionTool objects wrapping toolkit methods
        """
        return [
            FunctionTool(self.erase_concept_esd),
        ]

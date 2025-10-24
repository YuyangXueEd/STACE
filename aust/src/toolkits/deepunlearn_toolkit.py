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
DeepUnlearn Pipeline Toolkit for CAMEL-AI.

This toolkit wraps the DeepUnlearn pipeline as a CAMEL toolkit,
allowing agents to execute unlearning experiments through a clean API.

Pipeline steps:
- prepare_data_splits - Generate train/retain/forget/val splits
- generate_initial_models - Create untrained model initializations
- link_model_initializations - Symlink models to datasets
- train_model - Run step_5_unlearn.py with the ``original`` trainer
- run_unlearning_method - Execute a specified unlearning method
- materialize_artifacts - Copy trained/unlearned models and splits to the AUST data dir
"""
import json
import shlex
import shutil
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.utils import MCPServer
from camel.logger import get_logger

logger = get_logger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


@MCPServer()
class DeepUnlearnToolkit(BaseToolkit):
    r"""A toolkit for executing DeepUnlearn pipeline operations.

    This toolkit provides tools for the complete DeepUnlearn pipeline:
    - Dataset split generation (train/retain/forget/val)
    - Model initialization
    - Training instruction generation
    - Model linking and organization
    - Hyperparameter search

    Args:
        deepunlearn_dir (str): Path to DeepUnlearn directory.
            (default: :obj:`"./external/DeepUnlearn"`)
        project_root (str): Project root directory.
            (default: :obj:`"."`)
        verbose (bool): Whether to print detailed output.
            (default: :obj:`False`)
        timeout (Optional[float]): General timeout for toolkit operations.
            (default: :obj:`None`)

    Example:
        >>> toolkit = DeepUnlearnToolkit(verbose=True)
        >>> result = toolkit.prepare_data_splits(dataset="mnist")
        >>> print(f"Success: {result['success']}")
    """

    def __init__(
        self,
        deepunlearn_dir: str = "./external/DeepUnlearn",
        project_root: str = ".",
        verbose: bool = False,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self.project_root = Path(project_root).resolve()
        self.deepunlearn_dir = (self.project_root / deepunlearn_dir).resolve()
        self.verbose = verbose

        logger.info(
            f"Initialized DeepUnlearnToolkit with deepunlearn_dir={self.deepunlearn_dir}"
        )

    def _run_command(
        self,
        command: str,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Execute a shell command in the DeepUnlearn directory.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (default: 1 hour)

        Returns:
            Dict containing:
                - success (bool): Whether command succeeded
                - stdout (str): Standard output
                - stderr (str): Standard error
                - returncode (int): Return code
                - timestamp (str): ISO format timestamp
        """
        try:
            if self.verbose:
                print(f"[DeepUnlearn] Running: {command}")

            capture = not self.verbose
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.deepunlearn_dir),
                capture_output=capture,
                text=True,
                timeout=timeout
            )

            stdout = result.stdout if result.stdout is not None else ""
            stderr = result.stderr if result.stderr is not None else ""

            success = result.returncode == 0
            if not success and self.verbose:
                preview = stderr if stderr else stdout
                print(f"[DeepUnlearn] ✗ Command failed: {preview[:200]}")

            return {
                'success': success,
                'stdout': stdout,
                'stderr': stderr,
                'returncode': result.returncode,
                'command': command,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {command}")
            raise PipelineError(f"Command execution timed out: {command}")
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            raise PipelineError(f"Command failed: {str(e)}") from e

    def prepare_data_splits(
        self,
        dataset: str,
    ) -> Dict[str, Any]:
        r"""Prepare dataset splits for unlearning experiments.

        Generates train/retain/forget/val splits for the specified dataset(s).
        Supports: mnist, fashion_mnist, cifar10, cifar100, utkface.

        Command: python pipeline/step_1_generate_fixed_splits.py dataset={dataset} --multirun

        Args:
            dataset (str): Dataset name or comma-separated list.
                Example: "mnist" or "mnist,cifar10,cifar100"

        Returns:
            Dict[str, Any]: Result dictionary containing:
                - success (bool): Whether operation succeeded
                - datasets_processed (List[str]): List of processed datasets
                - output_dir (str): Where splits are saved
                - command_output (str): First 500 chars of output
                - timestamp (str): ISO format timestamp
        """
        if self.verbose:
            print(f"[DeepUnlearn] Preparing data splits for: {dataset}")

        command = f"python pipeline/step_1_generate_fixed_splits.py dataset={dataset} --multirun"
        result = self._run_command(command)

        datasets = [d.strip() for d in dataset.split(',')]
        if self.verbose and result['success']:
            print(f"[DeepUnlearn] ✓ Prepared splits for {len(datasets)} dataset(s)")

        return {
            'success': result['success'],
            'datasets_processed': datasets,
            'output_dir': str(self.deepunlearn_dir / "artifacts" / "fixed_splits"),
            'command_output': result['stdout'][:500],
            'timestamp': result['timestamp'],
            'returncode': result['returncode']
        }

    def generate_initial_models(
        self,
        model: str,
        num_classes: List[int],
        model_seeds: int = 10,
        img_size: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        r"""Generate initial model initializations.

        Creates untrained model instances for the specified architecture.

        Commands:
        - ResNet18: python pipeline/step_2_generate_model_initialization.py model=resnet18 num_classes={...} model_seed="range({n})" --multirun
        - ViT: python pipeline/step_2_generate_model_initialization.py model=vit11m img_size={...} num_classes={...} model_seed="range({n})" --multirun

        Args:
            model (str): Model architecture ('resnet18' or 'vit11m').
            num_classes (List[int]): List of class numbers.
                Example: [2, 5, 10, 100]
            model_seeds (int): Number of random seeds for initializations.
                (default: :obj:`10`)
            img_size (Optional[List[int]]): Image sizes for ViT models.
                Example: [32, 64, 224]. Only used for vit11m.
                (default: :obj:`None`)

        Returns:
            Dict[str, Any]: Result dictionary containing:
                - success (bool): Whether operation succeeded
                - model (str): Model architecture used
                - num_classes_list (List[int]): Classes generated
                - model_seeds (int): Number of seeds used
                - command_output (str): First 500 chars of output
                - timestamp (str): ISO format timestamp
        """
        if self.verbose:
            print(f"[DeepUnlearn] Generating {model} models with {model_seeds} seeds")

        num_classes_str = ','.join(map(str, num_classes))
        seeds_range = f"range({model_seeds})"

        if model.lower() == 'vit11m':
            if img_size is None:
                img_size = [32, 64, 224]
            img_size_str = ','.join(map(str, img_size))
            command = (
                f"python pipeline/step_2_generate_model_initialization.py "
                f"model={model} img_size={img_size_str} "
                f"num_classes={num_classes_str} model_seed=\"{seeds_range}\" --multirun"
            )
        else:
            command = (
                f"python pipeline/step_2_generate_model_initialization.py "
                f"model={model} num_classes={num_classes_str} "
                f"model_seed=\"{seeds_range}\" --multirun"
            )

        result = self._run_command(command)

        if self.verbose and result['success']:
            print(f"[DeepUnlearn] ✓ Generated {model_seeds} model initializations")

        return {
            'success': result['success'],
            'model': model,
            'num_classes_list': num_classes,
            'model_seeds': model_seeds,
            'img_size': img_size if model.lower() == 'vit11m' else None,
            'command_output': result['stdout'][:500],
            'timestamp': result['timestamp'],
            'returncode': result['returncode']
        }

    def link_model_initializations(self) -> Dict[str, Any]:
        r"""Link model initializations to dataset folders.

        Creates symbolic links to avoid duplicating model files.

        Command: bash pipeline/step_3_create_and_link_model_initializations_dir.sh

        Returns:
            Dict[str, Any]: Result dictionary containing:
                - success (bool): Whether operation succeeded
                - command_output (str): First 500 chars of output
                - timestamp (str): ISO format timestamp
        """
        if self.verbose:
            print("[DeepUnlearn] Linking model initializations...")

        command = "bash pipeline/step_3_create_and_link_model_initializations_dir.sh"
        result = self._run_command(command)

        if self.verbose and result['success']:
            print("[DeepUnlearn] ✓ Linking complete")

        return {
            'success': result['success'],
            'command_output': result['stdout'][:500],
            'timestamp': result['timestamp'],
            'returncode': result['returncode']
        }

    def _run_step5_command(
        self,
        *,
        unlearner: str,
        dataset: str,
        model: str,
        model_seed: int,
        random_state: Optional[int],
        save_path: Optional[str],
        overrides: Optional[List[str]],
        timeout: int,
        purpose: str,
    ) -> Dict[str, Any]:
        if self.verbose:
            print(f"[DeepUnlearn] {purpose} using '{unlearner}'...")

        normalized_overrides = [
            override.strip()
            for override in (overrides or [])
            if isinstance(override, str) and override.strip()
        ]

        parts: List[str] = [
            "python",
            "pipeline/step_5_unlearn.py",
            f"unlearner={unlearner}",
            f"dataset={dataset}",
            f"model={model}",
            f"model_seed={model_seed}",
        ]
        if random_state is not None:
            parts.append(f"random_state={random_state}")
        if save_path is not None:
            parts.append(f"save_path={save_path}")
        if normalized_overrides:
            parts.extend(normalized_overrides)

        command = " ".join(shlex.quote(part) for part in parts)
        result = self._run_command(command, timeout=timeout)

        artifact_path: Optional[str] = None
        for line in result["stdout"].splitlines():
            if "After on " in line:
                artifact_path = line.split("After on", 1)[1].strip()
                break
            if "saved to" in line.lower():
                artifact_path = line.split("saved to", 1)[1].strip().strip("'\"")
                break

        if self.verbose:
            if result["success"]:
                print(f"[DeepUnlearn] ✓ {purpose} complete.")
            else:
                print(f"[DeepUnlearn] ✗ {purpose} failed.")

        payload: Dict[str, Any] = {
            "success": result["success"],
            "command": command,
            "timestamp": result["timestamp"],
            "returncode": result["returncode"],
            "command_output": result["stdout"][:500],
            "operation": purpose,
            "unlearner": unlearner,
        }
        if artifact_path:
            payload["artifact_path"] = artifact_path
        if result["stderr"]:
            payload["stderr"] = result["stderr"][:500]
        return payload

    def train_model(
        self,
        dataset: str,
        model: str,
        *,
        method: str = "original",
        model_seed: int = 0,
        random_state: Optional[int] = None,
        overrides: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        """Train a baseline model via ``pipeline/step_5_unlearn.py`` (unlearner=original)."""
        return self._run_step5_command(
            unlearner=method,
            dataset=dataset,
            model=model,
            model_seed=model_seed,
            random_state=random_state,
            save_path=save_path,
            overrides=overrides,
            timeout=timeout,
            purpose="Training model",
        )

    def run_unlearning_method(
        self,
        method: str,
        dataset: str,
        model: str,
        model_seed: int = 0,
        random_state: Optional[int] = None,
        overrides: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        """Apply an unlearning method using ``pipeline/step_5_unlearn.py``."""
        return self._run_step5_command(
            unlearner=method,
            dataset=dataset,
            model=model,
            model_seed=model_seed,
            random_state=random_state,
            save_path=save_path,
            overrides=overrides,
            timeout=timeout,
            purpose="Running unlearning",
        )

    def materialize_artifacts(
        self,
        dataset: str,
        model: str,
        train_method: str,
        unlearn_method: str,
        model_seed: int,
        data_dir: str = "data",
    ) -> Dict[str, Any]:
        """Copy dataset splits and checkpoints into the project ``data`` directory."""
        artifacts_root = self.deepunlearn_dir / "artifacts" / dataset / "unlearn"

        def _discover_roots(method: str) -> List[Path]:
            pattern = f"model_{model}-unlearner_{method}*"
            return [path for path in artifacts_root.glob(pattern) if path.is_dir()]

        train_roots = _discover_roots(train_method)
        if not train_roots:
            raise PipelineError(
                f"Training artifacts not found for method '{train_method}' under {artifacts_root}"
            )
        unlearn_roots = _discover_roots(unlearn_method)
        if not unlearn_roots:
            raise PipelineError(
                f"Unlearned artifacts not found for method '{unlearn_method}' under {artifacts_root}"
            )

        def _find_checkpoint(roots: List[Path]) -> Path:
            candidates: List[Path] = []
            for root in roots:
                candidates.extend(
                    root.rglob(f"*_{model_seed}.pth")
                )
            candidates = [path for path in candidates if path.exists()]
            if not candidates:
                raise PipelineError(
                    f"No checkpoint matching seed {model_seed} found in {roots}"
                )
            candidates.sort(key=lambda path: path.stat().st_mtime)
            return candidates[-1]

        train_ckpt = _find_checkpoint(train_roots)
        unlearn_ckpt = _find_checkpoint(unlearn_roots)

        data_root = (self.project_root / data_dir).resolve()
        models_root = data_root / "models"
        models_root.mkdir(parents=True, exist_ok=True)

        model_suffix = f"{model}_{dataset}_seed{model_seed:03d}"
        original_name = f"{model_suffix}_{train_method}_original.pth"
        unlearned_name = f"{model_suffix}_{unlearn_method}.pth"

        original_dest = models_root / original_name
        unlearned_dest = models_root / unlearned_name
        shutil.copy2(train_ckpt, original_dest)
        shutil.copy2(unlearn_ckpt, unlearned_dest)

        def _copy_sidecar(source: Path, destination: Path) -> Optional[str]:
            candidate = source.with_name(source.stem + "_meta.txt")
            if candidate.exists():
                dest_meta = destination.with_suffix(".meta.txt")
                shutil.copy2(candidate, dest_meta)
                return str(dest_meta)
            return None

        original_meta = _copy_sidecar(train_ckpt, original_dest)
        unlearned_meta = _copy_sidecar(unlearn_ckpt, unlearned_dest)

        splits_src = self.deepunlearn_dir / "artifacts" / "fixed_splits" / dataset
        splits_dest = data_root / "splits" / dataset
        splits_copied = False
        if splits_src.exists():
            shutil.copytree(splits_src, splits_dest, dirs_exist_ok=True)
            splits_copied = True

        metadata = {
            "dataset": dataset,
            "model_arch": model,
            "train_method": train_method,
            "unlearn_method": unlearn_method,
            "model_seed": model_seed,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "original_checkpoint": str(original_dest.relative_to(self.project_root)),
            "unlearned_checkpoint": str(unlearned_dest.relative_to(self.project_root)),
            "source_original": str(train_ckpt.relative_to(self.deepunlearn_dir)),
            "source_unlearned": str(unlearn_ckpt.relative_to(self.deepunlearn_dir)),
            "splits_copied": splits_copied,
        }
        if original_meta:
            metadata["original_meta"] = original_meta
        if unlearned_meta:
            metadata["unlearned_meta"] = unlearned_meta

        metadata_path = models_root / f"{model_suffix}_{unlearn_method}.metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "success": True,
            "data_dir": str(data_root),
            "original_model": str(original_dest),
            "unlearned_model": str(unlearned_dest),
            "splits_copied": splits_copied,
            "metadata_file": str(metadata_path),
        }

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        pipeline functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        return [
            FunctionTool(self.prepare_data_splits),
            FunctionTool(self.generate_initial_models),
            FunctionTool(self.link_model_initializations),
            FunctionTool(self.train_model),
            FunctionTool(self.run_unlearning_method),
            FunctionTool(self.materialize_artifacts),
        ]

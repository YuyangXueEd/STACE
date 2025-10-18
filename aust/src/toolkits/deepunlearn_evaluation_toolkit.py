"""DeepUnlearn evaluation toolkit for CAMEL agents.

This toolkit surfaces the evaluation metrics described in the DeepUnlearn
benchmark (retain/forget accuracies, membership inference, indiscernibility,
etc.) as callable tools that an agent can invoke while orchestrating the
pipeline.  It complements :mod:`DeepUnlearnToolkit` by focusing on post-hoc
assessment of trained / unlearned model checkpoints.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from camel.logger import get_logger
from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.utils import MCPServer

logger = get_logger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation logic cannot complete successfully."""


@MCPServer()
class DeepUnlearnEvaluationToolkit(BaseToolkit):
    """Expose DeepUnlearn evaluation routines as CAMEL tools.

    The toolkit imports helper utilities from the DeepUnlearn repository to
    compute accuracy, membership inference attack (MIA) scores, derived
    indiscernibility metrics, retention ratios, and symmetric absolute
    percentage error (SAPE) measurements.

    Args:
        deepunlearn_dir: Relative or absolute path to the DeepUnlearn repo.
        project_root: Root directory containing the DeepUnlearn checkout.
        timeout: Optional timeout (seconds) for long-running evaluations.
        verbose: Toggle additional logging when set to ``True``.
    """

    def __init__(
        self,
        deepunlearn_dir: str = "./external/DeepUnlearn",
        project_root: str = ".",
        *,
        timeout: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(timeout=timeout)
        self.project_root = Path(project_root).resolve()
        self.deepunlearn_dir = (self.project_root / deepunlearn_dir).resolve()
        self.verbose = verbose
        self._ensure_import_path()
        logger.info(
            "Initialized DeepUnlearnEvaluationToolkit with deepunlearn_dir=%s",
            self.deepunlearn_dir,
        )

    # ------------------------------------------------------------------ #
    # Public tool functions
    # ------------------------------------------------------------------ #

    def evaluate_model_checkpoint(
        self,
        model_path: str,
        *,
        model_type: str,
        dataset: str,
        splits_dir: Optional[str] = None,
        batch_size: int = 128,
        random_state: int = 0,
        device: str = "auto",
        split_index: Optional[int] = None,
        forget_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate a trained/unlearned checkpoint on retain/forget/val/test splits.

        This mirrors the CLI in ``munl/evaluation/evaluate_model.py`` and returns
        a JSON-serialisable structure containing accuracy and MIA scores.

        Args:
            model_path: Path to the checkpoint ``.pt`` file.
            model_type: Model architecture identifier understood by DeepUnlearn.
            dataset: Dataset name (e.g., ``cifar10``) for loader construction.
            splits_dir: Optional directory containing ``retain.npy``,
                ``forget.npy``, ``val.npy`` and ``test.npy`` split indices.
                The path is resolved relative to ``project_root`` when not
                absolute. When provided, these custom splits are used instead
                of the default dataset partitions.
            batch_size: Batch size used for evaluation loaders.
            random_state: RNG seed forwarded to DeepUnlearn helpers.
            device: ``"auto"`` chooses CUDA when available, otherwise CPU.
            split_index: Optional LIRA split index (requires ``forget_index``).
            forget_index: Optional LIRA forget index (requires ``split_index``).

        Returns:
            Dict[str, Any]: Metrics dictionary keyed by split and attack name.
        """

        resolved_device = self._resolve_device(device)
        timestamp = datetime.now(timezone.utc).isoformat()
        splits_path: Optional[Path] = None
        if splits_dir is not None:
            splits_path = (self.project_root / splits_dir).resolve()

        try:
            df = self._run_checkpoint_evaluation(
                model_path=Path(model_path),
                model_type=model_type,
                dataset=dataset,
                splits_dir=splits_path,
                batch_size=batch_size,
                random_state=random_state,
                device=resolved_device,
                split_index=split_index,
                forget_index=forget_index,
            )
            metrics = self._serialise_dataframe_metrics(df)
            metrics["val_indiscernibility"] = self.compute_indiscernibility(
                metrics["val_mia"]
            )["indiscernibility"]
            metrics["test_indiscernibility"] = self.compute_indiscernibility(
                metrics["test_mia"]
            )["indiscernibility"]
            payload: Dict[str, Any] = {
                "success": True,
                "timestamp": timestamp,
                "model_path": str(Path(model_path)),
                "dataset": dataset,
                "device": resolved_device,
                "metrics": metrics,
            }
            if splits_path is not None:
                payload["splits_dir"] = str(splits_path)
            if split_index is not None and forget_index is not None:
                payload["lira_indices"] = {
                    "split_index": split_index,
                    "forget_index": forget_index,
                }
            return payload
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Failed to evaluate model checkpoint: %s", exc)
            return {
                "success": False,
                "timestamp": timestamp,
                "model_path": str(Path(model_path)),
                "dataset": dataset,
                "error": str(exc),
            }

    def compute_indiscernibility(self, mia_score: float) -> Dict[str, Any]:
        """Compute the indiscernibility metric (1 - |MIA-0.5| / 0.5)."""
        from munl.evaluation.indiscernibility import indiscernibility

        score = float(indiscernibility(mia_score))
        return {
            "success": True,
            "mia_score": float(mia_score),
            "indiscernibility": score,
        }

    def compute_accuracy_retention(
        self, new_accuracy: float, reference_accuracy: float
    ) -> Dict[str, Any]:
        """Compute retention ratio between unlearned and reference accuracy."""
        from munl.evaluation.retention import compute_accuracy_retention

        retention = float(compute_accuracy_retention(new_accuracy, reference_accuracy))
        return {
            "success": True,
            "new_accuracy": float(new_accuracy),
            "reference_accuracy": float(reference_accuracy),
            "retention": retention,
        }

    def compute_sape_score(self, a: float, b: float) -> Dict[str, Any]:
        """Compute the symmetric absolute percentage error (SAPE) between two values."""
        from munl.evaluation.symmetric_absolute_percentage_error import sape_score

        score = float(sape_score(float(a), float(b)))
        return {
            "success": True,
            "a": float(a),
            "b": float(b),
            "sape": score,
        }

    def compute_model_weight_distance(
        self,
        model_a_path: str,
        model_b_path: str,
        *,
        model_type: str,
        dataset: str,
        device: str = "auto",
    ) -> Dict[str, Any]:
        """Measure L2 and normalised-L2 distances between two checkpoints."""
        from munl.evaluation.model_distances import (
            model_l2_norm,
            models_l2_distance,
            models_normalized_l2_distance,
        )

        resolved_device = self._resolve_device(device)
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            model_a = self._load_model(
                model_path=Path(model_a_path),
                model_type=model_type,
                dataset=dataset,
                device=resolved_device,
            )
            model_b = self._load_model(
                model_path=Path(model_b_path),
                model_type=model_type,
                dataset=dataset,
                device=resolved_device,
            )

            distance = float(models_l2_distance(model_a, model_b))
            normalised = float(models_normalized_l2_distance(model_a, model_b))
            norm_a = float(model_l2_norm(model_a))
            norm_b = float(model_l2_norm(model_b))
            return {
                "success": True,
                "timestamp": timestamp,
                "model_a_path": str(Path(model_a_path)),
                "model_b_path": str(Path(model_b_path)),
                "dataset": dataset,
                "device": resolved_device,
                "metrics": {
                    "l2_distance": distance,
                    "normalized_l2_distance": normalised,
                    "model_a_l2_norm": norm_a,
                    "model_b_l2_norm": norm_b,
                },
            }
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Failed to compute model distance: %s", exc)
            return {
                "success": False,
                "timestamp": timestamp,
                "model_a_path": str(Path(model_a_path)),
                "model_b_path": str(Path(model_b_path)),
                "dataset": dataset,
                "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    # Toolkit registration
    # ------------------------------------------------------------------ #

    def get_tools(self) -> List[FunctionTool]:
        """Return FunctionTool objects representing evaluation utilities."""
        return [
            FunctionTool(self.evaluate_model_checkpoint),
            FunctionTool(self.compute_indiscernibility),
            FunctionTool(self.compute_accuracy_retention),
            FunctionTool(self.compute_sape_score),
            FunctionTool(self.compute_model_weight_distance),
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_import_path(self) -> None:
        """Add ``deepunlearn_dir`` to ``sys.path`` for dynamic imports."""
        repo_path = str(self.deepunlearn_dir)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    def _resolve_device(self, requested: str) -> str:
        if requested is None:
            requested = "auto"
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency upstream
            return "cpu"

        if requested.lower() != "auto":
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _serialise_dataframe_metrics(self, df) -> Dict[str, float]:
        """Convert evaluation DataFrame into a flat float dictionary."""
        row = df.iloc[0].to_dict()
        return {
            "retain_accuracy": float(row["Retain"]),
            "forget_accuracy": float(row["Forget"]),
            "val_accuracy": float(row["Val"]),
            "test_accuracy": float(row["Test"]),
            "val_mia": float(row["Val MIA"]),
            "test_mia": float(row["Test MIA"]),
        }

    def _run_checkpoint_evaluation(
        self,
        *,
        model_path: Path,
        model_type: str,
        dataset: str,
        splits_dir: Optional[Path],
        batch_size: int,
        random_state: int,
        device: str,
        split_index: Optional[int],
        forget_index: Optional[int],
    ):
        """Invoke DeepUnlearn evaluation helpers and return a DataFrame."""
        from munl.evaluation.evaluate_model import ModelEvaluationFromPathApp

        app = ModelEvaluationFromPathApp(
            model_path=model_path,
            model_type=model_type,
            dataset=dataset,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
        )

        if split_index is None and forget_index is None:
            if splits_dir is not None:
                retain_loader, forget_loader, val_loader, test_loader = (
                    self._build_loaders_from_splits(
                        dataset=dataset,
                        batch_size=batch_size,
                        splits_dir=splits_dir,
                    )
                )
                return app.run_on_loaders(
                    retain_loader=retain_loader,
                    forget_loader=forget_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                )
            return app.run()

        if split_index is None or forget_index is None:
            raise EvaluationError(
                "Both split_index and forget_index must be provided for LIRA evaluation."
            )
        if splits_dir is not None:
            raise EvaluationError(
                "Custom splits cannot be combined with LIRA indices. "
                "Provide either splits_dir or LIRA indices."
            )

        from munl.settings import default_evaluation_loaders
        from munl.datasets import get_loaders_from_dataset_and_unlearner_from_cfg_with_indices
        from munl.utils import DictConfig
        from pipeline.lira.step_1_lira_generate_splits import (
            get_retain_forget_val_test_indices,
        )

        lira_path = self.deepunlearn_dir / "artifacts" / "lira" / "splits"
        if not lira_path.exists():
            raise EvaluationError(
                f"LIRA split directory not found at {lira_path}. Generate LIRA splits first."
            )

        retain, forget, val, test = get_retain_forget_val_test_indices(
            lira_path=lira_path,
            split_ndx=split_index,
            forget_ndx=forget_index,
        )
        dataset_cfg = DictConfig({"name": dataset})
        unlearner_cfg = DictConfig(
            {"loaders": default_evaluation_loaders(), "batch_size": batch_size}
        )

        _train_loader, retain_loader, forget_loader, val_loader, test_loader = (
            get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
                root=self.deepunlearn_dir,
                indices=[
                    np.concatenate([retain, forget]),
                    retain,
                    forget,
                    val,
                    test,
                ],
                dataset_cfg=dataset_cfg,
                unlearner_cfg=unlearner_cfg,
            )
        )
        return app.run_on_loaders(
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

    def _load_model(
        self,
        *,
        model_path: Path,
        model_type: str,
        dataset: str,
        device: str,
    ):
        """Load a DeepUnlearn model into memory for distance calculations."""
        import torch
        from munl.configurations import get_img_size_for_dataset, get_num_classes
        from munl.models import get_model

        if not model_path.exists():
            raise EvaluationError(f"Model checkpoint not found: {model_path}")

        num_classes = get_num_classes(dataset)
        img_size = get_img_size_for_dataset(dataset)
        model = get_model(model_type, num_classes, img_size)
        weights = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(weights)
        model.eval()
        return model

    def _build_loaders_from_splits(
        self,
        *,
        dataset: str,
        batch_size: int,
        splits_dir: Path,
    ):
        """Construct dataset loaders from a directory of split indices."""
        if not splits_dir.exists():
            raise EvaluationError(f"Splits directory not found: {splits_dir}")

        indices = self._load_split_indices(splits_dir)
        required = ("retain", "forget", "val", "test")
        missing = [name for name in required if name not in indices]
        if missing:
            raise EvaluationError(
                f"Missing required split files in {splits_dir}: {', '.join(missing)}"
            )

        train_indices = indices.get("train")
        if train_indices is None:
            train_indices = np.concatenate([indices["retain"], indices["forget"]])

        from munl.datasets import (
            get_loaders_from_dataset_and_unlearner_from_cfg_with_indices,
        )
        from munl.settings import default_evaluation_loaders
        from munl.utils import DictConfig

        dataset_cfg = DictConfig({"name": dataset})
        unlearner_cfg = DictConfig(
            {"loaders": default_evaluation_loaders(), "batch_size": batch_size}
        )

        _train_loader, retain_loader, forget_loader, val_loader, test_loader = (
            get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
                root=self.deepunlearn_dir,
                indices=[
                    train_indices,
                    indices["retain"],
                    indices["forget"],
                    indices["val"],
                    indices["test"],
                ],
                dataset_cfg=dataset_cfg,
                unlearner_cfg=unlearner_cfg,
            )
        )
        return retain_loader, forget_loader, val_loader, test_loader

    def _load_split_indices(self, splits_dir: Path) -> Dict[str, np.ndarray]:
        """Load split indices from ``.npy`` files in the specified directory."""
        split_map: Dict[str, np.ndarray] = {}
        for name in ("train", "retain", "forget", "val", "test"):
            file_path = splits_dir / f"{name}.npy"
            if file_path.exists():
                split_map[name] = np.load(file_path)
        return split_map

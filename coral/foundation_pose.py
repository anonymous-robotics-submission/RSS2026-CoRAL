"""Wrapper utilities for interacting with FoundationPose."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import FoundationPoseConfig
from .types import PoseEstimate

LOGGER = logging.getLogger(__name__)

try:
    from foundationpose import FoundationPose
except ImportError:  # pragma: no cover - dependency is optional at import time
    FoundationPose = None  # type: ignore[assignment]


@dataclass
class FoundationPoseInputs:
    """Container for RGB-D data and intrinsics."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    segmentation_mask: Optional[np.ndarray] = None


class FoundationPoseTracker:
    """High-level interface around FoundationPose for 6-DoF tracking."""

    def __init__(self, config: FoundationPoseConfig) -> None:
        if FoundationPose is None:
            raise ImportError(
                "FoundationPose package is required but not installed. Install it from"
                " https://github.com/NVlabs/FoundationPose or PyPI before using the"
                " CoRAL vision module."
            )
        self._config = config
        LOGGER.debug("Initialising FoundationPose with config: %s", config)
        kwargs: Dict[str, Any] = dict(config.additional_kwargs)
        if config.checkpoint_path is not None:
            kwargs.setdefault("checkpoint_path", str(config.checkpoint_path))
        kwargs.setdefault("device", config.device)
        kwargs.setdefault("max_points", config.max_points)
        kwargs.setdefault("enable_slam", config.enable_slam)
        self._model = FoundationPose(**kwargs)
        self._track_history = {}

    def estimate_pose(
        self,
        object_id: str,
        inputs: FoundationPoseInputs,
        previous_pose: Optional[PoseEstimate] = None,
    ) -> PoseEstimate:
        """Estimate a 6-DoF pose for ``object_id`` from the provided RGB-D inputs."""

        LOGGER.debug("Estimating pose for %s", object_id)
        prediction = self._run_model(object_id=object_id, inputs=inputs)
        pose = PoseEstimate(
            object_id=object_id,
            position=self._to_tuple(prediction["position"]),
            orientation=self._to_tuple(prediction["orientation"]),
            confidence=float(prediction.get("confidence", 1.0)),
            metadata={"raw": prediction},
        )
        self._update_history(object_id, pose)
        if previous_pose is not None and pose.confidence < previous_pose.confidence:
            LOGGER.debug(
                "Keeping previous pose for %s because confidence dropped (%.3f < %.3f)",
                object_id,
                pose.confidence,
                previous_pose.confidence,
            )
            return previous_pose
        return pose

    def reset(self) -> None:
        """Clear internal tracking history."""

        LOGGER.debug("Resetting FoundationPose tracker history")
        self._track_history.clear()

    def _run_model(self, object_id: str, inputs: FoundationPoseInputs) -> Dict[str, Any]:
        if hasattr(self._model, "estimate_pose"):
            output = self._model.estimate_pose(
                rgb=inputs.rgb,
                depth=inputs.depth,
                intrinsics=inputs.intrinsics,
                object_id=object_id,
                mask=inputs.segmentation_mask,
            )
        else:
            output = self._model(
                rgb=inputs.rgb,
                depth=inputs.depth,
                intrinsics=inputs.intrinsics,
                object_id=object_id,
                mask=inputs.segmentation_mask,
            )
        if not {"position", "orientation"}.issubset(output):
            raise ValueError(
                "FoundationPose output is missing required keys: position/orientation"
            )
        LOGGER.debug(
            "FoundationPose output for %s: position=%s, orientation=%s, confidence=%s",
            object_id,
            output.get("position"),
            output.get("orientation"),
            output.get("confidence"),
        )
        return output

    @staticmethod
    def _to_tuple(array_like: Any) -> tuple:
        if isinstance(array_like, (tuple, list)):
            return tuple(float(x) for x in array_like)
        if hasattr(array_like, "tolist"):
            return tuple(float(x) for x in array_like.tolist())
        raise TypeError(f"Cannot convert type {type(array_like)} to tuple")

    def _update_history(self, object_id: str, pose: PoseEstimate) -> None:
        history = self._track_history.setdefault(object_id, [])
        history.append(pose)
        if len(history) > self._config.track_history:
            del history[0]

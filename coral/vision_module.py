"""Vision and vision-language modules for the CoRAL pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from .config import FoundationPoseConfig, VisionLanguageConfig
from .foundation_pose import FoundationPoseInputs, FoundationPoseTracker
from .types import PhysicalParameters, PoseEstimate

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency for default VLMs
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = AutoProcessor = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]


@dataclass
class VisionObservations:
    """Aggregate output from the vision module."""

    pose: PoseEstimate
    physical_parameters: PhysicalParameters


class VisionLanguageModel:
    """Lightweight wrapper providing text generation over multi-modal inputs."""

    def __init__(
        self,
        config: VisionLanguageConfig,
        generator: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        self._config = config
        if generator is not None:
            self._generator = generator
            self._processor = None
            self._model = None
            LOGGER.info("Using custom vision-language generator callback")
        else:
            if AutoModelForCausalLM is None or AutoProcessor is None or torch is None:
                raise ImportError(
                    "transformers and torch are required to instantiate the default"
                    " vision-language model. Provide a custom generator callable if"
                    " you cannot install these dependencies."
                )
            LOGGER.info("Loading vision-language model '%s'", config.model_name)
            self._processor = AutoProcessor.from_pretrained(config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(config.model_name)
            self._model.eval()
            self._generator = None

    def infer(self, inputs: Dict[str, Any]) -> str:
        """Generate a textual response from the VLM."""

        if self._generator is not None:
            return self._generator(inputs)
        if self._processor is None or self._model is None:
            raise RuntimeError("Vision-language model is not properly initialised")
        prompt = inputs["prompt"]
        images = inputs.get("images")
        processed = self._processor(text=prompt, images=images, return_tensors="pt")
        generated_ids = self._model.generate(
            **processed,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
        )
        output = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        LOGGER.debug("VLM output: %s", output)
        return output


class VisionModule:
    """Handles pose tracking and physical parameter estimation."""

    def __init__(
        self,
        tracker: FoundationPoseTracker,
        vlm: VisionLanguageModel,
        config: FoundationPoseConfig,
    ) -> None:
        self._tracker = tracker
        self._vlm = vlm
        self._vision_config = config

    def observe(
        self,
        object_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        task_description: str,
        previous_pose: Optional[PoseEstimate] = None,
    ) -> VisionObservations:
        """Estimate pose and physical parameters for the target object."""

        pose = self._tracker.estimate_pose(
            object_id=object_id,
            inputs=FoundationPoseInputs(rgb=rgb, depth=depth, intrinsics=intrinsics),
            previous_pose=previous_pose,
        )
        parameters = self._infer_physical_parameters(pose=pose, task_description=task_description)
        return VisionObservations(pose=pose, physical_parameters=parameters)

    def _infer_physical_parameters(
        self, pose: PoseEstimate, task_description: str
    ) -> PhysicalParameters:
        prompt = self._compose_prompt(pose=pose, task_description=task_description)
        LOGGER.debug("Submitting prompt to VLM: %s", prompt)
        response = self._vlm.infer({"prompt": prompt, "images": pose.metadata.get("image")})
        LOGGER.debug("Received VLM response: %s", response)
        parsed = self._parse_response(response)
        return PhysicalParameters(
            object_id=pose.object_id,
            mass=parsed.get("mass", 1.0),
            friction_coefficient=parsed.get("friction", 0.5),
            compliance=parsed.get("compliance"),
            metadata={"raw_response": response},
        )

    @staticmethod
    def _compose_prompt(pose: PoseEstimate, task_description: str) -> str:
        return (
            "You are provided with the following object pose in SE(3) coordinates:"
            f" position={pose.position}, orientation quaternion={pose.orientation}."
            f" The task is: {task_description}."
            " Estimate the object's mass (kg), surface friction coefficient (0-1),"
            " and compliance (0-1, optional) needed for contact-rich planning."
            " Respond in a concise key-value format, e.g. mass: 1.2, friction: 0.3,"
            " compliance: 0.4."
        )

    @staticmethod
    def _parse_response(response: str) -> Dict[str, float]:
        patterns = {
            "mass": r"mass\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            "friction": r"friction\s*(?:coefficient)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            "compliance": r"compliance\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        }
        parsed: Dict[str, float] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, response, flags=re.IGNORECASE)
            if match:
                parsed[key] = float(match.group(1))
        return parsed

    def reset(self) -> None:
        self._tracker.reset()

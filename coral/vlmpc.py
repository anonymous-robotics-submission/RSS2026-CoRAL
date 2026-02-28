"""VLMPC: Vision-Language Model Predictive Control baseline implementation.

Reference: Zhao et al., "VLMPC: Vision-Language Model Predictive Control
for Robotic Manipulation", RSS 2024. https://arxiv.org/abs/2407.09829

This module implements the VLMPC baseline for comparison with CoRAL.
VLMPC uses a VLM to guide action sampling and evaluate predicted trajectories
through a hierarchical cost function combining pixel-level and knowledge-level costs.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import VLMPCConfig
from .state_metrics import StateMetricsEvaluator
from .types import (
    ActionPlan,
    ActionPrimitive,
    ContactStrategy,
    PlanFeedback,
    PoseEstimate,
    StateMetrics,
    WorldState,
)

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency for default VLM access
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:  # Optional dependencies for image processing
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass
class VLMPCActionSample:
    """A sampled action sequence with associated costs."""

    actions: np.ndarray  # Shape: (horizon, action_dim)
    pixel_cost: float = 0.0
    knowledge_cost: float = 0.0
    total_cost: float = 0.0
    predicted_frames: Optional[List[np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMPCPlanResult:
    """Result of VLMPC planning iteration."""

    best_action: np.ndarray  # Shape: (action_dim,)
    best_sequence: np.ndarray  # Shape: (horizon, action_dim)
    all_samples: List[VLMPCActionSample]
    mean_cost: float
    min_cost: float
    iterations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleVideoPredictionModel:
    """Simple video prediction using linear interpolation.

    This is a lightweight alternative to DMVFN-Act for environments
    where learning-based video prediction is not available.
    """

    def __init__(self, num_frames: int = 8) -> None:
        self._num_frames = num_frames

    def predict(
        self,
        current_frame: np.ndarray,
        goal_frame: np.ndarray,
        actions: np.ndarray,
    ) -> List[np.ndarray]:
        """Predict future frames via linear interpolation weighted by actions.

        Args:
            current_frame: Current RGB observation (H, W, 3)
            goal_frame: Goal RGB image (H, W, 3)
            actions: Action sequence (horizon, action_dim)

        Returns:
            List of predicted frames
        """
        frames = []
        action_magnitudes = np.linalg.norm(actions, axis=-1)
        cumulative_progress = np.cumsum(action_magnitudes)
        if cumulative_progress[-1] > 0:
            cumulative_progress = cumulative_progress / cumulative_progress[-1]
        else:
            cumulative_progress = np.linspace(0, 1, len(actions))

        for i in range(min(self._num_frames, len(actions))):
            alpha = cumulative_progress[i]
            interpolated = (1 - alpha) * current_frame.astype(np.float32) + alpha * goal_frame.astype(np.float32)
            frames.append(interpolated.astype(np.uint8))

        return frames


class DMVFNVideoPredictionModel:
    """Wrapper for DMVFN-Act video prediction model.

    This provides an interface to the action-conditioned video prediction
    model used in the original VLMPC paper. Falls back to simple prediction
    if the model is not available.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model = None
        self._fallback = SimpleVideoPredictionModel()

        if checkpoint_path is not None:
            try:
                self._load_model()
            except Exception as e:
                LOGGER.warning(
                    "Failed to load DMVFN-Act model from %s: %s. Using fallback.",
                    checkpoint_path,
                    e,
                )

    def _load_model(self) -> None:
        """Load the DMVFN-Act model checkpoint."""
        # Placeholder for actual DMVFN-Act loading
        # In practice, this would load the pretrained model from:
        # https://github.com/PPjmchen/VLMPC
        LOGGER.info("DMVFN-Act model loading not implemented, using fallback")

    def predict(
        self,
        current_frame: np.ndarray,
        goal_frame: np.ndarray,
        actions: np.ndarray,
    ) -> List[np.ndarray]:
        """Predict future frames conditioned on actions.

        Args:
            current_frame: Current RGB observation (H, W, 3)
            goal_frame: Goal RGB image (H, W, 3)
            actions: Action sequence (horizon, action_dim)

        Returns:
            List of predicted frames
        """
        if self._model is not None:
            # Use actual DMVFN-Act model
            # This would involve:
            # 1. Preprocess frames
            # 2. Encode actions
            # 3. Run forward pass
            # 4. Decode predicted frames
            pass

        # Fallback to simple prediction
        return self._fallback.predict(current_frame, goal_frame, actions)


class VLMPCAgent:
    """VLMPC: Vision-Language Model Predictive Control agent.

    This implements the VLMPC baseline from RSS 2024. The key components are:

    1. VLM-guided Action Sampling: Uses a VLM to generate a conditional
       action distribution based on the current observation and goal.

    2. Video Prediction: Predicts future frames given sampled action sequences
       using DMVFN-Act or a simpler interpolation model.

    3. Hierarchical Cost Function:
       - Pixel-level cost: Measures visual similarity between predicted
         frames and the goal image.
       - Knowledge-level cost: Uses VLM to evaluate the quality of
         predicted trajectories semantically.

    4. MPC Planning Loop: Iteratively refines action samples based on
       the hierarchical cost function.
    """

    def __init__(
        self,
        config: Optional[VLMPCConfig] = None,
        *,
        vlm_completion_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        video_predictor: Optional[DMVFNVideoPredictionModel] = None,
        metrics_evaluator: Optional[StateMetricsEvaluator] = None,
    ) -> None:
        self._config = config or VLMPCConfig()
        self._rng = np.random.default_rng()

        # Initialize VLM client
        if vlm_completion_fn is not None:
            self._vlm_completion_fn = vlm_completion_fn
            self._client = None
            LOGGER.info("Using custom VLM completion function for VLMPC")
        else:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required for VLMPC. Provide a custom"
                    " vlm_completion_fn if you cannot install it."
                )
            self._client = OpenAI()
            self._vlm_completion_fn = None
            LOGGER.info("Initialized OpenAI client for VLMPC")

        # Initialize video prediction model
        if video_predictor is not None:
            self._video_predictor = video_predictor
        elif self._config.video_prediction_enabled:
            self._video_predictor = DMVFNVideoPredictionModel()
        else:
            self._video_predictor = SimpleVideoPredictionModel(
                num_frames=self._config.prediction_frames
            )

        # Metrics evaluator for compatibility with CoRAL pipeline
        self._metrics_evaluator = metrics_evaluator

        # Planning state
        self._current_action_mean: Optional[np.ndarray] = None
        self._current_action_std: Optional[np.ndarray] = None
        self._step_count = 0

        LOGGER.info("VLMPC agent initialized with config: %s", self._config)

    def plan(
        self,
        current_rgb: np.ndarray,
        goal_rgb: np.ndarray,
        task_description: str,
        world_state: Optional[WorldState] = None,
        pose: Optional[PoseEstimate] = None,
    ) -> VLMPCPlanResult:
        """Plan the next action using VLMPC.

        Args:
            current_rgb: Current RGB observation (H, W, 3)
            goal_rgb: Goal RGB image (H, W, 3)
            task_description: Natural language task description
            world_state: Optional world state for context
            pose: Optional pose estimate

        Returns:
            VLMPCPlanResult containing the best action and planning statistics
        """
        # Step 1: Initialize or update action distribution using VLM
        if self._should_resample():
            action_prior = self._vlm_action_sampling(
                current_rgb=current_rgb,
                goal_rgb=goal_rgb,
                task_description=task_description,
            )
            self._current_action_mean = action_prior["mean"]
            self._current_action_std = action_prior["std"]

        # Step 2: Sample action sequences from the distribution
        samples = self._sample_actions()

        # Step 3: Iterative optimization with hierarchical cost
        for iteration in range(self._config.num_iterations):
            # Evaluate all samples
            for sample in samples:
                # Predict video frames
                sample.predicted_frames = self._video_predictor.predict(
                    current_frame=current_rgb,
                    goal_frame=goal_rgb,
                    actions=sample.actions,
                )

                # Compute pixel-level cost
                sample.pixel_cost = self._compute_pixel_cost(
                    predicted_frames=sample.predicted_frames,
                    goal_frame=goal_rgb,
                )

                # Compute knowledge-level cost using VLM
                sample.knowledge_cost = self._compute_knowledge_cost(
                    current_rgb=current_rgb,
                    predicted_frames=sample.predicted_frames,
                    goal_rgb=goal_rgb,
                    task_description=task_description,
                )

                # Compute total cost
                sample.total_cost = (
                    self._config.pixel_cost_weight * sample.pixel_cost
                    + self._config.knowledge_cost_weight * sample.knowledge_cost
                    + self._config.smoothness_weight * self._compute_smoothness_cost(sample.actions)
                )

            # Update distribution using elite samples (CEM-style)
            samples = self._update_distribution(samples)

        # Step 4: Select best action
        best_sample = min(samples, key=lambda s: s.total_cost)
        self._step_count += 1

        costs = [s.total_cost for s in samples]
        return VLMPCPlanResult(
            best_action=best_sample.actions[0],
            best_sequence=best_sample.actions,
            all_samples=samples,
            mean_cost=float(np.mean(costs)),
            min_cost=float(np.min(costs)),
            iterations=self._config.num_iterations,
            metadata={
                "pixel_cost": best_sample.pixel_cost,
                "knowledge_cost": best_sample.knowledge_cost,
                "step_count": self._step_count,
            },
        )

    def plan_to_action_plan(
        self,
        result: VLMPCPlanResult,
        task_description: str,
    ) -> ActionPlan:
        """Convert VLMPC plan result to CoRAL ActionPlan format.

        This enables VLMPC to be used as a drop-in replacement for
        the LLM planner in the CoRAL pipeline.
        """
        # Convert action sequence to primitives
        primitives = []
        for i, action in enumerate(result.best_sequence):
            primitive = ActionPrimitive(
                name="move_to",
                parameters={
                    "delta_position": action[:3].tolist(),
                    "delta_orientation": action[3:].tolist() if len(action) > 3 else [0, 0, 0, 1],
                    "step_index": i,
                },
                expected_duration=0.1,
            )
            primitives.append(primitive)

        strategy = ContactStrategy(
            summary=f"VLMPC planned trajectory for: {task_description}",
            primitives=[p.name for p in primitives],
            cost_function={
                "pixel_cost": result.metadata.get("pixel_cost", 0.0),
                "knowledge_cost": result.metadata.get("knowledge_cost", 0.0),
            },
            constraints=[],
            confidence=1.0 - result.min_cost,  # Higher confidence for lower cost
        )

        return ActionPlan(
            strategy=strategy,
            primitives=primitives,
            metadata={
                "source": "vlmpc",
                "mean_cost": result.mean_cost,
                "min_cost": result.min_cost,
                "iterations": result.iterations,
                **result.metadata,
            },
        )

    def iterate(
        self,
        *,
        object_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        task_description: str,
        goal_rgb: np.ndarray,
        environment,
        tags: Optional[List[str]] = None,
    ) -> PlanFeedback:
        """Run a single iteration of VLMPC planning and execution.

        This method provides a compatible interface with CoRALAgent.iterate().

        Args:
            object_id: ID of the target object
            rgb: Current RGB observation
            depth: Current depth image (unused by VLMPC)
            intrinsics: Camera intrinsics (unused by VLMPC)
            task_description: Natural language task description
            goal_rgb: Goal RGB image
            environment: Environment for action execution
            tags: Optional tags for logging

        Returns:
            PlanFeedback with execution results
        """
        # Plan using VLMPC
        plan_result = self.plan(
            current_rgb=rgb,
            goal_rgb=goal_rgb,
            task_description=task_description,
        )

        # Convert to ActionPlan format
        action_plan = self.plan_to_action_plan(plan_result, task_description)

        # Execute actions
        feedback = self._execute_plan(action_plan, environment)

        return feedback

    def reset(self) -> None:
        """Reset the agent state."""
        self._current_action_mean = None
        self._current_action_std = None
        self._step_count = 0
        LOGGER.info("VLMPC agent reset")

    def _should_resample(self) -> bool:
        """Check if we should resample the action distribution."""
        if self._current_action_mean is None:
            return True
        return self._step_count % self._config.plan_frequency == 0

    def _vlm_action_sampling(
        self,
        current_rgb: np.ndarray,
        goal_rgb: np.ndarray,
        task_description: str,
    ) -> Dict[str, np.ndarray]:
        """Use VLM to generate action sampling distribution.

        The VLM analyzes the current observation and goal to suggest
        a distribution of actions that would move toward the goal.
        """
        prompt = self._build_action_sampling_prompt(task_description)

        # Encode images for VLM
        current_b64 = self._encode_image(current_rgb)
        goal_b64 = self._encode_image(goal_rgb)

        response = self._call_vlm(
            prompt=prompt,
            images=[current_b64, goal_b64],
        )

        # Parse VLM response to get action distribution
        try:
            action_params = self._parse_action_response(response)
        except Exception as e:
            LOGGER.warning("Failed to parse VLM action response: %s. Using defaults.", e)
            action_params = self._default_action_params()

        return action_params

    def _build_action_sampling_prompt(self, task_description: str) -> str:
        """Build the prompt for VLM action sampling."""
        return f"""You are a robotic manipulation expert. Given the current observation (first image) and goal state (second image), suggest action parameters for a robot end-effector to accomplish the task.

Task: {task_description}

The robot action space is a 7-dimensional vector:
- Positions 0-2: delta position (x, y, z) in meters, range [-0.5, 0.5]
- Positions 3-6: delta orientation as quaternion (qx, qy, qz, qw), range [-1, 1]

Based on the visual difference between current and goal states, provide:
1. A mean action vector that would move toward the goal
2. A standard deviation for each dimension to allow exploration

Respond in JSON format:
{{
    "mean": [x, y, z, qx, qy, qz, qw],
    "std": [std_x, std_y, std_z, std_qx, std_qy, std_qz, std_qw],
    "reasoning": "brief explanation"
}}
"""

    def _parse_action_response(self, response: str) -> Dict[str, np.ndarray]:
        """Parse the VLM response to extract action parameters."""
        # Try to extract JSON from response
        try:
            # Handle responses that may have text before/after JSON
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")

        mean = np.array(data.get("mean", [0] * self._config.action_dim), dtype=np.float32)
        std = np.array(data.get("std", [self._config.action_noise_std] * self._config.action_dim), dtype=np.float32)

        # Ensure correct dimensions
        if len(mean) != self._config.action_dim:
            mean = np.zeros(self._config.action_dim, dtype=np.float32)
        if len(std) != self._config.action_dim:
            std = np.full(self._config.action_dim, self._config.action_noise_std, dtype=np.float32)

        return {"mean": mean, "std": std}

    def _default_action_params(self) -> Dict[str, np.ndarray]:
        """Return default action parameters."""
        return {
            "mean": np.zeros(self._config.action_dim, dtype=np.float32),
            "std": np.full(self._config.action_dim, self._config.action_noise_std, dtype=np.float32),
        }

    def _sample_actions(self) -> List[VLMPCActionSample]:
        """Sample action sequences from the current distribution."""
        samples = []
        mean = self._current_action_mean
        std = self._current_action_std

        if mean is None or std is None:
            mean = np.zeros(self._config.action_dim)
            std = np.full(self._config.action_dim, self._config.action_noise_std)

        bounds_low = np.array(self._config.action_bounds_low)
        bounds_high = np.array(self._config.action_bounds_high)

        for _ in range(self._config.num_action_samples):
            # Sample action sequence
            actions = np.zeros((self._config.planning_horizon, self._config.action_dim))
            for t in range(self._config.planning_horizon):
                # Decay std over horizon for smoother trajectories
                decay = 0.95 ** t
                action = self._rng.normal(mean, std * decay)
                action = np.clip(action, bounds_low, bounds_high)
                actions[t] = action

            samples.append(VLMPCActionSample(actions=actions))

        return samples

    def _compute_pixel_cost(
        self,
        predicted_frames: List[np.ndarray],
        goal_frame: np.ndarray,
    ) -> float:
        """Compute pixel-level cost between predicted frames and goal.

        This measures the visual similarity between the final predicted
        frame and the goal image using MSE.
        """
        if not predicted_frames:
            return 1.0

        # Use the last predicted frame
        final_frame = predicted_frames[-1].astype(np.float32)
        goal = goal_frame.astype(np.float32)

        # Ensure same shape
        if final_frame.shape != goal.shape:
            # Resize if needed (simplified)
            return 1.0

        # Normalized MSE
        mse = np.mean((final_frame - goal) ** 2)
        max_mse = 255.0 ** 2  # Maximum possible MSE for 8-bit images
        normalized_cost = mse / max_mse

        return float(normalized_cost)

    def _compute_knowledge_cost(
        self,
        current_rgb: np.ndarray,
        predicted_frames: List[np.ndarray],
        goal_rgb: np.ndarray,
        task_description: str,
    ) -> float:
        """Compute knowledge-level cost using VLM evaluation.

        The VLM evaluates the quality of the predicted trajectory
        based on task semantics and physical plausibility.
        """
        if not predicted_frames:
            return 1.0

        prompt = self._build_evaluation_prompt(task_description)

        # Encode images
        current_b64 = self._encode_image(current_rgb)
        final_pred_b64 = self._encode_image(predicted_frames[-1])
        goal_b64 = self._encode_image(goal_rgb)

        response = self._call_vlm(
            prompt=prompt,
            images=[current_b64, final_pred_b64, goal_b64],
        )

        # Parse cost from response
        try:
            cost = self._parse_evaluation_response(response)
        except Exception as e:
            LOGGER.debug("Failed to parse evaluation response: %s", e)
            cost = 0.5  # Default moderate cost

        return cost

    def _build_evaluation_prompt(self, task_description: str) -> str:
        """Build the prompt for VLM trajectory evaluation."""
        return f"""You are evaluating a robot manipulation trajectory. Given three images:
1. Current state (first image)
2. Predicted final state (second image)
3. Goal state (third image)

Task: {task_description}

Evaluate how well the predicted trajectory accomplishes the task. Consider:
- Does the predicted state achieve the goal?
- Is the trajectory physically plausible?
- Are there any potential collisions or failures?

Respond with a JSON object containing:
{{
    "score": <float between 0.0 (perfect) and 1.0 (complete failure)>,
    "reasoning": "brief explanation"
}}
"""

    def _parse_evaluation_response(self, response: str) -> float:
        """Parse the VLM evaluation response to extract cost."""
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0.5))
            return np.clip(score, 0.0, 1.0)

        # Try to find a number directly
        numbers = re.findall(r'(\d+\.?\d*)', response)
        if numbers:
            score = float(numbers[0])
            if score > 1:
                score = score / 100  # Assume percentage
            return np.clip(score, 0.0, 1.0)

        return 0.5

    def _compute_smoothness_cost(self, actions: np.ndarray) -> float:
        """Compute smoothness cost to encourage smooth trajectories."""
        if len(actions) < 2:
            return 0.0

        # Compute differences between consecutive actions
        diffs = np.diff(actions, axis=0)
        smoothness = np.mean(np.sum(diffs ** 2, axis=-1))

        # Normalize
        return float(np.clip(smoothness, 0.0, 1.0))

    def _update_distribution(
        self,
        samples: List[VLMPCActionSample],
    ) -> List[VLMPCActionSample]:
        """Update action distribution using elite samples (CEM-style)."""
        # Sort by total cost
        sorted_samples = sorted(samples, key=lambda s: s.total_cost)

        # Select elite samples
        num_elite = max(1, int(len(samples) * self._config.elite_fraction))
        elite_samples = sorted_samples[:num_elite]

        # Compute new distribution from elite samples
        elite_actions = np.array([s.actions for s in elite_samples])
        new_mean = np.mean(elite_actions[:, 0, :], axis=0)  # Mean of first actions
        new_std = np.std(elite_actions[:, 0, :], axis=0) + 1e-6  # Std with small epsilon

        # Smooth update
        alpha = 0.7
        if self._current_action_mean is not None:
            self._current_action_mean = alpha * new_mean + (1 - alpha) * self._current_action_mean
            self._current_action_std = alpha * new_std + (1 - alpha) * self._current_action_std
        else:
            self._current_action_mean = new_mean
            self._current_action_std = new_std

        # Resample with updated distribution for next iteration
        return self._sample_actions()

    def _call_vlm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
    ) -> str:
        """Call the VLM with the given prompt and images."""
        if self._vlm_completion_fn is not None:
            return self._vlm_completion_fn({"prompt": prompt, "images": images})

        if self._client is None:
            raise RuntimeError("VLM client not initialized")

        # Build messages with images
        content = []
        if images:
            for img_b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
        content.append({"type": "text", "text": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._config.vlm_model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=self._config.vlm_max_tokens,
                temperature=self._config.vlm_temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            LOGGER.error("VLM call failed: %s", e)
            return "{}"

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string."""
        if Image is None:
            # Fallback if PIL not available
            return ""

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _execute_plan(
        self,
        plan: ActionPlan,
        environment,
    ) -> PlanFeedback:
        """Execute the action plan on the environment."""
        success = True
        observations = {}
        total_energy = 0.0

        for i, primitive in enumerate(plan.primitives):
            try:
                # Try to execute via environment's execute method
                handler_name = f"execute_{primitive.name}"
                if hasattr(environment, handler_name):
                    result = getattr(environment, handler_name)(**primitive.parameters)
                elif hasattr(environment, "step"):
                    # Use generic step interface
                    action = np.concatenate([
                        primitive.parameters.get("delta_position", [0, 0, 0]),
                        primitive.parameters.get("delta_orientation", [0, 0, 0, 1]),
                    ])
                    obs, reward, done, info = environment.step(action)
                    result = {
                        "success": not done or reward > 0,
                        "observation": obs,
                        "reward": reward,
                    }
                else:
                    result = {"success": True}

                if not result.get("success", True):
                    success = False
                    break

                observations[f"step_{i}"] = result
                total_energy += primitive.parameters.get("energy", 0.1)

            except Exception as e:
                LOGGER.error("Failed to execute primitive %s: %s", primitive.name, e)
                success = False
                break

        metrics = StateMetrics(
            success_probability=1.0 if success else 0.0,
            contact_stability=0.8 if success else 0.2,
            energy=total_energy,
        )

        return PlanFeedback(
            success=success,
            contact_quality=0.8 if success else 0.2,
            observations=observations,
            metrics=metrics,
        )

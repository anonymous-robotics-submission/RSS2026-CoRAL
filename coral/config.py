"""Configuration dataclasses for the CoRAL pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class FoundationPoseConfig:
    """Settings for initialising the FoundationPose tracker."""

    checkpoint_path: Optional[Path] = None
    device: str = "cuda"
    max_points: int = 20000
    confidence_threshold: float = 0.4
    track_history: int = 20
    enable_slam: bool = False
    additional_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class VisionLanguageConfig:
    """Parameters for the vision-language physical property estimator."""

    model_name: str = "llava"
    temperature: float = 0.2
    max_new_tokens: int = 512
    system_prompt: str = (
        "You are a robotics perception expert. Given an object pose and task,"
        " estimate its relevant physical parameters (mass, friction, compliance)."
    )
    additional_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class LLMPlannerConfig:
    """Settings for the language-driven contact strategy planner."""

    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    system_prompt: str = (
        "You are a contact-rich manipulation strategist. Generate explainable"
        " contact strategies, motion primitives, and cost functions for a"
        " robot based on scene understanding."
    )
    adaptation_prompt: str = (
        "You are refining an existing manipulation plan given new feedback."
        " Update the contact strategy, primitives, and costs accordingly."
    )
    additional_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class MotionPlannerConfig:
    """Settings for motion planning with contact-aware objectives."""

    planner_type: str = "mppi"
    replan_horizon: float = 2.5
    time_step: float = 0.05
    optimisation_iters: int = 40
    sample_count: int = 128
    noise_sigma: float = 0.1
    discount: float = 0.95
    temperature_lambda: float = 1.0
    refinement_loops: int = 2
    seed: Optional[int] = None
    additional_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Settings for the simulated-world refinement stage."""

    enabled: bool = True
    rollout_horizon: int = 10
    max_attempts: int = 2
    parameter_update_rate: float = 0.2
    log_episodes: bool = True


@dataclass
class ReactiveControllerConfig:
    """Settings for the reactive controllers executing the planned actions."""

    controller_type: str = "impedance"
    stiffness: float = 200.0
    damping: float = 20.0
    force_threshold: float = 25.0
    velocity_limit: float = 0.5
    additional_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class StateMetricsConfig:
    """Thresholds and weights for state metric evaluation."""

    success_threshold: float = 0.7
    stability_threshold: float = 0.6
    energy_limit: float = 15.0
    success_weight: float = 0.5
    stability_weight: float = 0.3
    energy_weight: float = 0.2


@dataclass
class MemoryConfig:
    """Settings for the episodic memory unit."""

    storage_path: Path = Path("~/.cache/coral_memory.json").expanduser()
    max_entries: int = 1000
    auto_flush: bool = True
    reuse_threshold: float = 0.65


@dataclass
class VLMPCConfig:
    """Configuration for VLMPC (Vision-Language Model Predictive Control) baseline.

    Reference: Zhao et al., "VLMPC: Vision-Language Model Predictive Control
    for Robotic Manipulation", RSS 2024.
    """

    # VLM settings for action sampling and cost evaluation
    vlm_model_name: str = "gpt-4o"
    vlm_temperature: float = 0.3
    vlm_max_tokens: int = 512

    # MPC planning settings
    planning_horizon: int = 10
    num_action_samples: int = 50
    plan_frequency: int = 3  # Re-plan every N steps
    action_dim: int = 7  # End-effector action dimension (position + orientation)

    # Video prediction settings
    video_prediction_enabled: bool = True
    video_prediction_model: str = "dmvfn"  # "dmvfn" or "simple"
    prediction_frames: int = 8

    # Hierarchical cost function weights
    pixel_cost_weight: float = 0.4
    knowledge_cost_weight: float = 0.6
    smoothness_weight: float = 0.1

    # Action sampling parameters
    action_noise_std: float = 0.1
    action_bounds_low: tuple = (-0.5, -0.5, -0.5, -1.0, -1.0, -1.0, -1.0)
    action_bounds_high: tuple = (0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0)

    # Optimization settings
    num_iterations: int = 3
    elite_fraction: float = 0.1  # Top fraction for CEM-style updates

    additional_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class CoRALConfig:
    """Master configuration grouping all sub-components."""

    vision: FoundationPoseConfig = field(default_factory=FoundationPoseConfig)
    vision_language: VisionLanguageConfig = field(default_factory=VisionLanguageConfig)
    planner: LLMPlannerConfig = field(default_factory=LLMPlannerConfig)
    motion: MotionPlannerConfig = field(default_factory=MotionPlannerConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    controller: ReactiveControllerConfig = field(default_factory=ReactiveControllerConfig)
    metrics: StateMetricsConfig = field(default_factory=StateMetricsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

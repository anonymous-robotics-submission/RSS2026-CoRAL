"""Shared dataclasses and type definitions for the CoRAL pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


@dataclass
class PoseEstimate:
    """6-DoF pose information for a tracked object."""

    object_id: str
    position: Vector3
    orientation: Quaternion
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicalParameters:
    """Physical properties inferred for an object."""

    object_id: str
    mass: float
    friction_coefficient: float
    compliance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateMetrics:
    """Aggregated metrics describing contact-rich execution quality."""

    success_probability: float
    contact_stability: float
    energy: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContactStrategy:
    """LLM-synthesised contact strategy description and cost design."""

    summary: str
    primitives: List[str]
    cost_function: Dict[str, Any]
    constraints: List[str] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class ActionPrimitive:
    """Single motion primitive specification to be executed by the controller."""

    name: str
    parameters: Dict[str, Any]
    expected_duration: Optional[float] = None


@dataclass
class ActionPlan:
    """Sequence of motion primitives produced by the planner."""

    strategy: ContactStrategy
    primitives: Sequence[ActionPrimitive]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanFeedback:
    """Feedback collected from execution to close the planning loop."""

    success: bool
    contact_quality: Optional[float] = None
    observations: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[StateMetrics] = None


@dataclass
class ExperienceRecord:
    """An experience tuple stored in memory for later reuse."""

    task_description: str
    pose: PoseEstimate
    physical_parameters: PhysicalParameters
    strategy: ContactStrategy
    action_plan: ActionPlan
    feedback: PlanFeedback
    tags: List[str] = field(default_factory=list)


@dataclass
class WorldState:
    """Snapshot of the dynamic world model used by the planner."""

    poses: Dict[str, PoseEstimate] = field(default_factory=dict)
    physical_parameters: Dict[str, PhysicalParameters] = field(default_factory=dict)
    last_updated: Optional[float] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Outputs produced by the simulation and MPPI refinement loop."""

    refined_plan: ActionPlan
    updated_parameters: PhysicalParameters
    metrics: StateMetrics
    episodes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MemoryHit:
    """Result of querying the episodic memory."""

    score: float
    record: ExperienceRecord

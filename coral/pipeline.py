"""End-to-end pipeline orchestrating the CoRAL framework."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from .config import (
    CoRALConfig,
    FoundationPoseConfig,
    LLMPlannerConfig,
    MemoryConfig,
    MotionPlannerConfig,
    ReactiveControllerConfig,
    SimulationConfig,
    VisionLanguageConfig,
)
from .controllers import ReactiveController
from .foundation_pose import FoundationPoseTracker
from .language_module import LLMPlanner
from .memory import ExperienceMemory, build_memory_record
from .motion_planner import MotionPlanner
from .simulation import SimulatedWorld
from .state_metrics import StateMetricsEvaluator
from .types import ActionPlan, PlanFeedback, PhysicalParameters, WorldState
from .vision_module import VisionLanguageModel, VisionModule
from .world_model import WorldModel

LOGGER = logging.getLogger(__name__)

SimulationEnvBuilder = Optional[Callable[[], object]]


class CoRALAgent:
    """Main entry-point for running the CoRAL pipeline."""

    def __init__(
        self,
        config: Optional[CoRALConfig] = None,
        *,
        vision_config: Optional[FoundationPoseConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        planner_config: Optional[LLMPlannerConfig] = None,
        motion_config: Optional[MotionPlannerConfig] = None,
        simulation_config: Optional[SimulationConfig] = None,
        controller_config: Optional[ReactiveControllerConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        vlm_generator=None,
        llm_completion_fn=None,
        simulation_env_builder: SimulationEnvBuilder = None,
    ) -> None:
        self._config = config or CoRALConfig()
        vision_cfg = vision_config or self._config.vision
        vl_cfg = vision_language_config or self._config.vision_language
        planner_cfg = planner_config or self._config.planner
        motion_cfg = motion_config or self._config.motion
        simulation_cfg = simulation_config or self._config.simulation
        controller_cfg = controller_config or self._config.controller
        memory_cfg = memory_config or self._config.memory

        self._vision_module = VisionModule(
            tracker=FoundationPoseTracker(vision_cfg),
            vlm=VisionLanguageModel(vl_cfg, generator=vlm_generator),
            config=vision_cfg,
        )
        self._planner = LLMPlanner(planner_cfg, completion_fn=llm_completion_fn)
        self._metrics_evaluator = StateMetricsEvaluator(self._config.metrics)
        self._motion_planner = MotionPlanner(motion_cfg)
        self._simulation = SimulatedWorld(simulation_cfg, motion_cfg, self._metrics_evaluator, simulation_env_builder)
        self._controller = ReactiveController(controller_cfg, self._metrics_evaluator)
        self._memory = ExperienceMemory(memory_cfg)
        self._world_model = WorldModel(WorldState())
        LOGGER.info("CoRAL agent initialised")

    @property
    def world_model(self) -> WorldModel:
        return self._world_model

    def perceive(
        self,
        object_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        task_description: str,
    ):
        previous_pose = self._world_model.get_pose(object_id)
        observation = self._vision_module.observe(
            object_id=object_id,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            task_description=task_description,
            previous_pose=previous_pose,
        )
        self._world_model.update_pose(observation.pose)
        self._world_model.update_physical_parameters(observation.physical_parameters)
        return observation

    def plan(self, object_id: str, task_description: str) -> ActionPlan:
        pose = self._ensure_pose(object_id)
        parameters = self._ensure_parameters(object_id)
        world_state = self._world_model.snapshot()
        memory_hits = self._memory.retrieve(task_description)
        reuse_hit = self._select_memory_hit(memory_hits)
        if reuse_hit is not None:
            plan = self._clone_plan(reuse_hit.record.action_plan)
            parameters = self._blend_parameters(parameters, reuse_hit.record.physical_parameters)
            self._world_model.update_physical_parameters(parameters)
            plan.metadata = {**plan.metadata, "source": "memory", "memory_score": reuse_hit.score}
        else:
            plan = self._planner.plan(
                world_state=world_state,
                pose=pose,
                physical_parameters=parameters,
                task_description=task_description,
                retrieved_memories=memory_hits,
            )
            plan.metadata = {**plan.metadata, "source": "llm"}
        sim_result = self._simulation.refine_plan(plan, world_state, parameters)
        plan = sim_result.refined_plan
        parameters = sim_result.updated_parameters
        self._world_model.update_physical_parameters(parameters)
        plan.metadata.setdefault("simulation_metrics", sim_result.metrics.__dict__)
        if not self._metrics_evaluator.is_success(sim_result.metrics):
            sim_feedback = PlanFeedback(
                success=False,
                contact_quality=sim_result.metrics.contact_stability,
                observations={"simulation": sim_result.episodes},
                metrics=sim_result.metrics,
            )
            plan = self._planner.refine_plan(plan, sim_feedback, world_state, task_description)
        plan = self._motion_planner.optimise_plan(plan, parameters)
        plan.metadata.setdefault("task_description", task_description)
        return plan

    def act(
        self,
        object_id: str,
        plan: ActionPlan,
        environment,
        task_description: str,
    ) -> tuple[PlanFeedback, ActionPlan]:
        feedback = self._controller.execute(plan=plan, environment=environment)
        if feedback.success:
            return feedback, plan
        LOGGER.info("Plan adaptation triggered after unsuccessful execution")
        world_state = self._world_model.snapshot()
        refined_plan = self._planner.refine_plan(plan, feedback, world_state, task_description)
        parameters = self._ensure_parameters(object_id)
        refined_plan = self._motion_planner.optimise_plan(refined_plan, parameters)
        feedback = self._controller.execute(plan=refined_plan, environment=environment)
        return feedback, refined_plan

    def iterate(
        self,
        *,
        object_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        task_description: str,
        environment,
        tags: Optional[list[str]] = None,
    ) -> PlanFeedback:
        observation = self.perceive(object_id, rgb, depth, intrinsics, task_description)
        plan = self.plan(object_id, task_description)
        feedback, executed_plan = self.act(
            object_id=object_id,
            plan=plan,
            environment=environment,
            task_description=task_description,
        )
        record = build_memory_record(
            task_description=task_description,
            pose=observation.pose,
            parameters=observation.physical_parameters,
            action_plan=executed_plan,
            feedback=feedback,
            tags=tags,
        )
        self._memory.add(record)
        return feedback

    def reset(self) -> None:
        LOGGER.info("Resetting CoRAL agent")
        self._vision_module.reset()
        self._world_model = WorldModel(WorldState())

    def _ensure_pose(self, object_id: str):
        pose = self._world_model.get_pose(object_id)
        if pose is None:
            raise RuntimeError(f"Pose for object '{object_id}' is not available. Call perceive() first.")
        return pose

    def _ensure_parameters(self, object_id: str):
        params = self._world_model.get_parameters(object_id)
        if params is None:
            raise RuntimeError(
                f"Physical parameters for object '{object_id}' are not available. Call perceive() first."
            )
        return params

    def _select_memory_hit(self, hits):
        if not hits:
            return None
        best = hits[0]
        if best.score >= self._config.memory.reuse_threshold:
            return best
        return None

    @staticmethod
    def _clone_plan(plan: ActionPlan) -> ActionPlan:
        primitives = [
            replace(primitive, parameters=dict(primitive.parameters))
            for primitive in plan.primitives
        ]
        strategy = replace(plan.strategy, primitives=list(plan.strategy.primitives), cost_function=dict(plan.strategy.cost_function))
        return ActionPlan(strategy=strategy, primitives=primitives, metadata=dict(plan.metadata))

    @staticmethod
    def _blend_parameters(current: PhysicalParameters, remembered: PhysicalParameters) -> PhysicalParameters:
        alpha = 0.5
        compliance_current = current.compliance if current.compliance is not None else remembered.compliance
        compliance_remembered = remembered.compliance if remembered.compliance is not None else compliance_current
        blended_compliance = None
        if compliance_current is not None or compliance_remembered is not None:
            blended_compliance = (float(compliance_current or 0.5) * (1 - alpha)) + (float(compliance_remembered or 0.5) * alpha)
        metadata = dict(current.metadata)
        metadata.setdefault("memory_blends", []).append({
            "mass": remembered.mass,
            "friction": remembered.friction_coefficient,
            "compliance": remembered.compliance,
        })
        return PhysicalParameters(
            object_id=current.object_id,
            mass=(1 - alpha) * current.mass + alpha * remembered.mass,
            friction_coefficient=(1 - alpha) * current.friction_coefficient + alpha * remembered.friction_coefficient,
            compliance=blended_compliance,
            metadata=metadata,
        )

"""Motion planning module that consumes language-derived strategies."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence

from .config import MotionPlannerConfig
from .mppi import MPPIOptimizer
from .types import ActionPlan, ActionPrimitive, ContactStrategy, PhysicalParameters, PlanFeedback

LOGGER = logging.getLogger(__name__)
EvaluateFn = Callable[[Sequence[ActionPrimitive]], tuple[float, Dict[str, float]]]


class MotionPlanner:
    """Contact-aware motion planner that refines LLM strategies."""

    def __init__(self, config: MotionPlannerConfig) -> None:
        self._config = config
        self._mppi = MPPIOptimizer(config)

    def optimise_plan(
        self,
        plan: ActionPlan,
        parameters: PhysicalParameters,
        evaluate: Optional[EvaluateFn] = None,
    ) -> ActionPlan:
        LOGGER.debug("Optimising action plan with MPPI (planner type '%s')", self._config.planner_type)
        if evaluate is None:
            evaluate = lambda primitives: self._default_evaluate(primitives, parameters)
        result = self._mppi.optimise(plan, evaluate)
        return result.refined_plan

    def _default_evaluate(
        self,
        primitives: Sequence[ActionPrimitive],
        parameters: PhysicalParameters,
    ) -> tuple[float, Dict[str, float]]:
        cost = 0.0
        contact_quality = 0.0
        energy = 0.0
        for primitive in primitives:
            params = primitive.parameters
            force = float(params.get("force_limit", parameters.friction_coefficient * 50.0))
            duration = float(params.get("time_step", self._config.time_step))
            energy += force * duration
            contact_quality += 1.0 / (1.0 + force)
            cost += force * duration
        contact_quality = max(contact_quality / max(len(primitives), 1), 1e-3)
        metrics = {
            "contact_quality": contact_quality,
            "energy": energy,
            "success_probability": max(0.05, min(1.0, parameters.friction_coefficient * contact_quality)),
        }
        return cost, metrics

    def update_with_feedback(self, action_plan: ActionPlan, feedback: PlanFeedback) -> ActionPlan:
        LOGGER.debug("Updating action plan based on feedback: success=%s", feedback.success)
        if feedback.success:
            return action_plan
        adjusted_primitives = []
        for primitive in action_plan.primitives:
            new_params = dict(primitive.parameters)
            new_params["force_limit"] = new_params.get("force_limit", 10.0) * 0.9
            new_params["time_step"] = new_params.get("time_step", self._config.time_step) * 1.1
            adjusted_primitives.append(
                ActionPrimitive(
                    name=primitive.name,
                    parameters=new_params,
                    expected_duration=primitive.expected_duration,
                )
            )
        metadata = dict(action_plan.metadata)
        metadata.setdefault("feedback", []).append(feedback.observations)
        strategy = ContactStrategy(
            summary=action_plan.strategy.summary,
            primitives=action_plan.strategy.primitives,
            cost_function=dict(action_plan.strategy.cost_function),
            constraints=action_plan.strategy.constraints,
            confidence=action_plan.strategy.confidence,
        )
        return ActionPlan(strategy=strategy, primitives=adjusted_primitives, metadata=metadata)

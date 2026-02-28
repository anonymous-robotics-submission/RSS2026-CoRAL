"""Reactive controllers that execute contact-rich motion plans."""

from __future__ import annotations

import logging

from .config import ReactiveControllerConfig
from .state_metrics import StateMetricsEvaluator
from .types import ActionPlan, PlanFeedback

LOGGER = logging.getLogger(__name__)


class ReactiveController:
    """Simple impedance-style controller abstraction."""

    def __init__(self, config: ReactiveControllerConfig, metrics_evaluator: StateMetricsEvaluator) -> None:
        self._config = config
        self._metrics = metrics_evaluator

    def execute(self, plan: ActionPlan, environment) -> PlanFeedback:
        """Execute the plan against the provided environment interface."""

        LOGGER.info("Executing plan with %d primitives", len(plan.primitives))
        observations = []
        success = True
        for primitive in plan.primitives:
            LOGGER.debug("Executing primitive %s with params %s", primitive.name, primitive.parameters)
            result = self._execute_primitive(primitive, environment)
            observations.append(result)
            if not result.get("success", True):
                success = False
                LOGGER.warning("Primitive %s failed", primitive.name)
                break
        metrics = self._metrics.from_observations(observations)
        success = success and self._metrics.is_success(metrics)
        feedback = PlanFeedback(
            success=success,
            contact_quality=metrics.contact_stability,
            observations={"primitives": observations},
            metrics=metrics,
        )
        LOGGER.debug("Controller feedback: %s", feedback)
        return feedback

    def _execute_primitive(self, primitive, environment) -> dict:
        handler = getattr(environment, f"execute_{primitive.name}", None)
        if callable(handler):
            return handler(**primitive.parameters)
        LOGGER.warning("Environment does not implement primitive '%s'", primitive.name)
        return {"success": False, "reason": "primitive_not_supported"}

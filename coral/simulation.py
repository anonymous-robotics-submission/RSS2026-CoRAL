"""Simulated-world refinement and parameter adaptation."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import MotionPlannerConfig, SimulationConfig
from .mppi import MPPIOptimizer
from .state_metrics import StateMetricsEvaluator
from .types import ActionPlan, PhysicalParameters, SimulationResult, WorldState

SimulationEnvBuilder = Optional[Callable[[], object]]


class SimulatedWorld:
    """Runs MPPI-based refinement loops inside a simulated environment."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        motion_config: MotionPlannerConfig,
        metrics_evaluator: StateMetricsEvaluator,
        env_builder: SimulationEnvBuilder = None,
    ) -> None:
        self._config = sim_config
        self._metrics = metrics_evaluator
        self._builder = env_builder
        self._mppi = MPPIOptimizer(motion_config)

    def refine_plan(
        self,
        plan: ActionPlan,
        world_state: WorldState,
        parameters: PhysicalParameters,
    ) -> SimulationResult:
        if not self._config.enabled:
            metrics = self._metrics.from_observations([])
            return SimulationResult(refined_plan=plan, updated_parameters=parameters, metrics=metrics, episodes=[])
        env = self._build_environment()
        episodes: List[Dict[str, object]] = []
        refined_plan = plan
        updated_parameters = parameters
        obs_collector: List[dict] = []
        for attempt in range(max(1, self._config.max_attempts)):
            result = self._mppi.optimise(
                refined_plan,
                evaluate=lambda primitives: self._evaluate(primitives, env, world_state, updated_parameters, episodes),
            )
            refined_plan = result.refined_plan
            metrics_info = result.metadata
            obs_collector.append(
                {
                    "success_probability": 1.0 - metrics_info.get("mean_cost", 1.0),
                    "contact_quality": metrics_info.get("info_contact_quality", 0.5),
                    "energy": metrics_info.get("info_energy", 0.0),
                }
            )
            updated_parameters = self._update_parameters(updated_parameters, metrics_info)
            if self._metrics.is_success(self._metrics.from_observations(obs_collector)):
                break
        metrics = self._metrics.from_observations(obs_collector)
        return SimulationResult(
            refined_plan=refined_plan,
            updated_parameters=updated_parameters,
            metrics=metrics,
            episodes=episodes,
        )

    def _build_environment(self) -> Optional[object]:
        if self._builder is None:
            return None
        return self._builder()

    def _evaluate(
        self,
        primitives: Sequence,
        env: Optional[object],
        world_state: WorldState,
        parameters: PhysicalParameters,
        episodes: List[Dict[str, object]],
    ) -> tuple[float, Dict[str, float]]:
        if env is not None and hasattr(env, "simulate_plan"):
            result = env.simulate_plan(
                primitives=primitives,
                parameters=parameters,
                world_state=world_state,
            )
            cost = float(result.get("cost", self._heuristic_cost(primitives, parameters)))
            metrics = dict(result.get("metrics", {}))
            if self._config.log_episodes:
                episodes.append(result)
            return cost, self._ensure_metric_defaults(metrics)
        cost = self._heuristic_cost(primitives, parameters)
        metrics = self._ensure_metric_defaults(self._heuristic_metrics(primitives, parameters, world_state))
        if self._config.log_episodes:
            episodes.append({"cost": cost, "metrics": metrics})
        return cost, metrics

    def _heuristic_cost(self, primitives: Sequence, parameters: PhysicalParameters) -> float:
        energy = 0.0
        for primitive in primitives:
            params = getattr(primitive, "parameters", {})
            force = float(params.get("force_limit", parameters.friction_coefficient * 50.0))
            duration = float(params.get("time_step", 0.05))
            energy += force * duration
        mass_penalty = abs(parameters.mass - 1.0) * 0.05
        compliance_penalty = 0.0 if parameters.compliance is None else abs(parameters.compliance - 0.5)
        return energy + mass_penalty + compliance_penalty

    def _heuristic_metrics(
        self,
        primitives: Sequence,
        parameters: PhysicalParameters,
        world_state: WorldState,
    ) -> Dict[str, float]:
        energy = self._heuristic_cost(primitives, parameters)
        contact_quality = np.clip(1.0 - energy / 10.0, 0.0, 1.0)
        success_probability = np.clip(contact_quality * parameters.friction_coefficient, 0.0, 1.0)
        return {
            "energy": float(energy),
            "contact_quality": float(contact_quality),
            "success_probability": float(success_probability),
        }

    def _update_parameters(self, parameters: PhysicalParameters, metadata: Dict[str, float]) -> PhysicalParameters:
        friction = metadata.get("info_estimated_friction")
        mass = metadata.get("info_estimated_mass")
        compliance = metadata.get("info_estimated_compliance")
        updated = parameters
        if friction is not None:
            updated = replace(
                updated,
                friction_coefficient=self._blend(parameters.friction_coefficient, friction),
            )
        if mass is not None:
            updated = replace(updated, mass=self._blend(parameters.mass, mass))
        if compliance is not None:
            updated = replace(updated, compliance=self._blend(parameters.compliance or 0.5, compliance))
        return updated

    def _blend(self, current: float, observation: float) -> float:
        alpha = np.clip(self._config.parameter_update_rate, 0.0, 1.0)
        return float((1.0 - alpha) * current + alpha * observation)

    @staticmethod
    def _ensure_metric_defaults(metrics: Dict[str, float]) -> Dict[str, float]:
        metrics.setdefault("contact_quality", 0.5)
        metrics.setdefault("success_probability", 0.5)
        metrics.setdefault("energy", 0.0)
        return metrics

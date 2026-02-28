"""Model Predictive Path Integral (MPPI) optimisation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import MotionPlannerConfig
from .types import ActionPlan, ActionPrimitive, ContactStrategy

EvaluationFn = Callable[[Sequence[ActionPrimitive]], Tuple[float, Dict[str, float]]]


@dataclass
class MPPIResult:
    refined_plan: ActionPlan
    metadata: Dict[str, float]


class MPPIOptimizer:
    """Minimal MPPI implementation for refining motion primitives."""

    def __init__(self, config: MotionPlannerConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def optimise(self, plan: ActionPlan, evaluate: EvaluationFn) -> MPPIResult:
        primitives = list(plan.primitives)
        keys = self._infer_numeric_keys(primitives)
        if not keys:
            return MPPIResult(refined_plan=plan, metadata={"samples": 0})
        base_controls = self._extract_controls(primitives, keys)
        sample_count = max(1, self._config.sample_count)
        costs = []
        noises: List[np.ndarray] = []
        infos: List[Dict[str, float]] = []
        for _ in range(sample_count):
            noise = self._rng.normal(0.0, self._config.noise_sigma, size=len(base_controls))
            candidate_controls = base_controls + noise
            candidate = self._apply_controls(primitives, keys, candidate_controls)
            cost, info = evaluate(candidate)
            costs.append(cost)
            noises.append(noise)
            infos.append(info)
        weights = self._compute_weights(costs)
        delta = sum(w * n for w, n in zip(weights, noises)) if weights.size else np.zeros_like(base_controls)
        updated_controls = base_controls + delta
        refined_primitives = self._apply_controls(primitives, keys, updated_controls)
        metadata = {
            "samples": float(sample_count),
            "mean_cost": float(np.mean(costs)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "weight_norm": float(np.sum(weights)),
        }
        if infos:
            aggregated: Dict[str, float] = {}
            for info in infos:
                for key, value in info.items():
                    aggregated[key] = aggregated.get(key, 0.0) + value
            for key in aggregated:
                metadata[f"info_{key}"] = aggregated[key] / sample_count
        refined_plan = ActionPlan(
            strategy=ContactStrategy(
                summary=plan.strategy.summary,
                primitives=plan.strategy.primitives,
                cost_function=dict(plan.strategy.cost_function),
                constraints=list(plan.strategy.constraints),
                confidence=plan.strategy.confidence,
            ),
            primitives=refined_primitives,
            metadata={**plan.metadata, "mppi": metadata},
        )
        return MPPIResult(refined_plan=refined_plan, metadata=metadata)

    def _infer_numeric_keys(self, primitives: Sequence[ActionPrimitive]) -> List[Tuple[int, str]]:
        keys: List[Tuple[int, str]] = []
        for idx, primitive in enumerate(primitives):
            for key, value in primitive.parameters.items():
                if isinstance(value, (int, float)):
                    keys.append((idx, key))
        return keys

    @staticmethod
    def _extract_controls(primitives: Sequence[ActionPrimitive], keys: Sequence[Tuple[int, str]]) -> np.ndarray:
        values = [float(primitives[idx].parameters[key]) for idx, key in keys]
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _apply_controls(
        primitives: Sequence[ActionPrimitive],
        keys: Sequence[Tuple[int, str]],
        controls: np.ndarray,
    ) -> List[ActionPrimitive]:
        updated: List[ActionPrimitive] = []
        for idx, primitive in enumerate(primitives):
            params = dict(primitive.parameters)
            updated.append(
                ActionPrimitive(
                    name=primitive.name,
                    parameters=params,
                    expected_duration=primitive.expected_duration,
                )
            )
        for (prim_idx, key), value in zip(keys, controls):
            updated[prim_idx].parameters[key] = float(value)
        return updated

    def _compute_weights(self, costs: Sequence[float]) -> np.ndarray:
        costs_array = np.asarray(costs, dtype=np.float64)
        min_cost = np.min(costs_array)
        exponent = -(costs_array - min_cost) / max(self._config.temperature_lambda, 1e-6)
        weights = np.exp(exponent)
        normaliser = np.sum(weights)
        if not np.isfinite(normaliser) or normaliser <= 0.0:
            return np.zeros_like(weights)
        return weights / normaliser

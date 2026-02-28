"""Utilities for evaluating contact-rich manipulation metrics."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable, Optional

from .config import StateMetricsConfig
from .types import StateMetrics


class StateMetricsEvaluator:
    """Computes aggregate metrics and success decisions."""

    def __init__(self, config: StateMetricsConfig) -> None:
        self._config = config

    def from_observations(self, observations: Iterable[dict], *, fallback_quality: float = 0.5) -> StateMetrics:
        successes = []
        stabilities = []
        energies = []
        additional = []
        for obs in observations:
            if not obs:
                continue
            successes.append(self._extract_success_prob(obs))
            stabilities.append(float(obs.get("contact_quality", fallback_quality)))
            energies.append(float(obs.get("energy", 0.0)))
            extra = {k: v for k, v in obs.items() if k not in {"success", "contact_quality", "energy", "success_probability"}}
            if extra:
                additional.append(extra)
        success_probability = fmean(successes) if successes else 0.0
        contact_stability = fmean(stabilities) if stabilities else fallback_quality
        energy = float(sum(energies))
        additional_metrics = {f"episode_{idx}": data for idx, data in enumerate(additional)}
        return StateMetrics(
            success_probability=success_probability,
            contact_stability=contact_stability,
            energy=energy,
            additional_metrics=additional_metrics,
        )

    def is_success(self, metrics: StateMetrics) -> bool:
        return (
            metrics.success_probability >= self._config.success_threshold
            and metrics.contact_stability >= self._config.stability_threshold
            and metrics.energy <= self._config.energy_limit
        )

    def score(self, metrics: StateMetrics) -> float:
        return (
            self._config.success_weight * metrics.success_probability
            + self._config.stability_weight * metrics.contact_stability
            + self._config.energy_weight * (1.0 - min(metrics.energy / max(self._config.energy_limit, 1e-3), 1.0))
        )

    @staticmethod
    def _extract_success_prob(obs: dict) -> float:
        if "success_probability" in obs:
            return float(obs["success_probability"])
        if "success" in obs:
            return 1.0 if obs["success"] else 0.0
        return 0.5

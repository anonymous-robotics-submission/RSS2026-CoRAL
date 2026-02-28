"""Memory module for storing and retrieving contact-rich manipulation experiences."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import MemoryConfig
from .types import (
    ActionPlan,
    ActionPrimitive,
    ContactStrategy,
    ExperienceRecord,
    MemoryHit,
    PhysicalParameters,
    PlanFeedback,
    PoseEstimate,
    StateMetrics,
)

LOGGER = logging.getLogger(__name__)


class ExperienceMemory:
    """Stores and retrieves manipulation experiences for zero-shot transfer."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._storage_path = config.storage_path
        self._entries: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not self._storage_path.exists():
            LOGGER.info("No existing memory file at %s", self._storage_path)
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as handle:
                self._entries = json.load(handle)
            LOGGER.info("Loaded %d experiences from memory", len(self._entries))
        except json.JSONDecodeError:
            LOGGER.warning("Memory file %s is corrupted; starting fresh", self._storage_path)
            self._entries = []

    def add(self, record: ExperienceRecord) -> None:
        LOGGER.debug("Adding experience to memory for task: %s", record.task_description)
        self._entries.append(self._serialise_record(record))
        if len(self._entries) > self._config.max_entries:
            self._entries = self._entries[-self._config.max_entries :]
        if self._config.auto_flush:
            self.flush()

    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryHit]:
        LOGGER.debug("Retrieving experiences matching query: %s", query)
        scored = [
            (self._similarity(query, entry.get("task_description", "")), entry)
            for entry in self._entries
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: List[MemoryHit] = []
        for score, entry in scored[:top_k]:
            if score <= 0.0:
                continue
            hits.append(MemoryHit(score=score, record=self._deserialise_record(entry)))
        return hits

    def flush(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._storage_path.open("w", encoding="utf-8") as handle:
            json.dump(self._entries, handle, indent=2)
        LOGGER.info("Persisted %d experiences to %s", len(self._entries), self._storage_path)

    def _serialise_record(self, record: ExperienceRecord) -> Dict:
        data = asdict(record)
        for obj in (data["pose"], data["physical_parameters"]):
            obj.get("metadata", {}).pop("raw", None)
        return data

    def _deserialise_record(self, data: Dict) -> ExperienceRecord:
        pose = PoseEstimate(**data["pose"])
        parameters = PhysicalParameters(**data["physical_parameters"])
        strategy = ContactStrategy(**data["strategy"])
        primitives = [ActionPrimitive(**primitive) for primitive in data["action_plan"]["primitives"]]
        action_plan = ActionPlan(
            strategy=strategy,
            primitives=primitives,
            metadata=data["action_plan"].get("metadata", {}),
        )
        feedback_metrics = None
        if data["feedback"].get("metrics") is not None:
            feedback_metrics = StateMetrics(**data["feedback"]["metrics"])
        feedback = PlanFeedback(
            success=data["feedback"]["success"],
            contact_quality=data["feedback"].get("contact_quality"),
            observations=data["feedback"].get("observations", {}),
            metrics=feedback_metrics,
        )
        return ExperienceRecord(
            task_description=data["task_description"],
            pose=pose,
            physical_parameters=parameters,
            strategy=strategy,
            action_plan=action_plan,
            feedback=feedback,
            tags=list(data.get("tags", [])),
        )

    @staticmethod
    def _similarity(query: str, text: str) -> float:
        if not query or not text:
            return 0.0
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())
        intersection = len(query_tokens & text_tokens)
        union = len(query_tokens | text_tokens)
        return intersection / union if union else 0.0


def build_memory_record(
    task_description: str,
    pose: PoseEstimate,
    parameters: PhysicalParameters,
    action_plan: ActionPlan,
    feedback: PlanFeedback,
    tags: Optional[Iterable[str]] = None,
) -> ExperienceRecord:
    """Convenience helper to assemble an ExperienceRecord."""

    if tags is None:
        tags = []
    return ExperienceRecord(
        task_description=task_description,
        pose=pose,
        physical_parameters=parameters,
        strategy=action_plan.strategy,
        action_plan=action_plan,
        feedback=feedback,
        tags=list(tags),
    )

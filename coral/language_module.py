"""Language module that synthesises contact strategies via an LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import LLMPlannerConfig
from .types import (
    ActionPlan,
    ActionPrimitive,
    ContactStrategy,
    MemoryHit,
    PhysicalParameters,
    PlanFeedback,
    PoseEstimate,
    WorldState,
)

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency for default LLM access
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


class LLMPlanner:
    """Interface for generating contact strategies and cost functions using an LLM."""

    def __init__(
        self,
        config: LLMPlannerConfig,
        completion_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        self._config = config
        if completion_fn is not None:
            self._completion_fn = completion_fn
            self._client = None
            LOGGER.info("Using custom LLM completion function for planner")
        else:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required to instantiate the default LLM"
                    " planner. Provide a custom completion function if you cannot"
                    " install it or prefer a different provider."
                )
            LOGGER.info("Initialising OpenAI client for model '%s'", config.model_name)
            self._client = OpenAI()
            self._completion_fn = None

    def plan(
        self,
        world_state: WorldState,
        pose: PoseEstimate,
        physical_parameters: PhysicalParameters,
        task_description: str,
        retrieved_memories: Optional[Sequence[MemoryHit]] = None,
    ) -> ActionPlan:
        prompt = self._build_prompt(
            world_state=world_state,
            pose=pose,
            physical_parameters=physical_parameters,
            task_description=task_description,
            retrieved_memories=retrieved_memories,
        )
        LOGGER.debug("Planner prompt: %s", prompt)
        response = self._generate_completion(prompt)
        LOGGER.debug("Planner raw response: %s", response)
        strategy, primitives, metadata = self._parse_response(response)
        return ActionPlan(strategy=strategy, primitives=primitives, metadata=metadata)

    def refine_plan(
        self,
        previous_plan: ActionPlan,
        feedback: PlanFeedback,
        world_state: WorldState,
        task_description: str,
    ) -> ActionPlan:
        prompt = self._build_refinement_prompt(previous_plan, feedback, world_state, task_description)
        LOGGER.debug("Refinement prompt: %s", prompt)
        response = self._generate_completion(prompt)
        LOGGER.debug("Refinement raw response: %s", response)
        strategy, primitives, metadata = self._parse_response(response)
        return ActionPlan(strategy=strategy, primitives=primitives, metadata=metadata)

    def _build_prompt(
        self,
        world_state: WorldState,
        pose: PoseEstimate,
        physical_parameters: PhysicalParameters,
        task_description: str,
        retrieved_memories: Optional[Sequence[MemoryHit]],
    ) -> str:
        world_state_json = json.dumps(
            {
                "poses": {k: v.__dict__ for k, v in world_state.poses.items()},
                "physical_parameters": {
                    k: v.__dict__ for k, v in world_state.physical_parameters.items()
                },
            }
        )
        memories_text = ""
        if retrieved_memories:
            snippets = [self._format_memory_snippet(hit) for hit in retrieved_memories]
            memories_text = "Relevant prior episodes:\n" + "\n".join(snippets)
        prompt = (
            f"{self._config.system_prompt}\n"
            "Respond with JSON containing keys 'strategy', 'primitives', and"
            " 'cost_function'. The 'strategy' entry must include a summary,"
            " list of contact primitives, constraints, and confidence. The"
            " 'primitives' entry must be an ordered list of motion primitives with"
            " names and parameters. The 'cost_function' should list scalar weights"
            " for objectives such as force_regulation, pose_error, and energy.\n"
            f"Task description: {task_description}\n"
            f"Pose estimate: {pose.__dict__}\n"
            f"Physical parameters: {physical_parameters.__dict__}\n"
            f"World state: {world_state_json}\n"
            f"{memories_text}\n"
            "Return valid JSON only."
        )
        return prompt

    def _build_refinement_prompt(
        self,
        previous_plan: ActionPlan,
        feedback: PlanFeedback,
        world_state: WorldState,
        task_description: str,
    ) -> str:
        feedback_dict = {
            "success": feedback.success,
            "contact_quality": feedback.contact_quality,
            "metrics": feedback.metrics.__dict__ if feedback.metrics else None,
            "observations": feedback.observations,
        }
        prompt = (
            f"{self._config.adaptation_prompt}\n"
            "You will refine an existing plan described below. Respond with JSON"
            " matching the schema from previous instructions.\n"
            f"Task description: {task_description}\n"
            f"Current plan: {previous_plan.metadata}\n"
            f"Strategy summary: {previous_plan.strategy.summary}\n"
            f"Primitives: {[primitive.__dict__ for primitive in previous_plan.primitives]}\n"
            f"Feedback: {json.dumps(feedback_dict)}\n"
            f"World snapshot: {json.dumps({'history': world_state.history[-5:]})}\n"
            "Return valid JSON only."
        )
        return prompt

    def _generate_completion(self, prompt: str) -> str:
        if self._completion_fn is not None:
            return self._completion_fn({"prompt": prompt})
        if self._client is None:
            raise RuntimeError("LLM planner is not properly initialised")
        completion = self._client.responses.create(
            model=self._config.model_name,
            input=prompt,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
            top_p=self._config.top_p,
            presence_penalty=self._config.presence_penalty,
            frequency_penalty=self._config.frequency_penalty,
            **self._config.additional_kwargs,
        )
        return completion.output_text  # type: ignore[attr-defined]

    def _parse_response(
        self, response: str
    ) -> tuple[ContactStrategy, List[ActionPrimitive], Dict[str, Any]]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            LOGGER.warning("Planner response not JSON, attempting regex parsing")
            data = self._fallback_parse(response)
        strategy = self._build_strategy(data)
        primitives = [
            ActionPrimitive(name=item.get("name", "unknown"), parameters=item.get("parameters", {}),)
            for item in data.get("primitives", [])
        ]
        metadata = {k: v for k, v in data.items() if k not in {"strategy", "primitives"}}
        return strategy, primitives, metadata

    def _build_strategy(self, data: Dict[str, Any]) -> ContactStrategy:
        strategy_data = data.get("strategy", {})
        cost_function = data.get("cost_function", {})
        return ContactStrategy(
            summary=strategy_data.get("summary", ""),
            primitives=strategy_data.get("primitives", []),
            constraints=strategy_data.get("constraints", []),
            confidence=strategy_data.get("confidence"),
            cost_function=cost_function,
        )

    @staticmethod
    def _fallback_parse(response: str) -> Dict[str, Any]:
        primitives_matches = re.findall(r"-\s*Primitive:\s*(.+)", response)
        primitives = []
        for item in primitives_matches:
            primitives.append({"name": item.strip(), "parameters": {}})
        return {
            "strategy": {
                "summary": response.split("\n")[0][:256],
                "primitives": primitives_matches,
                "constraints": [],
                "confidence": None,
            },
            "primitives": primitives,
            "cost_function": {},
        }

    @staticmethod
    def _format_memory_snippet(hit: MemoryHit) -> str:
        record = hit.record
        summary = record.strategy.summary or "(no summary)"
        return json.dumps(
            {
                "score": hit.score,
                "summary": summary,
                "cost": record.feedback.observations.get("cost"),
                "plan": [primitive.parameters for primitive in record.action_plan.primitives],
            }
        )

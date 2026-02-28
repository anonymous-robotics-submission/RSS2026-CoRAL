# CoRAL Framework

This module provides a research-oriented implementation of the CoRAL (Contact-Rich Vision-Language-Action) framework described in the accompanying article. The framework explicitly separates perception, reasoning, simulation, motion planning, control, and memory so that complex manipulation behaviours can be adapted, explained, and reused across tasks.

## Architecture Overview

1. **Vision Module** (`vision_module.py`)
   - Wraps [FoundationPose](https://github.com/NVlabs/FoundationPose) for RGB-D based 6-DoF pose tracking.
   - Queries a vision-language model (VLM) to infer physical parameters (mass, friction, compliance) conditioned on the pose and task description.
2. **Language Module** (`language_module.py`)
   - Uses a large language model (LLM) for task formulation and online adaptation of contact strategies, motion primitives, and cost functions.
   - Accepts custom completion callbacks or the OpenAI API by default.
3. **World Model** (`world_model.py`)
   - Maintains a dynamic scene representation that fuses perception streams, parameter updates, and historical context for planning.
4. **Simulated World + MPPI** (`simulation.py`, `mppi.py`)
   - Runs a configurable MPPI optimisation loop inside a simulated environment (or a heuristic surrogate) to refine plans and update physical parameters before execution.
5. **Motion Planner** (`motion_planner.py`)
   - Applies MPPI refinement and feedback-driven adjustments to produce executable motion plans.
6. **State Metrics** (`state_metrics.py`)
   - Aggregates contact stability, success likelihood, and energy metrics used for success evaluation and adaptive prompting.
7. **Reactive Controller** (`controllers.py`)
   - Executes motion primitives on the real system, gathers per-primitive observations, and reports metrics for evaluation.
8. **Experience Memory** (`memory.py`)
   - Stores episodic experiences, retrieves high-confidence matches, and enables plan reuse without invoking the LLM when appropriate.
9. **Pipeline** (`pipeline.py`)
   - The `CoRALAgent` orchestrates perceive → simulate → plan → act → adapt → remember, mirroring the full flow in the paper.

## Getting Started

```python
import numpy as np
from coral import CoRALAgent, CoRALConfig

agent = CoRALAgent(CoRALConfig())

feedback = agent.iterate(
    object_id="target_object",
    rgb=np.zeros((480, 640, 3), dtype=np.uint8),
    depth=np.zeros((480, 640), dtype=np.float32),
    intrinsics=np.eye(3),
    task_description="Insert the peg into the angled slot without scratching the surface.",
    environment=your_robot_env,
)

print(feedback)
```

To plug in a high-fidelity simulator for the MPPI stage, provide a `simulation_env_builder` that returns an object exposing `simulate_plan(primitives, parameters, world_state)` when constructing `CoRALAgent`.

### FoundationPose Setup

Install FoundationPose and download the appropriate checkpoints if you have not already:

```bash
pip install foundationpose
# or follow instructions from the official repository
```

Pass the checkpoint path through `FoundationPoseConfig` if it differs from the default directory.

### Vision-Language and LLM Backends

- For the **vision-language module**, either supply a custom generator callback or install `transformers` and `torch` to automatically load a multi-modal model such as `llava` via `VisionLanguageConfig`.
- For the **LLM planner**, you can rely on the OpenAI API (`openai` Python package required) or inject a custom completion function to integrate with your model of choice.

### Memory and Metrics

- Experiences are persisted at `~/.cache/coral_memory.json` by default. Update `MemoryConfig` to customise the location, control automatic flushing, or adjust the similarity threshold for plan reuse.
- Tune `StateMetricsConfig` and `SimulationConfig` to align success criteria between simulation and the real robot.

## Notes

- The default simulated-world fallback uses analytic heuristics; swap in a domain simulator for higher fidelity.
- Environment integrations should implement methods such as `execute_<primitive>(**parameters)` for physical rollouts and optionally `simulate_plan(...)` for MPPI.
- The motion planner, controller, and memory modules are structured to let you slot in domain-specific optimisers while retaining the high-level loop.

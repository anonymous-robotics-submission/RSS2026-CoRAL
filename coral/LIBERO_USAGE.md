# Using `coral` with Existing LIBERO Tasks

This guide shows how to connect the CoRAL agent to LIBERO manipulation tasks defined in BDDL files (e.g. `my_suite/flip_the_blue_box_onto_its_side.bddl`). The goal is to reuse your existing environments while taking advantage of the perception→simulation→planning→control loop implemented in `coral`.

## 1. Prerequisites

- LIBERO repository installed in editable mode (`pip install -e .`).
- Dependencies for the CoRAL stack:
  - [FoundationPose](https://github.com/NVlabs/FoundationPose) + checkpoints for object pose estimation.
  - A vision-language backbone (install `torch` + `transformers`, or provide a custom generator).
  - Access to the LLM backend used for planning (OpenAI client or a custom completion function).
- RGB-D observations and camera intrinsics from the simulator or real robot.
- Python 3.8+.

## 2. Instantiate the LIBERO Environment

LIBERO exposes task loaders that read BDDL definitions. For example:

```python
from libero.libero.utils.init_loader import load_suite

suite = load_suite("my_suite")
env, task_meta = suite.build_env(task_name="flip_the_blue_box_onto_its_side")
```

The returned `env` should provide access to RGB-D observations and allow you to execute low-level actions (e.g. delta end-effector commands). You will wrap this environment to expose the primitives expected by `coral`.

## 3. Wrap the Environment with Motion Primitives

`coral.controllers.ReactiveController` expects the environment to implement callables named `execute_<primitive>` that consume the parameters emitted by the planner. Build a thin adapter:

```python
class LiberoPrimitiveAdapter:
    def __init__(self, env):
        self._env = env

    def reset(self):
        return self._env.reset()

    def get_rgbd(self):
        obs = self._env.observe()
        return obs["rgb"], obs["depth"], obs["intrinsics"]

    # Primitive: impedance-style push with target pose
    def execute_push(self, target_pose, force_limit, time_step, **_):
        result = self._env.apply_push(target_pose, force_limit, time_step)
        return {
            "success": result.success,
            "contact_quality": result.contact_quality,
            "energy": result.energy,
            "observations": result.info,
        }

    # Add `execute_grasp`, `execute_slide`, etc. as needed.
```

Match primitive names with those your LLM prompt produces (`push`, `slide`, `grasp`, etc.). Each handler should return a dictionary containing at least `success`, optional `contact_quality`, `energy`, and any additional telemetry.

## 4. Provide Vision Inputs

During each iteration you must provide the RGB-D data and camera intrinsics from the environment to the agent:

```python
adapter = LiberoPrimitiveAdapter(env)
obs = adapter.reset()

rgb, depth, K = adapter.get_rgbd()
```

If your environment already tracks object poses, you can seed FoundationPose with segmentation masks or object IDs via `FoundationPoseConfig.additional_kwargs`.

## 5. Optional: Simulation Builder for MPPI

If you have a high-fidelity simulator separate from the physical environment, supply a builder when creating the agent:

```python
def build_sim_world():
    sim_env, _ = suite.build_sim_env("flip_the_blue_box_onto_its_side")
    return sim_env  # must implement simulate_plan(primitives, parameters, world_state)
```

If omitted, `coral` falls back to analytical heuristics for MPPI evaluation.

## 6. Run the CoRAL Loop

```python
import numpy as np
from coral import CoRALAgent, CoRALConfig

agent = CoRALAgent(
    CoRALConfig(),
    simulation_env_builder=build_sim_world,  # optional
)

vision_obs = agent.perceive(
    object_id="blue_box",
    rgb=rgb,
    depth=depth,
    intrinsics=K,
    task_description="Flip the blue box onto its side.",
)

plan = agent.plan(object_id="blue_box", task_description="Flip the blue box onto its side.")
feedback, executed_plan = agent.act(
    object_id="blue_box",
    plan=plan,
    environment=adapter,
    task_description="Flip the blue box onto its side.",
)

print("Success:", feedback.success, feedback.metrics)
```

For compact use, call `agent.iterate(...)`, which performs perceive→plan→act and logs the episode to memory.

## 7. Feeding BDDL Metadata into Prompts

Many BDDL tasks contain rich textual descriptions (initial object states, goals, constraints). Pass this information to the planner by composing a task description string. For example:

```python
task_description = task_meta["natural_language_goal"]
feedback = agent.iterate(
    object_id="blue_box",
    rgb=rgb,
    depth=depth,
    intrinsics=K,
    task_description=task_description,
    environment=adapter,
)
```

## 8. Memory Warm Start

CoRAL stores episodes under `~/.cache/coral_memory.json`. When replaying the same task suite, the agent will reuse high-confidence plans (score ≥ `reuse_threshold`). Use task-specific tags when calling `agent.iterate(..., tags=["my_suite", "blue_box"])` so that later retrievals stay focused.

## 9. Troubleshooting Tips

- **LLM output not JSON**: enable debug logging (`logging.basicConfig(level=logging.DEBUG)`) to inspect prompts/responses. Adjust system prompts or provide a stricter completion function.
- **Primitive mismatch**: ensure the planner’s prompt mentions only primitives you implemented in the adapter.
- **Perception drift**: adjust `FoundationPoseConfig.track_history` and supply segmentation masks when available.
- **Simulation heuristic too coarse**: implement `simulate_plan` on your LIBERO sim to return realistic cost/metric dictionaries for MPPI.

## 10. Extending to New Tasks

To add another BDDL task:
1. Load the relevant suite and instantiate the environment.
2. Update the task description string.
3. Implement any new primitives required by that task in the adapter.
4. Optionally pre-fill memory by running `agent.iterate` multiple times and storing successful episodes.

With these steps, the CoRAL agent can operate on LIBERO’s existing environments while preserving the modular flow described in the paper.

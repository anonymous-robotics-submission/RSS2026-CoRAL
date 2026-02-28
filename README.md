# LIBERO + CoRAL Extensions

This repository is a fork of [LIBERO](https://libero-project.github.io/), a benchmark for lifelong robot learning. It keeps the original task suites, BDDL assets, and evaluation tooling while introducing CoRAL, a contact-rich Vision-Language-Action (VLA) framework that combines vision, large language models, simulation, motion planning, and reactive control for dynamic manipulation. The fork also adds a set of utility scripts and environment tweaks to support contact-oriented experimentation.

## What's New in This Fork

- **CoRAL Framework (`coral/`)**
  - FoundationPose-based perception with vision-language physical parameter estimation.
  - LLM task formulation and online refinement loops aligned with the article's architecture.
  - Simulated-world MPPI optimisation and metric-driven success evaluation.
  - Reactive controller, episodic memory, and pipeline orchestration for perceive → simulate → plan → act → learn.
- **LIBERO Environment Updates**
  - New object primitives and static assets for contact-rich tasks.
  - Additional BDDL task definitions (e.g., `my_suite/flip_the_blue_box_onto_its_side.bddl`).
  - Utility scripts for inspecting friction, density, and custom suite initialisations.
- **Documentation**
  - `coral/README.md` for module internals.
  - `coral/LIBERO_USAGE.md` for integrating the agent with LIBERO tasks.

## Retained from Upstream LIBERO

- Benchmark suites, evaluation scripts, and training pipelines.
- BDDL task library and templating system.
- Lifelong learning tooling (`libero/lifelong/`) and configuration utilities.

## Repository Structure

| Path | Description |
| --- | --- |
| `coral/` | CoRAL agent, configs, MPPI, simulated world, memory, and docs. |
| `libero/` | Upstream LIBERO codebase with task suites and environment definitions. |
| `datasets/`, `renders/`, `scripts/` | Existing LIBERO assets and helper scripts (some extended for contact tasks). |
| `coral/LIBERO_USAGE.md` | Step-by-step guide for running the agent on LIBERO BDDL tasks. |

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -e .
   pip install foundationpose torch transformers openai
   ```
   Adjust the package list if you rely on alternative VLM/LLM backends.

2. **Download FoundationPose checkpoints** and set the path in `coral/config.py` (`FoundationPoseConfig.checkpoint_path`).

3. **Configure LLM access**
   - Default planner expects OpenAI API credentials (`OPENAI_API_KEY`).
   - Provide a custom completion function if you use a different provider.

4. **(Optional) Provide a simulation builder** for MPPI by passing `simulation_env_builder` when constructing the agent.

## Using CoRAL with LIBERO Tasks

See `coral/LIBERO_USAGE.md` for a full walkthrough. A minimal example:

```python
import numpy as np
from coral import CoRALAgent, CoRALConfig
from libero.libero.utils.init_loader import load_suite

suite = load_suite("my_suite")
env, task_meta = suite.build_env("flip_the_blue_box_onto_its_side")
adapter = LiberoPrimitiveAdapter(env)  # implement execute_<primitive> handlers

agent = CoRALAgent(CoRALConfig())

rgb, depth, K = adapter.get_rgbd()
feedback = agent.iterate(
    object_id="blue_box",
    rgb=rgb,
    depth=depth,
    intrinsics=K,
    task_description=task_meta["natural_language_goal"],
    environment=adapter,
    tags=["my_suite", "blue_box"],
)
print("Success:", feedback.success, feedback.metrics)
```

The adapter example and additional tips are detailed in `coral/LIBERO_USAGE.md`.

## Contribution Summary

- **New**: contact-rich VLA architecture, simulated-world MPPI loop, on-line LLM refinement, episodic memory, documentation, and LIBERO task augmentations.
- **Existing**: original LIBERO benchmark tasks, evaluation scripts, lifelong learning toolkit, and dataset loaders.

## Roadmap

- Integrate higher-fidelity simulators for the MPPI stage.
- Expand primitive coverage for additional LIBERO suites.
- Add evaluation scripts comparing baseline LIBERO policies with CoRAL agents.

## Acknowledgements

- FoundationPose and the developers of the VLM/LLM backends leveraged in this project.

If you build on this fork, please cite both LIBERO and the CoRAL paper.

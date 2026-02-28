#!/usr/bin/env python3
"""
Save a `.pruned_init` file for a my_suite task using hand-crafted placements.

Two ways to set variables:
1) Edit CONFIG below in your editor and run the file
2) (Optional) Provide CLI args to override the task only

Edit placements in scripts/my_suite_positions.py.
This writes to the configured `init_states` directory from LIBERO's config,
under `my_suite/<task>.pruned_init`. A sidecar JSON `.meta.json` is also saved
to record the object overrides and placements for reproducibility.
"""

import argparse
import json
import os
from typing import List, Optional

import torch
import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

from scripts.my_suite_positions import TASK_POSITIONS


# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG = {
    "task": "push_the_blue_box_to_the_front_of_the_wall",
    # Optional per-object overrides applied at env construction time.
    # Keys are object instance names from BDDL (e.g., "block_1").
    # Example: set a smaller, lighter block.
    "overrides": {
        # "block_1": {"size": [0.06, 0.06, 0.06], "density": 200},
    },#if the overrides are set in my_suite_positions.py, this will be overridden
}
# ------------------------------------------------------


def bddl_path(task: str) -> str:
    return os.path.join(
        get_libero_path("bddl_files"),
        "my_suite",
        f"{task}.bddl",
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_block_pose(env: OffScreenRenderEnv, obj_name: str, xy: List[float], yaw: Optional[float] = None):
    geom = env.env.object_states_dict[obj_name].get_geom_state()
    pos = np.array(geom["pos"]).copy()
    quat = np.array(geom["quat"]).copy()

    pos[:2] = np.array(xy, dtype=float)

    if yaw is not None:
        # Yaw-only quaternion (w, x, y, z)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        quat = np.array([cy, 0.0, 0.0, sy], dtype=float)

    joint = env.env.get_object(obj_name).joints[-1]
    env.env.sim.data.set_joint_qpos(joint, np.concatenate([pos, quat]))
    env.env.sim.forward()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default=CONFIG["task"], help="Task name in my_suite")
    args, _ = parser.parse_known_args()

    task = args.task
    assert task in TASK_POSITIONS, f"No placements configured for task: {task}"
    placements = TASK_POSITIONS[task]

    states = []
    overrides_per_state = []
    for p in placements:
        overrides = p.get("overrides", CONFIG.get("overrides", {}))
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path(task),
            camera_names=["agentview"],
            camera_heights=128,
            camera_widths=128,
            camera_depths=False,
            horizon=1000,
            robots=["Panda"],
            controller="OSC_POSE",
            object_overrides=overrides,
        )
        env.seed(0)
        env.reset()
        set_block_pose(env, obj_name="block_1", xy=p["xy"], yaw=p.get("yaw"))
        # Ensure observations / caches are up to date, then record full sim state
        env.regenerate_obs_from_state(env.get_sim_state())
        states.append(env.get_sim_state())
        overrides_per_state.append(overrides)
        env.close()

    init_root = get_libero_path("init_states")
    out_dir = os.path.join(init_root, "my_suite")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{task}.pruned_init")
    torch.save(states, out_path)

    # Save sidecar metadata for reproducibility
    meta = {
        "task": task,
        "num_states": len(states),
        # Keep per-state overrides and placements to recreate envs faithfully
        "overrides_per_state": overrides_per_state,
        "placements": placements,
    }
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(states)} init states to: {out_path}")
    print(f"Saved metadata to: {out_path}.meta.json")


if __name__ == "__main__":
    main()

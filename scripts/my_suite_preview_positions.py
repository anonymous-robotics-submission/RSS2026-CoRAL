#!/usr/bin/env python3
"""
Preview manually specified placements for a my_suite task by saving screenshots.

Two ways to set variables:
1) Edit the CONFIG dict below in your editor and just run the file.
2) (Optional) Provide CLI args to override CONFIG when desired.

Edit placements in scripts/my_suite_positions.py.
This renders one image per placement to help you visually verify the setup.
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import imageio.v2 as imageio

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

from scripts.my_suite_positions import TASK_POSITIONS


# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG = {
    "task": "push_the_blue_box_to_the_front_of_the_wall",
    "out": "renders/my_suite",
    "camera": "frontview",  # e.g., agentview | frontview | galleryview
    "height": 256,
    "width": 256,
    # Optional per-object overrides applied at env construction time.
    # Keys are object instance names from BDDL (e.g., "block_1").
    # Example: set a smaller, lighter block.
    "overrides": {
        # "block_1": {"size": [0.06, 0.06, 0.06], "density": 200},
    },
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
    # Read current pose to keep z and orientation if yaw not specified
    geom = env.env.object_states_dict[obj_name].get_geom_state()
    pos = np.array(geom["pos"]).copy()
    quat = np.array(geom["quat"]).copy()

    pos[:2] = np.array(xy, dtype=float)

    if yaw is not None:
        # Construct yaw-only quaternion (w, x, y, z)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        # yaw about z-axis
        quat = np.array([cy, 0.0, 0.0, sy], dtype=float)

    joint = env.env.get_object(obj_name).joints[-1]
    env.env.sim.data.set_joint_qpos(joint, np.concatenate([pos, quat]))
    env.env.sim.forward()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default=CONFIG["task"], help="Task name in my_suite")
    parser.add_argument("--out", default=CONFIG["out"], help="Output folder for screenshots")
    parser.add_argument("--camera", default=CONFIG["camera"], help="Which camera to save")
    parser.add_argument("--height", type=int, default=CONFIG["height"], help="Camera height")
    parser.add_argument("--width", type=int, default=CONFIG["width"], help="Camera width")
    # Parse known args but allow running without any CLI usage
    args, _ = parser.parse_known_args()

    task = args.task
    assert task in TASK_POSITIONS, f"No placements configured for task: {task}"
    placements = TASK_POSITIONS[task]

    out_dir = os.path.join(args.out, task)
    ensure_dir(out_dir)

    for i, p in enumerate(placements):
        # Build env per placement to honor per-initialization overrides
        overrides = p.get("overrides", CONFIG.get("overrides", {}))
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path(task),
            camera_names=[args.camera],
            camera_heights=args.height,
            camera_widths=args.width,
            camera_depths=False,
            horizon=1000,
            robots=["Panda"],
            controller="OSC_POSE",
            object_overrides=overrides,
        )
        env.seed(0)
        env.reset()
        set_block_pose(env, obj_name="block_1", xy=p["xy"], yaw=p.get("yaw"))
        # Generate observations from the current sim state
        obs = env.regenerate_obs_from_state(env.get_sim_state())
        key = f"{args.camera}_image"
        assert key in obs, f"Camera key {key} not in observations. Available: {list(obs.keys())}"
        img = obs[key]
        # Rotate 180 degrees prior to saving
        img_rot = np.rot90(img, 2).copy()
        imageio.imwrite(os.path.join(out_dir, f"{i:02d}.png"), img_rot)
        env.close()
    print(f"Saved {len(placements)} previews to {out_dir}")


if __name__ == "__main__":
    main()

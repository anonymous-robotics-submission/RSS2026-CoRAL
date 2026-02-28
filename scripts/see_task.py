#!/usr/bin/env python3
"""
Quick utility to load a LIBERO task and save a screenshot.

Two ways to resolve the task:
1) Suite-based: select a suite + task_id
2) Name-based: select problem_folder + task_name

Optionally, you can load a specific initialization bundle (overrides + state)
to visualize exactly what evaluation will run.

Edit CONFIG below in your editor, or pass minimal CLI overrides.
"""

import argparse
import os
from typing import Any, Dict

import numpy as np
import imageio.v2 as imageio

from libero.libero import get_libero_path, benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
from libero.libero.utils.init_loader import (
    make_env_for_init,
    make_env_for_task_name,
)


# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG: Dict[str, Any] = {
    # Choose task resolution mode: "suite" or "name"
    "mode": "suite",

    # Suite-based
    "suite": "my_suite",
    "task_id": 1,

    # Cameras
    "camera_names": ["frontview"],
    "camera_height": 256,
    "camera_width": 256,
    "rotate_180": True,

    # Output
    "out_dir": "renders/see_task",

    # Optional: load a specific init bundle (overrides + state)
    "use_init_bundle": False,
    "init_idx": 0,
}
# ------------------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def suite_bddl_path(suite, task_id: int) -> str:
    return suite.get_task_bddl_file_path(task_id)


def name_bddl_path(problem_folder: str, task_name: str) -> str:
    return os.path.join(get_libero_path("bddl_files"), problem_folder, f"{task_name}.bddl")


def save_obs_images(obs: Dict[str, Any], out_dir: str, cameras, rotate_180: bool):
    ensure_dir(out_dir)
    for cam in cameras:
        key = f"{cam}_image"
        if key not in obs:
            continue
        img = obs[key]
        if rotate_180:
            img = np.rot90(img, 2).copy()
        imageio.imwrite(os.path.join(out_dir, f"{cam}.png"), img)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default=CONFIG.get("mode", "suite"))
    parser.add_argument("--suite", default=CONFIG.get("suite", None))
    parser.add_argument("--task_id", type=int, default=CONFIG.get("task_id", 0))
    parser.add_argument("--problem_folder", default=CONFIG.get("problem_folder", None))
    parser.add_argument("--task_name", default=CONFIG.get("task_name", None))
    parser.add_argument("--out_dir", default=CONFIG.get("out_dir", "renders/see_task"))
    parser.add_argument("--init_idx", type=int, default=CONFIG.get("init_idx", 0))
    parser.add_argument("--use_init_bundle", type=lambda x: str(x).lower() == "true", default=CONFIG.get("use_init_bundle", False))
    args, _ = parser.parse_known_args()

    # Materialize effective cfg
    cfg = dict(CONFIG)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    base_env_args = {
        "camera_names": cfg["camera_names"],
        "camera_heights": cfg["camera_height"],
        "camera_widths": cfg["camera_width"],
        "camera_depths": False,
        "robots": ["Panda"],
        "controller": "OSC_POSE",
    }

    # Resolve and create env
    mode = cfg["mode"].lower()
    if cfg["use_init_bundle"]:
        if mode == "suite":
            suite_cls = benchmark.get_benchmark_dict()[cfg["suite"]]
            suite = suite_cls()
            env, obs = make_env_for_init(suite, int(cfg["task_id"]), int(cfg["init_idx"]), base_env_args)
            task = suite.get_task(int(cfg["task_id"]))
            out_dir = os.path.join(cfg["out_dir"], cfg["suite"], task.name, f"init_{cfg['init_idx']}")
        else:
            env, obs = make_env_for_task_name(cfg["problem_folder"], cfg["task_name"], int(cfg["init_idx"]), base_env_args)
            out_dir = os.path.join(cfg["out_dir"], cfg["problem_folder"], cfg["task_name"], f"init_{cfg['init_idx']}")
    else:
        # Plain env reset (no specific init state)
        if mode == "suite":
            suite_cls = benchmark.get_benchmark_dict()[cfg["suite"]]
            suite = suite_cls()
            bddl = suite_bddl_path(suite, int(cfg["task_id"]))
            task = suite.get_task(int(cfg["task_id"]))
            env = OffScreenRenderEnv(bddl_file_name=bddl, **base_env_args)
            obs = env.reset()
            out_dir = os.path.join(cfg["out_dir"], cfg["suite"], task.name)
        else:
            bddl = name_bddl_path(cfg["problem_folder"], cfg["task_name"])
            env = OffScreenRenderEnv(bddl_file_name=bddl, **base_env_args)
            obs = env.reset()
            out_dir = os.path.join(cfg["out_dir"], cfg["problem_folder"], cfg["task_name"])

    # Save images
    save_obs_images(obs, out_dir, cfg["camera_names"], cfg["rotate_180"])
    print(f"Saved screenshots to: {out_dir}")
    env.close()


if __name__ == "__main__":
    main()

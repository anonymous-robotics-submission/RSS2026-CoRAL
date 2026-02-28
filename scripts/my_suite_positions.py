"""
Edit this file to define the exact placements you want to use
for each custom task in the `my_suite` benchmark.

Coordinates are in world XY (meters). Z and orientation are kept
from the environment's default reset so the object sits properly
on the floor / table. If you do want to control yaw explicitly,
add a "yaw" (radians) key to an entry; otherwise we preserve the
current orientation.

Example edit:
    TASK_POSITIONS["push_the_blue_box_to_the_front_of_the_wall"] = [
        {"xy": [ 0.00, -0.055]},
        {"xy": [ 0.02, -0.055]},
        {"xy": [-0.02, -0.055], "yaw": 0.0},
    ]

You can re-run the preview and save scripts after modifying this file.
"""

"""
{"xy": [0.000, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 200}}},  # ≈ 0.82 kg
        {"xy": [0.010, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 300}}},  # ≈ 1.23 kg
        {"xy": [-0.010, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 200}}},  # ≈ 0.35 kg
        {"xy": [0.020, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 300}}},  # ≈ 0.52 kg
        {"xy": [-0.020, -0.055], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 200}}},  # ≈ 1.60 kg
        {"xy": [0.000, -0.052], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 300}}},  # ≈ 2.40 kg
        {"xy": [0.000, -0.058], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 400}}},  # ≈ 1.64 kg
        {"xy": [0.015, -0.052], "overrides": {"block_1": {"size": [0.07, 0.07, 0.07], "density": 250}}},  # ≈ 0.69 kg
        {"xy": [-0.015, -0.052], "overrides": {"block_1": {"size": [0.09, 0.09, 0.09], "density": 250}}},  # ≈ 1.46 kg
        {"xy": [0.000, -0.060], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 500}}},  # ≈ 2.05 kg
"""


from typing import Dict, List, TypedDict, Optional


class Placement(TypedDict, total=False):
    xy: List[float]            # [x, y]
    yaw: float                 # optional, radians; if missing, keep current yaw
    overrides: dict            # optional, per-initialization object_overrides


# Map each task name to its list of placements to preview / save
# Notes about global scene setup (kept identical across current tasks):
# - Robot base pose and end-effector start: defined by the domain and robot
#   configuration in `libero/libero/envs/problems/my_floor_manipulation.py`
#   together with `OnTheGroundPanda` defaults (see
#   `libero/libero/envs/robots/on_the_ground_panda.py` for `init_qpos`). These
#   are shared for the currently implemented tasks (push / flip). If you want
#   to change base or EE start, define / update them in
#   `my_floor_manipulation.py`, not here.
#   • Panda init_qpos (for reference):
#     [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, ~0.785398]
#
# - Wall (wall2) specifications used by the push task:
#   • BDDL fixed region center (approx): (-0.14, 0.20) from
#     `libero/libero/bddl_files/my_suite/push_the_blue_box_to_the_front_of_the_wall.bddl`
#     wall_region ranges: (-0.1400001, 0.20, -0.14, 0.200001)
#   • Asset geometry (MuJoCo half-extents): from `libero/libero/assets/wall2.xml`
#       size = [0.45, 0.02, 0.50] (half-lengths)
#       pos  = [0.15, 0.05, 0.08] (local center)
#     Full dimensions: 0.90m (x) × 0.04m (y/thickness) × 1.00m (z/height)
#
# Mass note for comments below:
# • `Block` uses BoxObject with size as half-extents (meters).
# • Mass (kg) = density (kg/m^3) × (2*sx)×(2*sy)×(2*sz).
TASK_POSITIONS: Dict[str, List[Placement]] = {
    # Fill or modify this list with your 10 curated placements
    "push_the_blue_box_to_the_front_of_the_wall": [
        # Initial example positions near your BDDL's box_region
        {"xy": [0.000, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 0.82 kg
        {"xy": [0.010, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 1.23 kg
        {"xy": [-0.010, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 5}}},  # ≈ 0.35 kg
        {"xy": [0.020, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 5}}},  # ≈ 0.52 kg
        {"xy": [-0.020, -0.055], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 5}}},  # ≈ 1.60 kg
        {"xy": [0.000, -0.052], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 5}}},  # ≈ 2.40 kg
        {"xy": [0.000, -0.058], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 1.64 kg
        {"xy": [0.015, -0.052], "overrides": {"block_1": {"size": [0.07, 0.07, 0.07], "density": 5}}},  # ≈ 0.69 kg
        {"xy": [-0.015, -0.052], "overrides": {"block_1": {"size": [0.09, 0.09, 0.09], "density": 5}}},  # ≈ 1.46 kg
        {"xy": [0.000, -0.060], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 2.05 kg
    ],
    "flip_the_blue_box_onto_its_side": [
        # Start by reusing the same placements as the push task
        {"xy": [0.000, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 0.82 kg
        {"xy": [0.010, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 1.23 kg
        {"xy": [0.000, -0.058], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 1.64 kg
        {"xy": [0.000, -0.060], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},  # ≈ 2.05 kg
        {"xy": [-0.010, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 5}}},  # ≈ 0.35 kg
        {"xy": [0.020, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 5}}},  # ≈ 0.52 kg
        {"xy": [0.015, -0.052], "overrides": {"block_1": {"size": [0.07, 0.07, 0.07], "density": 5}}},  # ≈ 0.69 kg
        {"xy": [-0.015, -0.052], "overrides": {"block_1": {"size": [0.09, 0.09, 0.09], "density": 5}}},  # ≈ 1.46 kg
        {"xy": [-0.020, -0.055], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 5}}},  # ≈ 1.60 kg
        {"xy": [0.000, -0.052], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 5}}},  # ≈ 2.40 kg
        
        
    ],
    "push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side": [
        # Reuse the same initial placements as the push-to-wall task
        {"xy": [0.000, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},
        {"xy": [0.010, -0.055], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},
        {"xy": [-0.010, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 5}}},
        {"xy": [0.020, -0.055], "overrides": {"block_1": {"size": [0.06, 0.06, 0.06], "density": 5}}},
        {"xy": [-0.020, -0.055], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 5}}},
        {"xy": [0.000, -0.052], "overrides": {"block_1": {"size": [0.10, 0.10, 0.10], "density": 5}}},
        {"xy": [0.000, -0.058], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},
        {"xy": [0.015, -0.052], "overrides": {"block_1": {"size": [0.07, 0.07, 0.07], "density": 5}}},
        {"xy": [-0.015, -0.052], "overrides": {"block_1": {"size": [0.09, 0.09, 0.09], "density": 5}}},
        {"xy": [0.000, -0.060], "overrides": {"block_1": {"size": [0.08, 0.08, 0.08], "density": 5}}},
    ],
}

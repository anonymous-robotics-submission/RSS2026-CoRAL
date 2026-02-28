"""
Basic primitive objects used in custom tasks.

Block defaults:
- size: half-extents in meters (sx, sy, sz)
- density: kg/m^3 (MuJoCo / SI units)
- friction: (sliding, torsional, rolling) coefficients. Default matches
  common robosuite table defaults [1.0, 0.005, 0.0001]. You can override via
  object_overrides, e.g., {"block_1": {"friction": [1, 0.005, 0.0001]}}.
"""

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string

# BoxObject import path can vary slightly across robosuite versions
try:
    from robosuite.models.objects import BoxObject
except Exception:
    from robosuite.models.objects.primitive import BoxObject  # fallback

from libero.libero.envs.base_object import register_object


@register_object
class Block(BoxObject):
    """
    Solid cube-like object. Registered under category name 'block'.
    """
    def __init__(
        self,
        name,
        size=(0.08, 0.08, 0.08),     # ~10 cm half-extents per axis
        rgba=(0, 0, 1, 1.0),   # set alpha <1.0 if you want translucency
        joints="default",               # movable
        density=20,                # kg/m^3
        friction=(1.0, 0.0, 0.0),  # (sliding, torsional, rolling) #(1.0, 0.005, 0.0001) default
        **kwargs,
    ):
        super().__init__(
            name=name,
            size=np.array(size, dtype=float),
            rgba=np.array(rgba, dtype=float),
            joints=joints,
            density=density,
            **kwargs,
        )
        # Set per-geom friction in the object's XML so MuJoCo uses desired values.
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            # Be permissive if friction cannot be applied at construction time
            pass
        # Attributes LIBERO placement / wrappers expect:
        self.category_name = "block"
        self.rotation = (0.0, 0.0)      # yaw range; change to (0.0, 6.283185307) for random yaw
        self.rotation_axis = "z"

        # Some code accesses this dict & a 'vis_site_names' key
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})

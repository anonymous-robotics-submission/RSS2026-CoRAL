# libero/libero/envs/objects/static_objects.py
import os
import re
import pathlib

import numpy as np
from robosuite.models.objects import MujocoXMLObject

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import register_object


class GenericAssetObject(MujocoXMLObject):
    """
    Load a single XML from libreo/libero/assets/<obj_name>.xml with no articulation.
    Use an empty joints list to keep it fixed to the world by default.
    """
    def __init__(self, name, obj_name, joints=None):
        super().__init__(
            os.path.join(str(absolute_path), f"assets/{obj_name}.xml"),
            name=name,
            joints=joints or [],  # empty => fixed, not free-floating
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}


@register_object
class Wall(GenericAssetObject):
    def __init__(self, name="wall", obj_name="wall", joints=None):
        # keep it fixed: joints=[]
        super().__init__(name=name, obj_name=obj_name, joints=[])
        # If your placer tries to put things "on a table" using z_on_table, keep it zero.
        self.z_on_table = 0.0

@register_object
class Wall2(GenericAssetObject):
    def __init__(self, name="wall2", obj_name="wall2", joints=None):
        # empty joints => fixed / immovable
        super().__init__(name=name, obj_name=obj_name, joints=[])
        # keep placement simple: lie flat on the table plane
        self.z_on_table = 0.0
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"

try:
    from libero.libero.envs.base_object import OBJECTS_DICT as _OBJECTS_DICT
    # only add if missing (no-op on subsequent imports)
    _OBJECTS_DICT.setdefault("wall2", Wall2)
except Exception:
    pass
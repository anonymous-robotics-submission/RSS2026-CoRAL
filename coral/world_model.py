"""Dynamic world model that fuses perception and memory for planning."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .types import PhysicalParameters, PoseEstimate, WorldState

LOGGER = logging.getLogger(__name__)


@dataclass
class WorldModel:
    """Maintains a dynamic representation of the manipulation scene."""

    state: WorldState

    def update_pose(self, pose: PoseEstimate) -> None:
        LOGGER.debug("Updating world model pose for %s", pose.object_id)
        self.state.poses[pose.object_id] = pose
        self.state.last_updated = time.time()
        self.state.history.append({"type": "pose", "object_id": pose.object_id})

    def update_physical_parameters(self, parameters: PhysicalParameters) -> None:
        LOGGER.debug("Updating world model parameters for %s", parameters.object_id)
        self.state.physical_parameters[parameters.object_id] = parameters
        self.state.last_updated = time.time()
        self.state.history.append(
            {"type": "parameters", "object_id": parameters.object_id}
        )

    def get_pose(self, object_id: str) -> Optional[PoseEstimate]:
        return self.state.poses.get(object_id)

    def get_parameters(self, object_id: str) -> Optional[PhysicalParameters]:
        return self.state.physical_parameters.get(object_id)

    def batch_update(
        self, poses: Iterable[PoseEstimate], parameters: Iterable[PhysicalParameters]
    ) -> None:
        for pose in poses:
            self.update_pose(pose)
        for parameter in parameters:
            self.update_physical_parameters(parameter)

    def snapshot(self) -> WorldState:
        LOGGER.debug("Creating a snapshot of the world model")
        return WorldState(
            poses=dict(self.state.poses),
            physical_parameters=dict(self.state.physical_parameters),
            last_updated=self.state.last_updated,
            history=list(self.state.history),
        )

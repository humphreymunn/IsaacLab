# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.utils import configclass

from .rigid_object_collection import RigidObjectCollection


@configclass
class RigidObjectCollectionCfg:
    """Configuration parameters for a rigid object collection."""

    class_type: type = RigidObjectCollection
    """The associated asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """

    rigid_objects: dict[str, RigidObjectCfg] = MISSING
    """Dictionary of rigid object configurations to spawn.

    The keys are the names for the objects, which are used as unique identifiers throughout the code.
    """

    objects_external_wrench_frame: str = "local"
    """Frame in which external wrenches are applied. Defaults to "local".

    If "local", the external wrenches are applied in the local frame of the articulation root.
    If "world", the external wrenches are applied in the world frame.
    """

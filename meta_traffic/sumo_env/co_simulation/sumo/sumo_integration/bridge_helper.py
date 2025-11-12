""" This module provides a helper for the co-simulation between sumo and metadrive."""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import json
import logging
import math
import os
import random
from metadrive.type import MetaDriveType
import metadrive  # pylint: disable=import-error
import traci  # pylint: disable=import-error
from panda3d.core import Vec3
from metadrive.component.vehicle.vehicle_type import random_vehicle_type
from metadrive.manager.scenario_traffic_manager import get_vehicle_type
from .sumo_simulation import SumoSignalState, SumoVehSignal
from metadrive.utils.random_utils import get_np_random
# ==================================================================================================
# -- Bridge helper (SUMO <=> METADRIVE) ----------------------------------------------------------------
# ==================================================================================================


class BridgeHelper(object):
    """
    BridgeHelper provides methos to ease the co-simulation between sumo and metadrive.
    """

    blueprint_library = []
    offset = (0, 0)

    @staticmethod
    def get_metadrive_transform(in_sumo_transform, extent, sumo_actor=None):
        """
        Returns metadrive transform based on sumo transform.
        """
        offset = BridgeHelper.offset
        in_location = in_sumo_transform[0]
        in_rotation = in_sumo_transform[1]

        # From front-center-bumper to center
        # (http://sumo.sourceforge.net/userdoc/Purgatory/Vehicle_Values.html#angle)

        # sumo clockwise  <--> metadrive counterclockwise

        # pitch, yaw, roll
        pitch = in_rotation[0]
        yaw = 360 - in_rotation[1]

        if sumo_actor.vclass.value in ['passenger', 'evehicle', 'truck', 'authority', 'bicycle', 'motorcycle']:
            out_location = (
                in_location.x + math.sin(math.radians(yaw)) * extent.x,
                in_location.y - math.cos(math.radians(yaw)) * extent.x,
                in_location.z - math.sin(math.radians(pitch)) * extent.x
            )
        else:
            out_location = (in_location.x, in_location.y, in_location.z)

        # pitch, yaw, roll
        out_rotation = Vec3(in_rotation[0], yaw, in_rotation[2])

        # Applying offset sumo-metadrive net.
        out_location = Vec3(out_location[0] - offset[0], out_location[1] - offset[1], out_location[2])
        # Transform to metadrive reference system.
        out_transform = (out_location, out_rotation, Vec3(extent.x, extent.y, extent.z))
        return out_transform

    @staticmethod
    def get_sumo_transform(in_metadrive_transform, extent):
        """
        Returns sumo transform based on metadrive transform.
        """
        offset = BridgeHelper.offset
        in_location = in_metadrive_transform[0]
        in_rotation = in_metadrive_transform[1]

        # From center to front-center-bumper
        pitch = in_rotation[0]
        yaw = 360 - in_rotation[1]

        out_location = (
            in_location.x + math.sin(math.radians(yaw)) * extent.x,
            in_location.y - math.cos(math.radians(yaw)) * extent.x,
            in_location.z + math.sin(math.radians(pitch)) * extent.x
        )

        # roll pitch yaw
        out_rotation = Vec3(in_rotation[0], yaw, in_rotation[2])

        # Applying offset metadrive-sumo net
        out_location = Vec3(out_location[0] + offset[0], out_location[1] + offset[1], out_location[2])

        # Transform to sumo reference system.
        out_transform = (out_location, out_rotation, Vec3(extent.x, extent.y, extent.z))

        return out_transform

    @staticmethod
    def get_metadrive_blueprint(sumo_actor, sync_color=False):
        """
        Returns an appropriate blueprint based on the received sumo actor.
        """
        type_id = sumo_actor.type_id
        actor_type = None
        if sumo_actor.vclass.value in ['passenger', 'evehicle', 'truck', 'authority']:
            actor_type = get_vehicle_type(sumo_actor.extent.x * 2, get_np_random())
        elif sumo_actor.vclass.value in ['bicycle', 'motorcycle']:
            from metadrive.component.traffic_participants.cyclist import Cyclist
            actor_type = Cyclist
        elif sumo_actor.vclass.value in ['pedestrian']:
            from metadrive.component.traffic_participants.pedestrian import Pedestrian
            actor_type = Pedestrian
        else:
            print(actor_type, sumo_actor.vclass.value)
            actor_type = get_vehicle_type(sumo_actor.extent.x * 2, get_np_random())
        
        return actor_type

    @staticmethod
    def get_metadrive_traffic_light_state(sumo_tl_state):
        """
        Returns metadrive traffic light state based on sumo traffic light state.
        """
        if sumo_tl_state == SumoSignalState.RED or sumo_tl_state == SumoSignalState.RED_YELLOW:
            return MetaDriveType.LIGHT_RED

        elif sumo_tl_state == SumoSignalState.YELLOW:
            return MetaDriveType.LIGHT_YELLOW

        elif sumo_tl_state == SumoSignalState.GREEN or \
             sumo_tl_state == SumoSignalState.GREEN_WITHOUT_PRIORITY:
            return MetaDriveType.LIGHT_GREEN

        else:  # SumoSignalState.GREEN_RIGHT_TURN and SumoSignalState.OFF_BLINKING
            return MetaDriveType.LIGHT_UNKNOWN

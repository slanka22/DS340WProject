import copy
import logging
from collections import namedtuple
from typing import Dict

import math
import numpy as np
from copy import deepcopy


from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian

class TrafficManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(TrafficManager, self).__init__()
        # self.spawned_objects = dict()
        self._traffic_vehicles = []

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        pass

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            if p:
                v.before_step(p.act())
        return dict()

    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            # v.after_step()
            if not v.on_lane:
                v_to_remove.append(v)
              
        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)

        return dict()

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(TrafficManager, self).before_reset()
        self._traffic_vehicles = []

    def get_global_states(self) -> Dict:
        """
        Return all traffic vehicles' states
        :return: States of all vehicles
        """
        states = dict()
        traffic_states = dict()
        for vehicle in self._traffic_vehicles:
            traffic_states[vehicle.index] = vehicle.get_state()

        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v.get_state()
                                      for k, v in pending_obj.items()}, allow_new_keys=True
        )
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v_count[0].get_state()
                                      for k, v_count in dying_obj.items()},
            allow_new_keys=True
        )

        states[OBJECT_TO_AGENT] = copy.deepcopy(self.engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(self.engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        Special handling for first states of traffic vehicles
        :return: States of all vehicles
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        return vehicles

    def remove_object(self, actor):
        self.clear_objects([actor.id])
        if actor not in self._traffic_vehicles:
            return
        self._traffic_vehicles.remove(actor)

    def add_object(self, actor_type, metadrive_transform):

        location, rotation, extent = metadrive_transform[0], metadrive_transform[1], metadrive_transform[2]
        # pitch, yaw, roll
        if actor_type == Cyclist or actor_type == Pedestrian:
            random_v = self.spawn_object(
                actor_type,
                position=(location[0], location[1]),
                heading_theta=np.deg2rad(rotation[1] + 90),
            )
        else:
            v_config = copy.deepcopy(self.engine.global_config["vehicle_config"])
            v_config.update(
                dict(
                    show_navi_mark=False,
                    show_dest_mark=False,
                    enable_reverse=False,
                    show_lidar=False,
                    show_lane_line_detector=False,
                    show_side_detector=False,
                )
            )
            random_v = self.spawn_object(
                actor_type,
                position=(location[0], location[1]),
                heading=np.deg2rad(rotation[1]),
                vehicle_config=v_config
            )
            # from metadrive.policy.idm_policy import IDMPolicy
            # self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())

        self._traffic_vehicles.append(random_v)

        return random_v.id

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self.spawned_objects = dict()
        self._traffic_vehicles = []

        # traffic vehicle list
        self._traffic_vehicles = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self.vehicles.__repr__()

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        super(TrafficManager, self).seed(random_seed)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    def get_state(self):
        ret = super(TrafficManager, self).get_state()
        ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
        return ret

    def set_state(self, state: dict, old_name_to_current=None):
        super(TrafficManager, self).set_state(state, old_name_to_current)
        self._traffic_vehicles = list(
            self.get_objects([old_name_to_current[name] for name in state["_traffic_vehicles"]]).values()
        )

# For compatibility check
TrafficManager = TrafficManager
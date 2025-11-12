import copy
import time
import os
import sys

import numpy as np
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.manager.base_manager import BaseManager

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from metadrive.component.vehicle.vehicle_type import SVehicle, MVehicle, LVehicle, XLVehicle, DefaultVehicle
from meta_traffic.metatraffic_env.sumo_integration.bridge_helper import BridgeHelper
from meta_traffic.metatraffic_env.sumo_integration.sumo_simulation import INVALID_ACTOR_ID

actor_class_height_mapping = {SVehicle: 0.5364,
                              DefaultVehicle: 0.5371,
                              XLVehicle: 0.6354,
                              MVehicle: 0.5385,
                              LVehicle: 0.5378}

class SUMOManager(BaseManager):
    PRIORITY = 10  # make sure it has the lowest priority

    def __init__(self):
        super(SUMOManager, self).__init__()

        self.sync_vehicle_color = self.engine.global_config["sync_vehicle_color"]
        self.sync_vehicle_lights = self.engine.global_config["sync_vehicle_lights"]

        # # Set traffic lights.
        self._tls = {}

        # Mapped actor ids.
        self.sumo2metadrive_ids = {}  # Contains only actors controlled by sumo.
        self.metadrive2sumo_ids = {}  # Contains only actors controlled by metadrive.

        self.sumo2metadrive_ped_ids = {}  # Contains only ped actors controlled by sumo.
        self.metadrive2sumo_ped_ids = {}  # Contains only ped actors controlled by metadrive.

        self.vehicles_throughput = None
        self.inside_vehicles = None
        # debug
        self.sumo_total_time = 0  # unit ms

    @property
    def sumo(self):
        return self.engine.agent_manager.sumo

    def before_reset(self):
        """Clear all objects"""
        super(SUMOManager, self).before_reset()
        self.sumo2metadrive_ids = {}  # Contains only actors controlled by sumo.
        self.metadrive2sumo_ids = {}  # Contains only actors controlled by metadrive.

        self.sumo2metadrive_ped_ids = {}  # Contains only ped actors controlled by sumo.
        self.metadrive2sumo_ped_ids = {}  # Contains only ped actors controlled by metadrive.

        self.vehicles_throughput = {}
        self.inside_vehicles = {}

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.engine.agent_manager.sim_step

    def reset(self):
        super(SUMOManager, self).reset()
        # debug
        self.sumo_total_time = 0
        BridgeHelper.blueprint_library = []
        BridgeHelper.offset = self.sumo.offset
        entering_v = self.engine.agent_manager.conn.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.sim_step})

    def before_step(self, *args, **kwargs):
        # The simulation performance is bounded by SUMO.
        # In a casual test setting, the tick() costs 0.006s, while the whole after_step() takes 0.008s.
        # Thus remaining code makes the FPS drop from 1/0.006=166 FPS to 1/0.008 = 125FPS.
        start = time.time()
        time_to_act = False
        while not time_to_act:
            self.sumo.tick()
            self._synchronize_all()

            for ts in self.engine.agent_manager.traffic_signal_ids:
                self.engine.agent_manager.traffic_signals[ts].update()
                if self.engine.agent_manager.traffic_signals[ts].time_to_act:
                    time_to_act = True
        self.sumo_total_time += time.time() - start
        '''
        For vehicle actuation
        '''

    def _synchronize_all(self):
        # Spawning new sumo actors in meta (i.e, not controlled by metadrive).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.metadrive2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            metadrive_blueprint = BridgeHelper.get_metadrive_blueprint(sumo_actor, self.sync_vehicle_color)
            if metadrive_blueprint is not None:
                metadrive_transform = BridgeHelper.get_metadrive_transform(
                    sumo_actor.transform, sumo_actor.extent, sumo_actor
                )

                metadrive_actor_id = self.add_object(metadrive_blueprint, metadrive_transform)
                if metadrive_actor_id != INVALID_ACTOR_ID:
                    self.sumo2metadrive_ids[sumo_actor_id] = metadrive_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in meta.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2metadrive_ids:
                id = self.sumo2metadrive_ids.pop(sumo_actor_id)
                self.clear_objects([id])
                self.engine._clean_color(id)

        # Updating sumo actors in metadrive.
        for sumo_actor_id in self.sumo2metadrive_ids:
            metadrive_actor_id = self.sumo2metadrive_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            # metadrive_actor = self.metadrive.get_actor(metadrive_actor_id)

            metadrive_transform = BridgeHelper.get_metadrive_transform(
                sumo_actor.transform, sumo_actor.extent, sumo_actor
            )
            # if self.sync_vehicle_lights:
            #     metadrive_lights = BridgeHelper.get_metadrive_lights_state(metadrive_actor.get_light_state(),
            #                                                        sumo_actor.signals)
            # else:
            metadrive_lights = None

            self.synchronize_actor(metadrive_actor_id, metadrive_transform, sumo_actor, metadrive_lights)

        '''
        For Pedestrian actuation
        '''
        # Spawning new sumo actors in meta (i.e, not controlled by metadrive).
        sumo_spawned_ped_actors = self.sumo.spawned_ped_actors - set(self.metadrive2sumo_ped_ids.values())
        for sumo_actor_id in sumo_spawned_ped_actors:
            self.sumo.subscribe_pedestrian(sumo_actor_id)
            sumo_actor = self.sumo.get_actor_pedestrian(sumo_actor_id)

            metadrive_blueprint = BridgeHelper.get_metadrive_blueprint(sumo_actor, self.sync_vehicle_color)
            if metadrive_blueprint is not None:
                metadrive_transform = BridgeHelper.get_metadrive_transform(
                    sumo_actor.transform, sumo_actor.extent, sumo_actor
                )

                metadrive_actor_id = self.add_object(metadrive_blueprint, metadrive_transform)
                if metadrive_actor_id != INVALID_ACTOR_ID:
                    self.sumo2metadrive_ped_ids[sumo_actor_id] = metadrive_actor_id
            else:
                self.sumo.unsubscribe_pedestrian(sumo_actor_id)

        # Destroying sumo arrived actors in meta.
        for sumo_actor_id in self.sumo.destroyed_ped_actors:
            if sumo_actor_id in self.sumo2metadrive_ped_ids:
                self.clear_objects([self.sumo2metadrive_ped_ids.pop(sumo_actor_id)])

        # Updating sumo actors in metadrive.
        for sumo_actor_id in self.sumo2metadrive_ped_ids:
            metadrive_actor_id = self.sumo2metadrive_ped_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor_pedestrian(sumo_actor_id)
            # metadrive_actor = self.metadrive.get_actor(metadrive_actor_id)

            metadrive_transform = BridgeHelper.get_metadrive_transform(
                sumo_actor.transform, sumo_actor.extent, sumo_actor
            )
            metadrive_lights = None

            self.synchronize_actor(metadrive_actor_id, metadrive_transform, sumo_actor, metadrive_lights)

        entering_v = self.engine.agent_manager.conn.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.sim_step})
        exiting_v = self.engine.agent_manager.conn.simulation.getArrivedIDList()
        for v in exiting_v:
            self.vehicles_throughput.update({v: self.sim_step - self.inside_vehicles[v]})

    def destroy(self):
        super(SUMOManager, self).destroy()
        for metadrive_actor_id in self.sumo2metadrive_ids.values():
            try:
                self.clear_objects([metadrive_actor_id])
            except:
                pass

        for sumo_actor_id in self.metadrive2sumo_ids.values():
            try:
                self.sumo.destroy_actor(sumo_actor_id)
            except:
                pass

        self.sumo2metadrive_ids = {}
        self.metadrive2sumo_ids = {}

        self.sumo2metadrive_ped_ids = {}
        self.metadrive2sumo_ped_ids = {}

        self.vehicles_throughput = None
        self.inside_vehicles = None

    def get_actor(self, actor_id):
        """
        Accessor for metadrive object.
        """
        return self.spawned_objects[actor_id]

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

        return random_v.id

    def get_actor_light_state(self, actor_id):
        """
        Accessor for metadrive actor light state.

        If the actor is not alive, returns None.
        """
        try:
            actor = self.get_actor(actor_id)
            return actor.get_state()
        except RuntimeError:
            return None

    def synchronize_actor(self, actor_id, transform, sumo_actor, lights=None):
        """
        Updates actor state.

            :param actor_id: id of the actor to be updated.
            :param transform: new actor transform (i.e., position and rotation).
            :param lights: new actor light state.
            :return: True if successfully updated. Otherwise, False.
        """

        actor = self.get_actor(actor_id)
        if actor is None:
            return False

        position, rotation = transform[0], transform[1]

        # pitch, yaw, roll
        actor.set_pitch(rotation.x)
        actor.set_roll(rotation.z)

        if sumo_actor.vclass.value in ['bicycle', 'motorcycle', 'pedestrian']:

            actor.set_position((position.x, position.y), height=actor.HEIGHT / 2)
            # actor.set_velocity(
            #     [
            #         -sumo_actor.speed * math.sin(math.radians(rotation.y + 90)),
            #         sumo_actor.speed * math.cos(math.radians(rotation.y + 90))
            #     ]
            # )

            actor.set_heading_theta(rotation.y + 90, in_rad=False)
        else:
            # actor.set_position(
            #     (position.x, position.y, actor.height / 2.0 + actor.TIRE_RADIUS - actor.CHASSIS_TO_WHEEL_AXIS))
            actor.set_position((position.x, position.y), actor_class_height_mapping[actor.__class__])
            # actor.set_velocity(
            #     [
            #         -sumo_actor.speed * math.sin(math.radians(rotation.y)),
            #         sumo_actor.speed * math.cos(math.radians(rotation.y))
            #     ]
            # )
            actor.set_heading_theta(rotation.y, in_rad=False)

        # if lights is not None:
        #     actor.set_light_state(carla.actorLightState(lights))
        return True

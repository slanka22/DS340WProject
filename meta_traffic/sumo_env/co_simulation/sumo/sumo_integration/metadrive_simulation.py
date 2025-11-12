""" This module is responsible for the management of the metadrive simulation. """

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import logging
import math
import cv2
import metadrive
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.base_env import BaseEnv
from metadrive.type import MetaDriveType
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from panda3d.core import Vec3
from .traffic_manager import TrafficManager
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
import gymnasium as gym
import numpy as np

# ==================================================================================================
# -- metedrive simulation ------------------------------------------------------------------------------
# ==================================================================================================
class CustomScenarioEnv(ScenarioEnv):
    def __init__(self, config=None):
        super(CustomScenarioEnv, self).__init__(config)

    def setup_engine(self):
        super(CustomScenarioEnv, self).setup_engine()
        self.engine.register_manager("traffic_manager", TrafficManager())


class MyObservation(BaseObservation):
    def __init__(self, config):
        super(MyObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, config['vehicle_config']['image_source'], config["norm_pixel"])
        self.config  = config
        self.center_p = None

    @property
    def observation_space(self):
        os = {}
        # os={"entry_{}".format(idx): self.img_obs.observation_space for idx in range(4)}
        os["image"] = self.img_obs.observation_space
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        assert self.sumo is not None
        ret = {}
        if self.center_p is None:
            for node in self.sumo.net.getNodes():
                traffic_id = node.getID()
                if "traffic_light" in node.getType():
                    center_p = node._coord
                    self.center_p = center_p
                    self.traffic_id = traffic_id
                # for idx in range(4):
                #     ret["entry_{}".format(idx)]= self.img_obs.observe(self.engine.origin, position=[70, 8.75, 8], hpr=[idx*90, -15, 0], refresh=True)
                # ret["image".format(idx)]= self.img_obs.observe(self.engine.origin, position=[70, 8.75, 50], hpr=[0, -89.99, 0], refresh=True)
                # for idx in range(4):
                #     ret["entry_{}".format(idx)]= self.img_obs.observe(self.engine.origin, position=[vehicle.position[0], vehicle.position[1], 5], hpr=[idx*90, -15, 0])
                # self.engine.get_sensor(self.config['vehicle_config']['image_source']).cam.reparentTo(self.engine.origin)
        ret["image_{}".format(self.traffic_id)] = self.img_obs.observe(self.engine.origin, position=[self.center_p[0], self.center_p[1], 100], hpr=[0, -90, 0])
        # ret["image_{}".format(self.traffic_id)] = self.img_obs.observe()
            # vehicle.set_position((vehicle.position[0], vehicle.position[1]), height=vehicle.get_z()-1)

        self.ret = ret
        return ret

class MetaDriveSimulation(object):
    """
    MetaDriveSimulation is responsible for the management of the carla simulation.
    """
    def __init__(self, args, decision_repeat=5, image_on_cuda=False, rendering=False):
        self.args = args
        self.world = CustomScenarioEnv(
            config={
                # "manual_control": True,
                "use_render": rendering,
                "stack_size": 1,
                "use_mesh_terrain": True,
                "image_on_cuda": image_on_cuda,
                # TODO I can help connect the SUMO map with internal APIs, so we can avoid the convert step
                "data_directory": args.map,
                "num_scenarios": 1,
                "no_traffic": True,
                "no_light": True,
                # "debug": True,
                "start_scenario_index": 0,
                # "horizon": 1000,
                # "show_coordinates": True,
                "show_crosswalk": True,
                # "render_pipeline": True,
                # "disable_model_compression": True,
                "show_sidewalk": True,
                # "num_workers": 1,
                "map_region_size": 2048,
                "decision_repeat": decision_repeat,
                "vehicle_config": {
                    # "enable_reverse": True,
                    # "show_dest_mark": True,
                    # "image_source": "rgb_camera",
                    "image_source": "main_camera",
                },
                "window_size": (224, 224),
                "agent_observation": MyObservation,
                "image_observation": True,
                # "sensors": {
                #     "rgb_camera": (RGBCamera, 128, 128)
                # },
                "show_interface": False,
                "show_logo": False,
                "show_fps": False,
                "physics_world_step_size": 2e-2
            }
        )

        # self.world.reset(seed=0)
        # self.world.vehicle.origin.hide()
        # self.world.vehicle.disable_gravity()
        # self.world.vehicle.set_pitch(-90.0)
        # self.world.vehicle.set_roll(-90.0)
        # # self.world.vehicle.position
        # # self.world.vehicle.set_position((11800,13330), height=5)
        # self.world.engine.force_fps.disable()
        # center_p = self.world.engine.current_map.get_center_point()
        # self.world.vehicle.set_position((center_p[0], center_p[1]), height=100)

        # The following sets contain updated information for the current frame.
        self._active_actors = set()
        self.spawned_actors = set()
        self.destroyed_actors = set()

        self.image = None

        # # Set traffic lights.
        # light_filter = lambda actor: True if actor.metadrive_type == MetaDriveType.TRAFFIC_LIGHT else False
        self._tls = {}

    def get_actor(self, actor_id):
        """
        Accessor for metadrive object.
        """
        return self.world.engine.traffic_manager.get_objects([actor_id])[actor_id]

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

    @property
    def traffic_light_ids(self):
        return set([key for key, value in self._tls.items()])

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        traffic_light = None
        for key, value in self._tls.items():
            if landmark_id == key:
                traffic_light = value
                break

        if traffic_light == None:
            logging.warning('Landmark %s not found in metadrive', landmark_id)
            return False

        return traffic_light.get_state()

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        for actor in self._tls.values():
            actor.set_green()

    def spawn_actor(self, blueprint, transform):
        return self.world.engine.traffic_manager.add_object(blueprint, transform)

    def destroy_actor(self, actor_id):
        """
        Destroys the given actor.
        """
        actor = self.get_actor(actor_id)
        if actor is not None:
            self.world.engine.traffic_manager.remove_object(actor)
            return True
        return False

    def synchronize_vehicle(self, vehicle_id, transform, sumo_actor, lights=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param lights: new vehicle light state.
            :return: True if successfully updated. Otherwise, False.
        """

        vehicle = self.get_actor(vehicle_id)
        if vehicle is None:
            return False

        position, rotation = transform[0], transform[1]

        # pitch, yaw, roll
        vehicle.set_pitch(rotation.x)
        vehicle.set_roll(rotation.z)

        if sumo_actor.vclass.value in ['bicycle', 'motorcycle', 'pedestrian']:

            vehicle.set_position((position.x, position.y))
            # vehicle.set_velocity(
            #     [
            #         -sumo_actor.speed * math.sin(math.radians(rotation.y + 90)),
            #         sumo_actor.speed * math.cos(math.radians(rotation.y + 90))
            #     ]
            # )

            vehicle.set_heading_theta(rotation.y + 90, in_rad=False)
        else:
            # vehicle.set_position(
            #     (position.x, position.y, vehicle.height / 2.0 + vehicle.TIRE_RADIUS - vehicle.CHASSIS_TO_WHEEL_AXIS))
            vehicle.set_position((position.x, position.y))  # auto-determine
            # vehicle.set_velocity(
            #     [
            #         -sumo_actor.speed * math.sin(math.radians(rotation.y)),
            #         sumo_actor.speed * math.cos(math.radians(rotation.y))
            #     ]
            # )
            vehicle.set_heading_theta(rotation.y, in_rad=False)

        # if lights is not None:
        #     vehicle.set_light_state(carla.VehicleLightState(lights))
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param landmark_id: id of the landmark to be updated.
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        traffic_light = None
        for key, value in self._tls.items():
            if landmark_id == key:
                traffic_light = value
                break

        if traffic_light == None:
            logging.warning('Landmark %s not found in metadrive', landmark_id)
            return False

        if state == MetaDriveType.LIGHT_GREEN:
            traffic_light.set_green()
        elif state == MetaDriveType.LIGHT_RED:
            traffic_light.set_red()
        elif state == MetaDriveType.LIGHT_YELLOW:
            traffic_light.set_yellow()
        elif state == MetaDriveType.LIGHT_UNKNOWN:
            traffic_light.set_unknown()

        return True

    def tick(self):
        """
        Tick to metadrive simulation.
        """
        o, r, tm, tc, info = self.world.step([0, 0])

        # self.image = o["image"]
        # print(self.world.observations.keys())
        # print(o["image"].shape)
        # cv2.imshow('img', o["image"][..., -1])
        # cv2.waitKey(1)
        # # Update data structures for the current frame.
        current_actors = set([vehicle.id for vehicle in self.world.engine.traffic_manager.vehicles])
        self.spawned_actors = current_actors.difference(self._active_actors)
        self.destroyed_actors = self._active_actors.difference(current_actors)
        self._active_actors = current_actors

    def render(self):
        self.world.render(
            # mode="top_down",
            # scaling=1.2,
            # target_vehicle_heading_up=False,
            # film_size=(3000, 3000),
            # semantic_map=True,
            # screen_size=(6000, 6000),
            # text=dict(position=self.world.vehicle.position)
        )

    def close(self):
        """
        Closes metadrive client.
        """
        self.world.close()


if __name__ == "__main__":
    a = MetaDriveSimulation()

    for s in range(0, 100000):

        a.tick()

        object_filter = lambda actor: True if isinstance(actor, BaseVehicle) else False
        objects = a.world.engine.get_objects(object_filter)
        keys = list(objects.keys())[1:]

        agent = a.world.vehicle
        state = agent.get_state()

        transform = (
            Vec3(state['position'][0] + 1, state['position'][1],
                 state['position'][2]), Vec3(state['roll'], state['pitch'], state['heading_theta'])
        )
        a.synchronize_vehicle(agent.id, transform)

        if s % 100 == 0:
            a.world.engine.traffic_manager.add_objects()

        # if s % 20 == 0:
        #     a.synchronize_traffic_light(None, MetaDriveType.LIGHT_GREEN)
        # if s % 40 == 0:
        #     a.synchronize_traffic_light(None, MetaDriveType.LIGHT_GREEN)

        # a.world.render(position=a.world.vehicle.position, track_target_vehicle=True, mode="top_down", semantic_map=True)

        a.world.render(
            # mode="top_down",
            # scaling=1.2,
            target_vehicle_heading_up=False,
            # film_size=(3000, 3000),
            # semantic_map=True,
            # screen_size=(6000, 6000),
            text=dict(position=a.world.vehicle.position)
        )



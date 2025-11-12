import gymnasium as gym
import numpy as np
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.observation_base import BaseObservation
from statistics import mean 
import traci
import sumolib
import math

from abc import abstractmethod
from gymnasium import spaces
from typing import List

from .traffic_signal import TrafficSignal
import cv2
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

NOISE = True

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()

        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
    
def centeroid(data):
        x, y = zip(*data)
        l = len(x)
        return sum(x) / l, sum(y) / l

class TopDownObservation(BaseObservation):
    def __init__(self, config):
        super(TopDownObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, config['image_source'], config["norm_pixel"])
        self.config = config
        self.center_p = None
        self.ts = None  # TODO, modify for multi-agent setting

    def set_traffic_signal(self, traffic_signal):
        self.ts = traffic_signal

    @property
    def observation_space(self):
        os = {}
        numeric_obs_space = gym.spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
        os["observation"] = numeric_obs_space
        os["image"] = self.img_obs.observation_space
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        
        # img = self.engine.agent_manager.conn.gui.screenshot(traci.gui.DEFAULT_VIEW,
        #                                 f"temp/img_1.jpg",
        #                                 width=1024,
        #                                 height=1024)
        
        # import cv2
        # img = cv2.imread('temp/img_1.jpg')
        # if img is not None:
            
        #     cv2.imshow('image_x', img)
        #     cv2.waitKey(1)
            

        assert self.engine.sumo_manager.sumo
        ret = {}
        if self.center_p is None:
            for node in self.engine.sumo_manager.sumo.net.getNodes():
                if "traffic_light" in node.getType():
                    # TODO: get the center from the light
                    center_p = node._coord
                    self.center_p = center_p
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        ret["image"] = self.img_obs.observe()
        ret["observation"] = np.array(phase_id + min_green + density + queue, dtype=np.float32)


        # import cv2
        # cv2.imshow('image', (ret['image'][...,-1]*255).astype(np.uint8))
        # cv2.waitKey(1)

        self.ret = ret


        # print(self.ts.env.sim_step, 'hello')

        return self.ret 


class MultiAngleObservation(BaseObservation):
    """
    It captures the images of 4 ways
    """

    def __init__(self, config):
        super(MultiAngleObservation, self).__init__(config)
        self.pitch = config['pitch']
        self.vision_feature = config['vision_feature']
        self.sumo_feature = config['sumo_feature']
        self.angles = [(angle, self.pitch, 0) for angle in range(0, 359, 90)]
        assert len(self.angles) == 4, 'four angles required'
        self.all_obs = [ImageObservation(config, config['image_source'], config["norm_pixel"]) for _ in self.angles]
        
        self._count = 0
        self.config = config
        self.center_p = None
        self.ts = None  # TODO, modify for multi-agent setting
        self.error = None

    def set_traffic_signal(self, traffic_signal):
        self.ts = traffic_signal

    @property
    def observation_space(self):
        os = {}

        if self.sumo_feature:
            numeric_obs_space = gym.spaces.Box(
                low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
                high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            )
            os["observation"] = numeric_obs_space
        # else:
        #     numeric_obs_space = gym.spaces.Box(
        #         low=np.zeros(self.ts.num_green_phases + 1, dtype=np.float32),
        #         high=np.ones(self.ts.num_green_phases + 1, dtype=np.float32),
        #     )
        #     os["observation"] = numeric_obs_space

        if self.vision_feature:
            for i, _ in enumerate(self.angles):
                os["image_{}".format(i)] = self.all_obs[i].observation_space
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        
        ret = {}
                                       
        # we assumed the phase id and min_green encoding are provided to any system
        if self.sumo_feature:
            phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
            min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
            density = self.ts.get_lanes_density()
            queue = self.ts.get_lanes_queue()


            if NOISE:
                
                # if self.error is None:

                means =  np.array(queue, dtype=np.float32)*.7
                std = np.array(queue, dtype=np.float32)*.3/4
                error = np.random.normal(loc=means, scale=std)
                # self.error = error

                queue = [max(0, min(1, e)) for q, e in zip(queue, list(error))]
       

                means =  np.array(queue, dtype=np.float32)*.7
                std = np.array(queue, dtype=np.float32)*.3/4
                error = np.random.normal(loc=means, scale=std)
                # self.error2 = error
                
                density = [max(0, min(1, e)) for d, e in zip(density, list(error))]

            observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)

            ret["observation"] = observation
        # else:
        #     phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        #     min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        #     observation = np.array(phase_id + min_green, dtype=np.float32)

        #     ret["observation"] = observation

        assert self.engine.sumo_manager.sumo

        if self.center_p is None:
            # TODO: get the center from the light
            for node in self.engine.sumo_manager.sumo.net.getNodes():
                traffic_id = node.getID()
                if "traffic_light" in node.getType():
                    center_p = node._coord
                    self.center_p = center_p
                
                    
                    sensor_locs = []
                    orientations = []
                    offset = self.engine.agent_manager.sumo.offset
        
                    for lane in self.ts.lanes:
                        lane_shape =  traci.lane.getShape(lane)
                        lane_shape = [[x[0] - offset[0], x[1]- offset[1]] for x in lane_shape]
                        start_loc = lane_shape[-1]
                        far_loc = lane_shape[0]

                        # stop lane to the intersection point
                        sensor_loc = sumolib.geomhelper.positionAtOffset(start_loc, far_loc, -5)
                        sensor_locs.append(sensor_loc)
                        orientations.append(180-traci.lane.getAngle(lane))
              
                    assert len(orientations) % 4 == 0, 'assume four directions'
                    lane_num = len(orientations)//4
                    self.angles = [mean(orientations[i:i+lane_num])  for i in range(0, len(orientations), lane_num)]
                    assert len(self.angles) == 4, 'four angles required'

                    self.locations = [centeroid(sensor_locs[i:i+lane_num])  for i in range(0, len(sensor_locs), lane_num)]
                    self.angles = [(angle, self.pitch, 0) for angle in self.angles]
                    self.all_obs = [ImageObservation(self.config, self.config['image_source'], self.config["norm_pixel"]) for _ in self.angles]

        if self.vision_feature:

            if self.config["capture_all_at_once"]:
                # self.all_obs[0].observe()
                for i, hpr in enumerate(self.angles):
                    self.all_obs[i].observe(self.engine.origin,
                                            [*self.locations[i][:2], self.config["top_down_camera_initial_z"]],
                                            hpr)
            else:
                ret["image"] = self.all_obs[self._count % len(self.angles)].observe()
                self._count += 1
                self.engine.main_camera.set_bird_view_pos_hpr(self.center_p[:2],
                                                            self.angles[self._count % len(self.angles)])

            for i, obs in enumerate(self.all_obs):
                ret["image_{}".format(i)] = obs.state 
    
            # import cv2
            # for i, obs in enumerate(self.all_obs):
            #     cv2.imshow('direction_{}'.format(i), (ret['image_{}'.format(i)][...,-1]*255).astype(np.uint8))
            #     cv2.waitKey(1)

        return ret

    def reset(self, env, vehicle=None):
        self._count = 0

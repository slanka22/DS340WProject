"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
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

        if NOISE:
                print('xx')
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
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
    
class ImageObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> Dict[str, np.ndarray]:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()

        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
             
        # image = self.ts.env.metadrive_simulation.world.observations['default_agent'].ret['image_{}'.format(self.ts.id)]
        image = self.ts.env.metadrive_simulation.world.observations['default_agent'].ret['image_{}'.format(self.ts.id)]

        if self.ts.env.metadrive_simulation.world.config["image_on_cuda"]:
            image = image.get()

        # print('image_{}'.format(self.ts.id))
        # print(np.max(image[...,-1]))
        cv2.imshow('image_s{}'.format(self.ts.id), (image[...,-1]*255).astype(np.uint8))
        cv2.waitKey(1)
        # cv2.imshow('image_{}_2'.format(self.ts.id), (image[...,-2]*255).astype(np.uint8))
        # cv2.waitKey(1)
        # cv2.imshow('image_{}_3'.format(self.ts.id), (image[...,-3]*255).astype(np.uint8))
        # cv2.waitKey(1)
     
        # cv2.imwrite(f'images/image_{self.ts.env.sim_step}.jpg', (image[...,-1]*255).astype(np.uint8)) 


        return OrderedDict(
            [
                ('image', image), 
                ('obs', observation),   
            ]
        )

    def observation_space(self) -> spaces.Dict:
        """Return the observation space."""

        image_obs_space = self.ts.env.metadrive_simulation.world.observations['default_agent'].observation_space["image"]

        numeric_obs_space = spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

        return spaces.Dict({'image': image_obs_space, 
                            'obs': numeric_obs_space})



class ImageOnlyObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> Dict[str, np.ndarray]:
        """Return the default observation."""
        image = self.ts.env.metadrive_simulation.world.observations['default_agent'].ret['image_{}'.format(self.ts.id)]

        if self.ts.env.metadrive_simulation.world.config["image_on_cuda"]:
            image = image.get()

        # print('image_{}'.format(self.ts.id))
        # print(np.max(image[...,-1]))
        cv2.imshow('image_{}'.format(self.ts.id), (image[...,-1]*255).astype(np.uint8))
        cv2.waitKey(1)

        return OrderedDict(
            [
                ('image', image), 
                # ('obs', observation),   
            ]
        )

    def observation_space(self) -> spaces.Dict:
        """Return the observation space."""

        image_obs_space = self.ts.env.metadrive_simulation.world.observations['default_agent'].observation_space["image"]

        # numeric_obs_space = spaces.Box(
        #     low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        #     high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        # )

        return spaces.Dict({'image': image_obs_space})
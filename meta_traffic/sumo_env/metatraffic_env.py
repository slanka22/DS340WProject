import os 
import sys
import subprocess
import glob
import time
import json
import re
import random

from os import path, environ
import psutil
import tempfile
import argparse
import logging
import time
from panda3d.core import Vec3

# ==================================================================================================
# -- find metadrive module -----------------------------------------------------------------------------
# ==================================================================================================
from .co_simulation.sumo.sumo_integration.constants import PANDA_WORLD_SIZE
import glob
import os
import sys

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from .co_simulation.sumo.sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from .co_simulation.sumo.sumo_integration.metadrive_simulation import MetaDriveSimulation  # pylint: disable=wrong-import-position
from .co_simulation.sumo.sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from .co_simulation.sumo.sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position
from .co_simulation.sumo.run_synchronization import SimulationSynchronization  # pylint: disable=wrong-import-position

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import sumolib
import traci
import pandas as pd
from .observations import DefaultObservationFunction, ObservationFunction, ImageObservationFunction
from .traffic_signal import TrafficSignal

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class MetadriveSumoGym(gym.Env):
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(self, args,
                 out_csv_name: Optional[str] = None,
                 observation_class: ObservationFunction = ImageObservationFunction,
                 reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
                 begin_time: int = 0,
                 num_seconds: int = 1000,
                 max_depart_delay: int = -1,
                 waiting_time_memory: int = 1000,
                 time_to_teleport: int = -1,
                 delta_time: int = 5,
                 yellow_time: int = 2,
                 min_green: int = 5,
                 max_green: int = 50,
                 single_agent: bool = True,
                 add_system_info: bool = True,
                 add_per_agent_info: bool = True,
                 sumo_seed: Union[str, int] = "random",
                 fixed_ts: bool = False,
                 sumo_warnings: bool = True,
                 additional_sumo_cmd: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 image_on_cuda = True):
        super(MetadriveSumoGym, self).__init__()

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.render_mode = render_mode
        self.image_on_cuda = image_on_cuda
        self.label = str(MetadriveSumoGym.CONNECTION_LABEL)
        MetadriveSumoGym.CONNECTION_LABEL += 1
        
        self.sumo = None
        self.disp = None
        self.metadrive_simulation = None
        self.sumo_simulation = None

        self.connect_server_client(args, label="init_connection" + self.label, image_on_cuda=False)
        
        print("init_connection" + self.label)

        if LIBSUMO:
            # traci.start([sumolib.checkBinary("sumo"), "-n", args.sumo_net_file])
            conn = traci
        else:
            # traci.start([sumolib.checkBinary("sumo"), "-n", args.sumo_net_file], label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        self.ts_ids = list(conn.trafficlight.getIDList())
        self.observation_class = observation_class
  
        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    conn,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    conn,
                )
                for ts in self.ts_ids
            }
        self.args = args

        # conn.close()
        self.vehicles = dict()
        self.vehicles_throughput = dict()
        self.inside_vehicles = dict()

        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}

        
        self.synchronization.close()
        self.sumo_simulation = None
        self.metadrive_simulation = None
        self.synchronization = None


    def connect_server_client(self, args, label=None, image_on_cuda=False):
        
        
        if self.sumo_simulation is not None:
            traci.close()

        self.sumo_simulation = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host,
                                        args.sumo_port, args.sumo_gui, args.client_order, label=label,
                                        max_depart_delay=self.max_depart_delay)
        
        # metadrive makes repeat steps with each step length of 0.02s
        if self.metadrive_simulation is None:
            self.metadrive_simulation = MetaDriveSimulation(args, int(args.step_length/PANDA_WORLD_SIZE), image_on_cuda=image_on_cuda, rendering=args.rendering)

        self.metadrive_simulation.world.config["image_on_cuda"] = image_on_cuda
        # hook the environment
        self.metadrive_simulation.world.config["agent_observation"].sumo = self.sumo_simulation

        self.synchronization = SimulationSynchronization(self.sumo_simulation, self.metadrive_simulation, args.tls_manager,
                                                    args.sync_vehicle_color, args.sync_vehicle_lights)
        
    

    def _start_simulation(self):
        
        self.connect_server_client(self.args, label=self.label, image_on_cuda=self.image_on_cuda)

        if LIBSUMO:
            self.sumo = traci
        else:
            self.sumo = traci.getConnection(self.label)

        # if self.args.sumo_gui or self.render_mode is not None:
        #     self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)
        if self.episode != 0:
            # self.close()
            self.save_csv(self.out_csv_name, self.episode)
            

        self.episode += 1
        self.metrics = []
        

        if seed is not None:
            self.sumo_seed = seed

        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    self.sumo,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    self.sumo,
                )
                for ts in self.ts_ids
            }

        def stop_physics(*args, **kwargs):
            pass

        self.metadrive_simulation.world.engine.step_physics_world = stop_physics

        self.vehicles = dict()
        self.vehicles_throughput = dict()
        self.inside_vehicles = dict()
        entering_v = self.sumo.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.sim_step})

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    def tick(self):
        self.synchronization.tick()

        entering_v = self.sumo.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.sim_step})
        exiting_v = self.sumo.simulation.getArrivedIDList()
        for v in exiting_v:
            self.vehicles_throughput.update({v: self.sim_step - self.inside_vehicles[v]})

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self.tick()
        else:
            self._apply_actions(action)
            self._run_steps()
        

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()
        if self.sim_step % 100 == 0:
            print(info)
        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self.tick()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info

    def _compute_observations(self):
        self.observations.update(
            {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        self.rewards.update(
            {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].action_space

    def observation_spaces(self, ts_id: str):
        """Return the observation space of a traffic signal."""
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal."""
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_throughput": 0.0 if self.sim_step ==0 else len(self.vehicles_throughput)*1.0/(self.sim_step-self.begin_time)*3600,
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "system_travel_time": self.get_vehicles()
        }

    def get_vehicles(self):
        '''
        get_vehicles
        Get all vehicle ids.
        
        :param: None
        :return: None
        '''
        result = 0
        count = 0
        for v in self.vehicles_throughput.keys():
            count += 1
            result += self.vehicles_throughput[v]
        if count == 0:
            return 0
        else:
            return result/count

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        total_co2_emission = [self.traffic_signals[ts].get_total_emission() for ts in self.ts_ids]
        mean_co2_emission = [self.traffic_signals[ts].get_mean_emission() for ts in self.ts_ids]
        total_delay = [self.traffic_signals[ts].get_total_delay() for ts in self.ts_ids]
        total_travel_time = [self.traffic_signals[ts].get_total_travel_time() for ts in self.ts_ids]

        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_travel_time"] = sum(total_travel_time)
        info["agents_total_delay"] = sum(total_delay)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        info["agents_total_emission"] = sum(total_co2_emission)
        info["agents_mean_emission"] = sum(mean_co2_emission)
        return info

    def close(self):
        """Close the environment and stop the SUMO simulation."""

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.metrics = []
        self.traffic_signals = None
        if self.metadrive_simulation is not None:
            self.metadrive_simulation.close()
        if self.metadrive_simulation is not None:
            self.sumo_simulation.close()
        self.sumo_simulation = None
        self.metadrive_simulation = None
        self.synchronization = None
        self.vehicles = dict()
        self.vehicles_throughput = dict()
        self.inside_vehicles = dict()

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

            df = pd.DataFrame(self.metrics)
            mean_df = df.mean().to_frame().T
            mean_df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}_mean" + ".csv", index=False)
        

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        """Encode the state of the traffic signal into a hashable object."""
        
        state_obs = state['obs']
        state_img = state['image']

        phase = int(np.where(state_obs[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state_obs[self.traffic_signals[ts_id].num_green_phases]

        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + [state_img])

    def _discretize_density(self, density):
        return min(int(density * 10), 9)
        


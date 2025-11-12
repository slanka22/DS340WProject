""" This module is responsible for the management of the metadrive simulation. """
import cv2
# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from metadrive.envs.base_env import BaseEnv

from meta_traffic.metatraffic_env.observation import TopDownObservation, MultiAngleObservation
from meta_traffic.metatraffic_env.policy import SUMOTrafficSignalControlPolicy
from meta_traffic.metatraffic_env.sumo_manager import SUMOManager
from meta_traffic.metatraffic_env.sumo_map_manager import SUMOMapManager
from meta_traffic.metatraffic_env.sumo_ts_agent_manager import PhaseTrafficSignalAgentManager

import gymnasium as gym

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

MetaSUMOEnvConfig = dict(use_render=False,
                         stack_size=1,
                         use_mesh_terrain=True,
                         image_on_cuda=False,
                         num_scenarios=1,
                         start_seed=0,
                         show_crosswalk=True,
                         show_sidewalk=True,
                         map_region_size=1024,
                         decision_repeat=1,  # 1 can drop the cars to the ground and won't harm the performance
                         image_source="main_camera",
                         sensors={"main_camera": ()},
                         window_size=(256, 256),
                         show_interface=False,
                         show_logo=False,
                         show_fps=False,
                         physics_world_step_size=2e-2,

                         # ===== obs/action =====
                         agent_observation=MultiAngleObservation,
                         capture_all_at_once=True,
                         agent_policy=SUMOTrafficSignalControlPolicy,
                         image_observation=True,
                         top_down_camera_initial_z=10,
                         vision_feature=True,
                         sumo_feature=True,
                         pitch=-35,

                         # ===== SUMO =====
                         #  sumo_cfg_file="/home/ageratum/Desktop/Simulation/metatraffic-tsc/nets/2way-single-intersection/single-intersection.sumocfg",
                        #  sumo_cfg_file="nets/RESCO/cologne1/cologne1.sumocfg",
                         sumo_cfg_file="nets/2way-single-intersection/single-intersection.sumocfg",
                         # -intersection.sumocfg",
                         fixed_ts=False,
                         reward_fn="diff-waiting-time",  # Union[str, Callable, dict]
                         begin_time=0,
                         delta_time=5,
                         yellow_time=2,
                         min_green=10,
                         max_green=50,
                         step_length=1,
                         sumo_host=None,
                         sumo_port=None,
                         sumo_gui=False,
                         client_order=1,
                         sync_vehicle_color=False,
                         sync_vehicle_lights=False,
                         need_lane_localization=False,
                         show_line_indicator_for_traffic_light=True,  # show line on terrain
                         show_traffic_light_model=False,  # show model
                         )


class MetaSUMOEnv(BaseEnv):
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(self,
                 config,
                 out_csv_name: Optional[str] = None,
                 num_seconds: int = 100000,
                 single_agent: bool = True,
                 add_system_info: bool = True,
                 add_per_agent_info: bool = True,
                 sumo_warnings: bool = True):

        self.single_agent = single_agent
        self.sumo_warnings = sumo_warnings
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(MetaSUMOEnv.CONNECTION_LABEL)
        MetaSUMOEnv.CONNECTION_LABEL += 1

        super(MetaSUMOEnv, self).__init__(config)
        self.begin_time = self.config["begin_time"]
        self.sim_max_time = self.config["begin_time"] + num_seconds
        self.delta_time = self.config["delta_time"]  # seconds on sumo at each step
        self.min_green = self.config["min_green"]
        self.max_green = self.config["max_green"]
        self.yellow_time = self.config["yellow_time"]
        self.reward_fn = self.config["reward_fn"]

        # stat
        self.episode = 0
        self.out_csv_name = out_csv_name
        self.rewards = None
        self.metrics = None
        self.observations_ts = None

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.agent_manager.conn.simulation.getTime()

    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # No action, follow fixed TL defined in self.phases
        obs, _, _, _, _ = super(MetaSUMOEnv, self).step(action)

        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        # import traci
        # ts = self.traffic_signals[self.ts_ids[0]]
        # print(ts.env.sim_step,'xxxxx')
        # img = ts.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
        #                                 f"temp/img_1.jpg",
        #                                 width=1024,
        #                                 height=1024)

        # if img:
        #     import cv2
        #     cv2.imshow('image_x', img)
        #     cv2.waitKey(1)

        # cv2.imshow('image', (obs['image'][...,-1]*255).astype(np.uint8))
        # cv2.waitKey(1)
        if self.sim_step % 200 == 0:
            print(info)

        # assert obs
        if self.single_agent:
            return obs, rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return obs, rewards, dones, info

    # @time_me
    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_rewards(self):
        self.rewards.update(
            {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if
             self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    # @property
    # def observation_space(self):
    #     """Return the observation space of a traffic signal.
    #
    #     Only used in case of single-agent environment.
    #     """
    #     return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].action_space

    # def observation_spaces(self, ts_id: str):
    #     """Return the observation space of a traffic signal."""
    #     return self.traffic_signals[ts_id].observation_space
    #
    # def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
    #     """Return the action space of a traffic signal."""
    #     return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.agent_manager.conn.simulationStep()

    @property
    def inside_vehicles(self):
        return self.engine.sumo_manager.inside_vehicles

    @property
    def vehicles_throughput(self):
        return self.engine.sumo_manager.vehicles_throughput

    def _get_system_info(self):
        vehicles = self.agent_manager.conn.vehicle.getIDList()
        speeds = [self.agent_manager.conn.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.agent_manager.conn.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_throughput": 0.0 if self.sim_step == self.begin_time else len(self.vehicles_throughput) * 1.0 / (
                    self.sim_step - self.begin_time) * 3600,
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "system_travel_time": self.get_vehicles()
        }

    def get_vehicles(self):
        """
        get_vehicles
        Get all vehicle ids.

        :param: None
        :return: None
        """
        result = 0
        count = 0
        for v in self.vehicles_throughput.keys():
            count += 1
            result += self.vehicles_throughput[v]
        if count == 0:
            return 0
        else:
            return result / count

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
        super(MetaSUMOEnv, self).close()
        self.metrics = []

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

            print(mean_df['agents_total_emission'], '\n',
                  mean_df['agents_total_stopped'], '\n',
                  mean_df['agents_total_accumulated_waiting_time'], '\n',
                  mean_df['agents_total_delay'], '\n',
                  mean_df['system_travel_time'])

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

    @property
    def traffic_signals(self):
        return self.agent_manager.traffic_signals

    @property
    def ts_ids(self):
        return self.agent_manager.traffic_signal_ids

    def reset(self, seed=None):
        o, _ = super(MetaSUMOEnv, self).reset(seed=seed)

        self.reward_range = (-float("inf"), float("inf"))
        self.rewards = {ts: None for ts in self.ts_ids}
        self.observations_ts = {ts: None for ts in self.ts_ids}

        if self.episode != 0:
            # self.close()
            self.save_csv(self.out_csv_name, self.episode)

        self.episode += 1
        self.metrics = []

        return o, self._compute_info()

    @classmethod
    def default_config(cls):
        config = super(MetaSUMOEnv, cls).default_config()
        config.update(MetaSUMOEnvConfig)
        return config

    def reset_sensors(self):
        set_main_cam = False
        for node in self.engine.sumo_manager.sumo.net.getNodes():
            if "traffic_light" in node.getType():
                self.center_p = node._coord
                self.main_camera.stop_track()
                if self.config["agent_observation"] == TopDownObservation:
                    self.main_camera.set_bird_view_pos(self.center_p[:2])
                    # self.main_camera.set_bird_view_pos_hpr(center_p[:2], hpr=[-90, -90, 0])
                elif self.config["agent_observation"] == MultiAngleObservation:
                    self.main_camera.set_bird_view_pos_hpr(self.center_p[:2], hpr=[0, -15, 0])
                # else:
                #     raise ValueError("Use TopDownObservation or MultiAngleObservation")
                set_main_cam = True
                break
        assert set_main_cam, "Main camera is not set to proper positions"

    def _get_agent_manager(self):
        return PhaseTrafficSignalAgentManager(single_agent=self.single_agent,
                                              init_observations=self._get_observations())

    def setup_engine(self):
        super(MetaSUMOEnv, self).setup_engine()
        self.engine.register_manager("sumo_manager", SUMOManager())
        self.engine.register_manager("map_manager", SUMOMapManager())
        assert list(self.engine.managers)[-3] == "sumo_manager", "sumo_manager should have the lowest priority"

    def cost_function(self, object_id: str):
        return 0, {}


if __name__ == "__main__":

    # Best result: image_on_cuda=True + capture_all_at_once=False
    sleep=True
    render = True
    image_on_cuda = False

    print('xxxxxxxxx')
    env = MetaSUMOEnv(dict(
        image_on_cuda=image_on_cuda,
        begin_time=0,
        sumo_gui=True,
        use_render=False,
        agent_observation=TopDownObservation,
        delta_time=5,
        # window_size=(224, 224),
        # capture_all_at_once=False,
    ))
    print('xxxxxxxxx')
    env.reset()
    print('xxxxxxxxx')
    total_time = 0
    start = time.time()
    for i in range(10):
        for s in range(1, 10000):
            # start_1 = time.time()
            o, r, tm, tc, i = env.step(env.action_space.sample())
  
            # env.render(mode="top_down",
            #         screen_size=(512, 512),
            #         num_stack=1,
            #         camera_position=env.center_p[:2],
            #         scaling=2,)

            # img = env.engine.agent_manager.conn.gui.screenshot(env.engine.agent_manager.conn.gui.DEFAULT_VIEW,
            #                                                    "img_sumo.jpg",
            #                                                    width=512,
            #                                                    height=512)

            # import cv2

            # img = cv2.imread("img_sumo.jpg")
            # if img is not None:
            #     cv2.imshow('img_sumo', img)
            #     cv2.waitKey(1)

            if s % 720 == 0:
                total_time += time.time() - start
                print("Average Simulation FPS: {}, "
                    "Average step time: {} second".format(s / total_time, total_time / s))
                print("Average SUMO tick FPS: {}, "
                    "Average step time: {} second".format(500 / env.engine.sumo_manager.sumo_total_time,
                                                            env.engine.sumo_manager.sumo_total_time / 500))
                env.reset()
                start = time.time()

            if render:
                for id, img in o.items():
                    if image_on_cuda and "image" in id:
                        img = img.get()
                    cv2.imshow(id, img[..., -1])
                    cv2.waitKey(1)
            if s > 100 and sleep:
                time.sleep(2)
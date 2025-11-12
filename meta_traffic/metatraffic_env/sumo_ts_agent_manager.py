import os
import copy
from meta_traffic.metatraffic_env.traffic_signal import TrafficSignal
import sys
from collections import OrderedDict

import gymnasium as gym
from metadrive.base_class.base_runnable import BaseRunnable
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseAgentManager

from meta_traffic.metatraffic_env.sumo_integration.sumo_simulation import SumoSimulation

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import os
import traci

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class AllTrafficSignals(BaseRunnable):
    """
    This is an interface making a bundle of traffic signals look like a single object.
    Thus, it can be processed by the agentManager
    """

    def __init__(self, lights, link_id2ts_id, link2index):
        super(AllTrafficSignals, self).__init__()
        self.lights = lights
        self.link_id2ts_id = link_id2ts_id
        self.link2index = link2index
        self.index2link = {v: k for k, v in self.link2index.items()}

    def before_step(self, actions, set_link_state=True):
        """
        It communicates with the sumo and set the sumo light status
        """
        for link_id, action in actions.items():
            obj = self.lights[link_id]
            if action == 0 or action in ["r", "R"]:
                action = "R"
                obj.set_red()
            elif action == 1 or action in ["y", "Y"]:
                action = "Y"
                obj.set_yellow()
            elif action == 2 or action in ["g", "G"]:
                action = "G"
                obj.set_green()
            if set_link_state:
                traci.trafficlight.setLinkState(self.link_id2ts_id[link_id], self.link2index[link_id], action)
        return {}


class TrafficSignalAgentManager(BaseAgentManager):
    """
    This class allows using traffic signal as agent to control the traffic flow.
    It is designed for single agent environment, but an agent in this env is a set of lights
    """
    CONNECTION_LABEL = 0
    PRIORITY = 5  # loaded before the sumo_traffic_manager

    def __init__(self, init_observations):
        self.delta_time = self.global_config["delta_time"]
        self.yellow_time = self.global_config["yellow_time"]
        self.min_green = self.global_config["min_green"]
        self.max_green = self.global_config["max_green"]
        self.begin_time = self.global_config["begin_time"]
        self.reward_fn = self.global_config["reward_fn"]
        self.fixed_ts = self.global_config["fixed_ts"]
        self.label = str(self.CONNECTION_LABEL)
        self.traffic_signal_ids = None
        self.sumo = None
        self.vehicles_sumo = dict()

        # visualization & debug
        self._draw_lines_for_light = self.global_config["show_line_indicator_for_traffic_light"]
        self._draw_lines_for_light = self._draw_lines_for_light and self.global_config["use_render"]
        self._draw_model_for_light = self.global_config["show_traffic_light_model"]
        self._draw_model_for_light = self._draw_model_for_light and self.global_config["use_render"]

        # for getting {agent_id: BaseObject}, use agent_manager.active_agents
        self._active_objects = {}  # {object.id: BaseObject}

        # fake init. before creating engine, it is necessary when all objects re-created in runtime
        self.observations = copy.copy(init_observations)  # its value is map<agent_id, obs> before init() is called
        self._init_observations = init_observations  # map <agent_id, observation>

        # restart sumo here to know the observation_space
        self._restart_sumo()  # start sumo, so we know the action space by detecting how many lanes are under control

        # init spaces before initializing env.engine
        observation_space = {
            agent_id: single_obs.observation_space
            for agent_id, single_obs in init_observations.items()
        }
        init_action_space = self._get_action_space()
        assert isinstance(init_action_space, dict)
        assert isinstance(observation_space, dict)
        self._init_observation_spaces = observation_space
        self._init_action_spaces = init_action_space
        self.observation_spaces = copy.copy(observation_space)
        self.action_spaces = copy.copy(init_action_space)
        self.episode_created_agents = None

        # this map will be override when the env.init() is first called and objects are made
        self._agent_to_object = {k: k for k in self.observations.keys()}  # no target objects created, fake init
        self._object_to_agent = {k: k for k in self.observations.keys()}  # no target objects created, fake init

        self._debug = None

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.conn.simulation.getTime()

    def _restart_sumo(self):
        if self.sumo:
            self.sumo.close()
        from metadrive.engine.engine_utils import get_global_config
        self.sumo = SumoSimulation(get_global_config()["sumo_cfg_file"],
                                   get_global_config()["step_length"],
                                   get_global_config()["sumo_host"],
                                   get_global_config()["sumo_port"],
                                   get_global_config()["sumo_gui"],
                                   get_global_config()["client_order"],
                                   label="init_connection" + self.label)
        self.conn = traci if LIBSUMO else traci.getConnection("init_connection" + self.label)
        self.traffic_signal_ids = list(self.conn.trafficlight.getIDList())
        self._create_traffic_signal()

    def _create_traffic_signal(self):
        target = self.reward_fn.keys if isinstance(self.reward_fn, dict) else self.traffic_signal_ids
        self.traffic_signals = {
            ts: TrafficSignal(
                self,
                ts,
                self.delta_time,
                self.yellow_time,
                self.min_green,
                self.max_green,
                self.begin_time,
                self.reward_fn[ts] if isinstance(self.reward_fn, dict) else self.reward_fn,
                self.conn,
            )
            for ts in target}
        list(self.observations.values())[0].set_traffic_signal(self.traffic_signals[self.traffic_signal_ids[0]])

    def reset(self):
        self._restart_sumo()
        super(TrafficSignalAgentManager, self).reset()  # create lights and policies

    def _get_action_space(self):
        """
        The action space of the single agent env equals to (the num of the lights, ). Each element is chosen from 0,1,2
        """
        assert len(self.observations) == 1, "Currently, this manager only supports single agent setting"
        space = {}
        for light_id in self.traffic_signal_ids:
            for ret in traci.trafficlight.getControlledLinks(light_id):
                assert len(ret) == 1
                space[ret[0][-1]] = gym.spaces.Discrete(3)
        return {DEFAULT_AGENT: gym.spaces.Dict(space)}

    def _create_agents(self, config_dict):
        assert len(config_dict) == 1, "Currently, this manager only supports single agent setting"
        lights = OrderedDict()
        link_id2ts_id = {}
        link2index = {}
        for id in self.traffic_signal_ids:
            # Get traffic light position
            # position = traci.junction.getPosition(id)
            for idx, ret in enumerate(traci.trafficlight.getControlledLinks(id)):
                assert len(ret) == 1
                link_id = ret[0][-1]
                lane = self._look_up_lane(link_id)
                obj = self.spawn_object(BaseTrafficLight,
                                        lane=lane,
                                        draw_line=self._draw_lines_for_light,
                                        show_model=self._draw_model_for_light)
                lights[link_id] = obj
                link_id2ts_id[link_id] = id
                link2index[link_id] = idx
        signals = AllTrafficSignals(lights, link_id2ts_id, link2index)
        self.add_policy(signals.id, self.agent_policy, signals, self.generate_seed())
        return {DEFAULT_AGENT: signals}

    def _look_up_lane(self, lane_id):
        for lane_index, lane_info in self.engine.current_map.road_network.graph.items():
            if lane_id in lane_index:
                return lane_info.lane

    def destroy(self):
        super(TrafficSignalAgentManager, self).destroy()
        self.sumo.close()


class PhaseTrafficSignalAgentManager(TrafficSignalAgentManager):
    def __init__(self, single_agent, *args, **kwargs):
        super(PhaseTrafficSignalAgentManager, self).__init__(*args, **kwargs)
        self.traffic_signals = None
        self.single_agent = single_agent

    def try_actuate_agent(self, step_infos, stage="before_step"):
        """
        Some policies should make decision before physics world actuation, in particular, those need decision-making
        But other policies like ReplayPolicy should be called in after_step, as they already know the final state and
        exempt the requirement for rolling out the dynamic system to get it.
        """
        assert stage == "before_step" or stage == "after_step"
        for agent_id in self.active_agents.keys():
            policy = self.get_policy(self._agent_to_object[agent_id])
            assert policy is not None, "No policy is set for agent {}".format(agent_id)
            if stage == "before_step":
                action = policy.act(agent_id)
                step_infos[agent_id] = policy.get_action_info()
                action = self._preprocess_action(action)
                step_infos[agent_id].update(self.get_agent(agent_id).before_step(action, set_link_state=False))
        return step_infos

    def _preprocess_action(self, actions):
        """Set the next green phase for the traffic signals. This will be synchronized to AllTrafficSignals as well.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.traffic_signal_ids[0]].time_to_act:
                self.traffic_signals[self.traffic_signal_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)
        assert self.single_agent
        states = ""
        for light_id in self.traffic_signal_ids:
            states += traci.trafficlight.getRedYellowGreenState(light_id)

        actions = {}
        assert len(states) == len(self.active_agents[DEFAULT_AGENT].lights)
        for idx, link_id in enumerate(self.active_agents[DEFAULT_AGENT].lights):
            actions[link_id] = states[idx]
        return actions

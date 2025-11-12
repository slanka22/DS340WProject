""" This module is responsible for the management of the metadrive simulation. """

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================
import time

from metadrive.constants import DEFAULT_AGENT
from metadrive.envs.base_env import BaseEnv
from metadrive.obs.observation_base import DummyObservation
from meta_traffic.metatraffic_env.observation import TopDownObservation
from meta_traffic.metatraffic_env.policy import SUMOTrafficSignalControlPolicy
from meta_traffic.metatraffic_env.sumo_manager import SUMOManager
from meta_traffic.metatraffic_env.sumo_map_manager import SUMOMapManager
from meta_traffic.metatraffic_env.sumo_ts_agent_manager import TrafficSignalAgentManager

MetaSUMOTrafficLightEnvConfig = dict(use_mesh_terrain=True,
                                     num_scenarios=1,
                                     start_seed=0,
                                     show_crosswalk=True,
                                     show_sidewalk=True,
                                     map_region_size=512,
                                     decision_repeat=1,
                                     window_size=(800, 800),
                                     show_interface=False,
                                     show_logo=False,
                                     show_fps=False,
                                     physics_world_step_size=2e-2,

                                     # ===== Observations =====
                                     image_observation=True,
                                     image_source="main_camera",
                                     sensors={"main_camera": ()},
                                     stack_size=1,
                                     image_on_cuda=False,
                                     agent_observation=TopDownObservation,
                                     agent_policy=SUMOTrafficSignalControlPolicy,
                                     agent_config={DEFAULT_AGENT: dict(draw_line=True)},

                                     # agent_observation=DummyObservation,
                                     # image_observation=False,

                                     # ===== SUMO =====
                                     sumo_cfg_file="nets/hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.sumocfg",
                                     fixed_ts=False,
                                     reward_fn="diff-waiting-time",  # Union[str, Callable, dict]
                                     begin_time=0,
                                     delta_time=5,
                                     yellow_time=2,
                                     min_green=10,
                                     max_green=50,
                                     stop_physics=True,
                                     # harm physics simulation performance but useless in traffic signal control
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

                                     # ===== reward/done/cost =====
                                     # truncate will be true after physics_world_step_size * decision_repeat * horizon
                                     horizon=4000,  # currently, 400 second
                                     )


class MetaSUMOTrafficLightEnv(BaseEnv):

    @classmethod
    def default_config(cls):
        config = super(MetaSUMOTrafficLightEnv, cls).default_config()
        config.update(MetaSUMOTrafficLightEnvConfig)
        return config

    def reset_sensors(self):
        if not self.main_camera:
            return
        set_main_cam = False
        for node in self.engine.sumo_manager.sumo.net.getNodes():
            if "traffic_light" in node.getType():
                center_p = node._coord
                self.main_camera.stop_track()
                self.main_camera.set_bird_view_pos(center_p[:2])
                self.main_camera.top_down_camera_height = 50
                set_main_cam = True
                break
        assert set_main_cam, "Main camera is not set to proper positions"

    def reset(self, *args, **kwargs):
        ret = super(MetaSUMOTrafficLightEnv, self).reset(*args, **kwargs)

        # monkey-patch, it can speed up the simulation by 10~20 fps
        if self.config["stop_physics"]:
            def stop_physics(*args, **kwargs):
                pass

            self.engine.step_physics_world = stop_physics

        return ret

    def setup_engine(self):
        super(MetaSUMOTrafficLightEnv, self).setup_engine()
        self.engine.register_manager("sumo_manager", SUMOManager())
        self.engine.register_manager("map_manager", SUMOMapManager())

    def _get_agent_manager(self):
        return TrafficSignalAgentManager(init_observations=self._get_observations())


if __name__ == "__main__":
    import cv2

    render = True
    image_on_cuda = False
    env = MetaSUMOTrafficLightEnv(dict(image_on_cuda=image_on_cuda,
                                       # use_render=True,
                                       # debug=True,
                                       # debug_static_world=True,
                                       # pstats=True,
                                       window_size=(256, 256)))
    env.reset()
    total_time = 0
    start = time.time()
    for s in range(1, 10000):
        # start_1 = time.time()
        o, r, tm, tc, i = env.step(env.action_space.sample())
        # print("MD Step: {}".format(time.time() - start_1))
        if s % 500 == 0:
            total_time += time.time() - start
            print("Average FPS: {}".format(s / total_time))
            env.reset()
            start = time.time()

        #
        if render:
            img = list(o.values())[0][..., -1]
            if image_on_cuda:
                img = img.get()
            cv2.imshow('RGB Image in Observation', img)
            cv2.waitKey(1)

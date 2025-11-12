import argparse
import os
import sys
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from meta_traffic import SumoEnvironment
from meta_traffic.agents import SOTLAgent

from meta_traffic.metatraffic_env.traffic_signal_env import MetaSUMOEnv
import time

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""SOTL Single-Intersection""",
    )
    argparser.add_argument("sumo_cfg_file", type=str, help="sumo configuration file")
    argparser.add_argument("exp_name", type=str, help="experiment name")
    argparser.add_argument(
        "--sumo-gui", action="store_true", help="run the gui version of sumo"
    )
    argparser.add_argument("--debug", action="store_true", help="enable debug messages")
    argparser.add_argument(
        "-mingreen",
        dest="min_green",
        type=int,
        default=10,
        required=False,
        help="Minimum green time.\n",
    )
    argparser.add_argument(
        "-maxgreen",
        dest="max_green",
        type=int,
        default=30,
        required=False,
        help="Maximum green time.\n",
    )
    argparser.add_argument(
        "-fixed",
        action="store_true",
        default=False,
        help="Run with fixed timing traffic signals.\n",
    )
    argparser.add_argument(
        "-disable_vision_feature",
        action="store_true",
        default=False,
        help="hybrid features",
    )
    argparser.add_argument(
        "-disable_sumo_feature",
        action="store_true",
        default=False,
        help="hybrid features",
    )
    argparser.add_argument(
        "-rendering", action="store_true", default=False, help="rendering metadrive"
    )
    argparser.add_argument(
        "-s",
        dest="seconds",
        type=int,
        default=3600,
        required=False,
        help="Number of simulation seconds.\n",
    )
    argparser.add_argument(
        "-begin_time",
        dest="begin_time",
        type=int,
        default=0,
        required=False,
        help="Number of simulation seconds.\n",
    )
    argparser.add_argument(
        "-epoch",
        dest="epoch",
        type=int,
        default=100,
        required=False,
        help="maps for the simulation\n",
    )

    argparser.add_argument(
        "-runs", dest="runs", type=int, default=1, help="Number of runs.\n"
    )
    arg = argparser.parse_args()
    experiment_time = str(datetime.now()).split(".")[0].replace(" ", "")
    out_csv = f"outputs/{arg.exp_name}/stol_noise/csv/{experiment_time}/_"

    env = MetaSUMOEnv(
        dict(
            image_on_cuda=False,
            begin_time=arg.begin_time,
            delta_time=5,
            window_size=(256, 256),
            stack_size=1,
            min_green=arg.min_green,
            sumo_cfg_file=arg.sumo_cfg_file,
            sumo_gui=arg.sumo_gui,
            capture_all_at_once=True,
            vision_feature=False,
            sumo_feature=True
        ),
        num_seconds=arg.seconds,
        out_csv_name=out_csv,
    )

    for run in range(1, arg.runs + 1):

        o, _  = env.reset()

        # sotl_agents = {
        #     ts: SOTLAgent(
        #         env=env,
        #         ts_id=ts,
        #     )
        #     for ts in env.ts_ids
        # }
        ts_id = env.ts_ids[0]
        sotl_agent = SOTLAgent(
            env=env,
            ts_id=ts_id,
            )
        
        start = time.time()

        done = False
        infos = []
        i = 0
        while not done:
            action = sotl_agent.act() 
            s, r, _, done, info = env.step(action)
            print(info)
            i = i + 1
            if i % 1 == 0:
                print("Average FPS: {}".format(i / (time.time() - start)))

        env.save_csv(out_csv, run)
        env.close()

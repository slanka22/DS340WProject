import argparse
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


from meta_traffic import MetadriveSumoGym
from datetime import datetime
from meta_traffic.environment.observations import (
    ImageObservationFunction,
    ImageOnlyObservationFunction,
)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("sumo_cfg_file", type=str, help="sumo configuration file")
    argparser.add_argument("exp_name", type=str, help="experiment name")
    argparser.add_argument(
        "--sumo-host",
        metavar="H",
        default=None,
        help="IP of the sumo host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "--sumo-port",
        metavar="P",
        default=None,
        type=int,
        help="TCP port to listen to (default: 8813)",
    )
    argparser.add_argument(
        "--sumo-gui", action="store_true", help="run the gui version of sumo"
    )
    argparser.add_argument(
        "--step-length",
        default=1,
        type=float,
        help="set fixed delta seconds (default: 0.04s)",
    )
    argparser.add_argument(
        "--client-order",
        metavar="TRACI_CLIENT_ORDER",
        default=1,
        type=int,
        help="client order number for the co-simulation TraCI connection (default: 1)",
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
        "--sync-vehicle-lights",
        action="store_true",
        help="synchronize vehicle lights state (default: False)",
    )
    argparser.add_argument(
        "--sync-vehicle-color",
        action="store_true",
        help="synchronize vehicle color (default: False)",
    )
    argparser.add_argument(
        "--sync-vehicle-all",
        action="store_true",
        help="synchronize all vehicle properties (default: False)",
    )
    argparser.add_argument(
        "--tls-manager",
        type=str,
        choices=["none", "sumo", "meta"],
        help="select traffic light manager (default: none)",
        default="sumo",
    )
    argparser.add_argument(
        "-fixed",
        action="store_true",
        default=False,
        help="Run with fixed timing traffic signals.\n",
    )
    argparser.add_argument(
        "-vision_only", action="store_true", default=False, help="hybrid features"
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
    arguments = argparser.parse_args()

    experiment_time = str(datetime.now()).split(".")[0].replace(" ", "")
    vision_only = "only" if arguments.vision_only else ""
    out_csv = f"outputs/{arguments.exp_name}/VisionA2C_{vision_only}_2way/csv/{experiment_time}/_"
    env = MetadriveSumoGym(
        arguments,
        observation_class=ImageOnlyObservationFunction
        if arguments.vision_only
        else ImageObservationFunction,
        out_csv_name=out_csv,
        single_agent=True,
        min_green=arguments.min_green,
        max_green=arguments.max_green,
        num_seconds=arguments.seconds,
        image_on_cuda=False,  # unfixed bug
        max_depart_delay=-1,
    )
    env.reset()
    for _ in range(10000):
        env.step(env.action_space.sample())


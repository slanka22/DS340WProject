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
from meta_traffic.agents import MaxPressureAgent


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Max Pressure Single-Intersection"""
    )
    prs.add_argument("exp_name", type=str, help="experiment name")
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument(
        "-net",
        dest="net",
        type=str,
        default="nets/2way-single-intersection/single-intersection.net.xml",
        help="Network definition xml file.\n",
    )
    prs.add_argument(
        "-begin_time",
        dest="begin_time",
        type=int,
        default=0,
        required=False,
        help="Number of simulation seconds.\n",
    )
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=3600, required=False, help="Number of simulation seconds.\n")

    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0].replace(" ", "")
    out_csv = f"outputs/{args.exp_name}/maxpressure_noise/csv/{experiment_time}/_"

    env = SumoEnvironment(
        net_file=args.net,
        route_file=args.route,
        begin_time=args.begin_time,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        fixed_ts=False,
        # single_agent=True,
        sumo_warnings=False,
    )

    for run in range(1, args.runs + 1):
        initial_states = env.reset()
        maxpressure_agents = {
            ts: MaxPressureAgent(
                env=env,
                ts_id=ts,
            )
            for ts in env.ts_ids
        }

        done = {"__all__": False}
        infos = []

        while not done["__all__"]:
            actions = {ts: maxpressure_agents[ts].act() for ts in maxpressure_agents.keys()}
            s, r, done, info = env.step(action=actions)

        env.save_csv(out_csv, run)
        env.close()
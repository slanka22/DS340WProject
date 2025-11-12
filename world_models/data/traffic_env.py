"""
Generating data from the gym environment.
"""
import argparse
import glob
import os
from os.path import join, exists
import numpy as np
from meta_traffic.metatraffic_env.traffic_signal_env import MetaSUMOEnv
from meta_traffic.metatraffic_env.observation import TopDownObservation, MultiAngleObservation
from world_models.utils.misc import sample_discrete_policy

import re

def find_max_number(folder_path, pattern):
    max_number = 0
    regex_pattern = re.compile(pattern)

    for filename in os.listdir(folder_path):
        match = regex_pattern.search(filename)
        if match:
            number = int(match.group(1))
            max_number = max(number, max_number)

    return max_number


def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    env = MetaSUMOEnv(dict(begin_time=0,
                            delta_time=5,
                            window_size=(256, 256),
                            stack_size=1,
                            sumo_gui=False,
                            capture_all_at_once=True,
                            vision_feature=True,
                            sumo_feature=True,
                            agent_observation=TopDownObservation,
                            top_down_camera_initial_z = 100
                            ), num_seconds=3600)
    

    seq_len = (env.sim_max_time - env.begin_time) // 5
    
    pattern = r'rollout_(\d+)\.npz'

    start_rollouts = find_max_number(data_dir, pattern)
    print(start_rollouts, rollouts)
    for i in range(start_rollouts,rollouts):
        env.reset()
        # get random actions
        print(f"> Starting rollout {i}...")
        a_rollout = sample_discrete_policy(env.action_space, seq_len)
        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0 
        while True:
            action = a_rollout[t]
            # print(f"> Step {t} with action {action}...")
            t += 1
            s, r, done, truncated, _ = env.step(action)
            image = np.squeeze(s["image"])[..., ::-1]
            scaled_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            scaled_image = scaled_image.astype(np.uint8)
            s_rollout.append(scaled_image)
            r_rollout.append(r)
            d_rollout.append(done)

            if done or truncated or t >= seq_len: 
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)

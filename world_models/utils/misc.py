""" Various auxiliary utilities """
import math
from os.path import join, exists
import time
from matplotlib import pyplot as plt
import torch
import torch.multiprocessing
from torchvision import transforms
import numpy as np
from meta_traffic.metatraffic_env.observation import TopDownObservation
from meta_traffic.metatraffic_env.traffic_signal_env import MetaSUMOEnv
from world_models.models import MDRNNCell, VAE, Controller

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    4, 32, 256, 64, 256

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def sample_discrete_policy(action_space, seq_len):
    """ Sample a sequence of actions from a discrete policy.

    :args action_space: action space of the environment
    :args seq_len: length of the sequence

    :returns: sequence of actions
    """
    return [action_space.sample() for _ in range(seq_len)]

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit, Test=False):
        """ Build vae, rnn, controller and environment. """
        print(torch.cuda.is_available())
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
        self.Test = Test
        self.out_csv = None

        if self.Test:
            self.out_csv = mdir + "/output/" + "_eval"
        

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        print(torch.cuda.is_available())
        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        print("defining env")
        self.env = MetaSUMOEnv(dict(begin_time=0,
                            delta_time=5,
                            window_size=(256, 256),
                            stack_size=1,
                            sumo_gui=False,
                            capture_all_at_once=True,
                            vision_feature=True,
                            sumo_feature=True,
                            agent_observation=TopDownObservation,
                            top_down_camera_initial_z = 100,
                            ), num_seconds=3600,
                                out_csv_name=self.out_csv)

        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 256 x 256) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        guess, latent_mu, _ = self.vae(obs)

        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        if self.Test:
            self.env.reset()

        obs = self.env.reset()
        obs = obs[0]

        # This first render is required !
        self.env.render()

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        episode = 0
        done = False
        while not done:
            obs = np.squeeze(obs["image"])[..., ::-1]
            obs = transform(obs).unsqueeze(0).to(self.device)
            # print()
            # print("hidden", torch.sum(hidden[0]))
            # print("obs", torch.sum(obs))
            # print("latent", torch.sum(self.vae(obs)[0]))
            # print()
            action, hidden = self.get_action_and_transition(obs, hidden)
            action1 = np.argmax(action)
            obs, reward, done, *_ = self.env.step(action1)
            # print("Action weights", action, "Action:", action1, end="\r", flush=True)

            if render:
                self.env.render(mode='rgb_array')

            cumulative += reward

            # this is for testing and reporting the results
            if i > self.time_limit and self.Test:
                print("Done", cumulative)
                episode += 1
                i = 0
                self.env.reset()
                obs = self.env.reset()
                obs = obs[0]

            if episode > 2:
                break
            
            if done or i > self.time_limit and not self.Test:
                return - cumulative
            i += 1
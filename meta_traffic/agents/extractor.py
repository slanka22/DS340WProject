import argparse
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import gymnasium as gym

eps = np.finfo(np.float32).eps.item()

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

class ImageExtractor(nn.Module):
    def __init__(self, observation_space, init_weights=True):
        super().__init__()

        input_channel = observation_space.shape[2] * observation_space.shape[3]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1,1)), 
            nn.Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



        # model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='dfn2b')
        # self.clip = model
        # self.preprocess = preprocess
        # self.preprocess.transforms = self.preprocess.transforms[-1:]

        # for param in self.clip.parameters():
        #     param.requires_grad = False
    

    def forward(self, x):

        batch, height, width, channel, time = x.size()
        x = x.permute(0,3,4,1,2).reshape(batch, -1, height, width)
            # x = self.preprocess(x)
            # x = self.clip.encode_image(x)
        x = self.cnn(x)

        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        m.bias.data.fill_(0)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        
        
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if "image" in key:
                if 'shared' not in extractors.keys():
                    extractors['shared'] = ImageExtractor(subspace)
                proj = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 32),
                    )   
                proj.apply(init_weights)
                extractors[key] = proj
                total_concat_size += 32
            elif key == "observation":
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
           
            if "image" in key:
                encoded_tensor_list.append(extractor(self.extractors['shared'](observations[key])))
            elif key == "observation":
                encoded_tensor_list.append(extractor(observations[key]))

        
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.

        return torch.cat(encoded_tensor_list, dim=1)


        
import torch
from gymnasium import Env
from gymnasium.spaces import utils
from torch import nn

from torch.functional import F


class DQN(nn.Module):
    def __init__(self, env: Env):
        super(DQN, self).__init__()

        n_observations = utils.flatdim(env.observation_space)
        n_actions = utils.flatdim(env.action_space)

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Model(nn.Module):
    def __init__(self, env: Env):
        super(Model, self).__init__()

        self.n_observations = utils.flatdim(env.observation_space)

        self.layer1 = nn.Linear(self.n_observations + 1, 32)
        torch.nn.init.xavier_uniform(self.layer1.weight)
        self.layer2 = nn.Linear(32, 32)
        torch.nn.init.xavier_uniform(self.layer2.weight)
        self.layer3 = nn.Linear(32, self.n_observations + 2)
        torch.nn.init.xavier_uniform(self.layer3.weight)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        obs, reward, terminal = x.split([self.n_observations, 1, 1], 1)
        terminal = F.sigmoid(terminal)
        return obs, reward, terminal

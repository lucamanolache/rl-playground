import functools
import operator

import torch
from gymnasium import Env
from gymnasium.spaces import utils

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.GELU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DQN(nn.Module):
    def __init__(self, env: Env):
        super(DQN, self).__init__()

        n_observations = env.observation_space.shape
        n_actions = utils.flatdim(env.action_space)

        block = ResidualBlock
        layers = [3, 4, 6, 3]

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)

        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.maxpool,
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.avgpool
        )

        fc_size = functools.reduce(operator.mul, list(self.feature_extractor(torch.ones(1, *[n_observations[2], n_observations[0], n_observations[1]])).shape))
        self.fc1 = nn.Linear(fc_size, 512)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(512, n_actions)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

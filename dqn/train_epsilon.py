import sys

import gymnasium as gym
import numpy as np
from gymnasium.spaces import utils
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from torch import nn, Tensor

from common.replay_buffer import ReplayMemory, Transition
from dqn import DQN, Model

import torch
import torch.optim as optim

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

name = "CartPole-v1"
render = False

env = None
if render:
    env = gym.make(name, render_mode='human')
else:
    env = gym.make(name)

policy_net = DQN(env)
target_net = DQN(env)
target_net.load_state_dict(policy_net.state_dict())

model_use = 50
model_net = Model(env)
model_optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
model_losses = []
model_memory = ReplayMemory(5000)
MODEL_BATCH_SIZE = 32

optimizer = optim.AdamW(policy_net.parameters(), lr=10 * LR)
memory = ReplayMemory(10000)

steps_done = 0

n_actions = utils.flatdim(env.action_space)
n_observations = utils.flatdim(env.observation_space)
n_chosen = np.ones(n_actions)


def select_action_upper(state, c=0.5):
    global steps_done
    steps_done += 1

    steps = np.ones(n_actions) * steps_done

    q: Tensor = policy_net(state)
    confidence = c * np.sqrt(steps / n_chosen)
    choice = np.argmax(q.detach().numpy() + confidence)

    n_chosen[choice] += 1
    return torch.tensor([[choice]], dtype=torch.long)


def select_action_epsilon(state, greedy=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)

    if greedy:
        steps_done += 1

    if sample > eps_threshold and greedy:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)


select_action = select_action_epsilon

episode_durations = []


def plot_durations(show_result=False) -> float:
    if model_net is not None:
        plt.figure(0)
        plt.clf()
        plt.plot(model_losses)
        plt.pause(0.001)

    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        plt.pause(0.001)
        return means[-1]

    plt.pause(0.001)
    return 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    if MODEL_BATCH_SIZE != 0:
        if len(model_memory) < MODEL_BATCH_SIZE:
            return

        transitions = transitions + model_memory.sample(MODEL_BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE + MODEL_BATCH_SIZE)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train() -> int:
    global episode_durations

    global policy_net
    global target_net
    global optimizer
    global memory
    global steps_done

    global model_losses
    global model_optimizer
    global model_net
    global model_memory

    if model_net is not None:
        model_net = Model(env)
        model_optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        model_losses = []
        model_memory = ReplayMemory(5000)

    policy_net = DQN(env)
    target_net = DQN(env)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    for i_episode in range(1000):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            if model_net is not None:
                obs_loss = nn.L1Loss()
                reward_loss = nn.L1Loss()
                terminal_loss = nn.CrossEntropyLoss()

                observation_hat, reward_hat, terminated_hat = model_net(torch.cat([state, action], 1))
                if next_state is None:
                    term = torch.tensor([1.0]).unsqueeze(0)
                    next_exp = torch.zeros(1, n_observations)
                else:
                    term = torch.tensor([0.0]).unsqueeze(0)
                    next_exp = next_state

                print("PREDICTED:", observation_hat.detach(), reward_hat.detach(), terminated_hat.detach(),
                      "WAS:", observation, reward, terminated)

                obs_loss = obs_loss(observation_hat, next_exp)
                reward_loss = reward_loss(reward_hat, reward.unsqueeze(0))
                terminal_loss = terminal_loss(terminated_hat, term)

                loss = obs_loss + reward_loss + terminal_loss

                model_losses = np.append(model_losses, loss.item())
                model_optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model_state = Transition(*zip(*memory.sample(1))).state[0]
                for _ in range(model_use):
                    with torch.no_grad():
                        model_action = select_action(model_state, greedy=False)
                        model_next_state, model_reward, model_terminated = model_net(torch.cat([model_state, action], 1))

                        if math.ceil(model_terminated.item()) == 1:
                            model_next_state = None

                        model_memory.push(model_state, model_action, model_next_state, model_reward.squeeze().unsqueeze(0))

                        if model_terminated.item() > 0.5:
                            break

                        model_state = model_next_state

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                if plot_durations() >= 490:
                    print("Finished on episode", i_episode)
                    episode_durations = []
                    return i_episode

                break

    episode_durations = []
    return -1


if __name__ == '__main__':
    times = []
    for _ in range(10):
        times.append(train())

    print(times)

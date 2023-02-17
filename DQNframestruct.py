# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : DQNframestruct.py
# Author: JYW
# Date  : 2023/2/9

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the experience replay buffer
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQN training process
def train_dqn(env, q_network, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = q_network(state_batch).gather(1, action_batch.unsqueeze(-1))

    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = q_network(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = reward_batch + gamma * next_state_values

    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step
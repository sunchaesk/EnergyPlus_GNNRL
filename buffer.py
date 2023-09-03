
import sys
import os

import numpy as np

from collections import deque

class ReplayBuffer(object):
    def __init__(self, buffer_size, obs_space, n_action, n_agent):
        self.buffer_size = buffer_size
        self.n_ant = n_agent
        self.pointer = 0
        self.len = 0
        self.states = deque(maxlen=self.buffer_size)
        self.edge_indices = deque(maxlen=self.buffer_size)
        self.actions = deque(maxlen=self.buffer_size)
        self.rewards = deque(maxlen=self.buffer_size)
        self.next_states = deque(maxlen=self.buffer_size)
        self.next_edge_indices = deque(maxlen=self.buffer_size)
        self.dones = deque(maxlen=self.buffer_size)

    def get_batch(self, batch_size):
        indices = np.random.choice(self.len, batch_size, replace=False)
        ret_states = []
        ret_edge_indices = []
        ret_actions = []
        ret_rewards = []
        ret_next_states = []
        ret_next_edge_indices = []
        ret_dones = []
        for index in indices:
            ret_states.append(self.states[index])
            ret_edge_indices.append(self.edge_indices[index])
            ret_actions.append(self.actions[index])
            ret_rewards.append(self.rewards[index])
            ret_next_states.append(self.next_states[index])
            ret_next_edge_indices.append(self.next_edge_indices[index])
            ret_dones.append(self.dones[index])

        return ret_states, ret_edge_indices, ret_actions, ret_rewards, ret_next_states, ret_next_edge_indices, ret_dones

    def add(self, state, edge_index, action, reward, next_state, next_edge_indices, done):
        self.states.append(state)
        self.edge_indices.append(edge_index)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.next_edge_indices.append(next_edge_indices)
        self.dones.append(done)
        self.pointer = (self.pointer + 1)%self.buffer_size
        self.len = min(self.len + 1, self.buffer_size)

    def buffer_filled_percentage(self):
        return (self.len / self.buffer_size) * 100

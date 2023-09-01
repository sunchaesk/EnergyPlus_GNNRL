
import sys
import os

import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size, obs_space, n_action, n_agent):
        self.buffer_size = buffer_size
        self.n_ant = n_agent
        self.pointer = 0
        self.len = 0
        self.states = np.ndarray()
        self.edge_indices = np.ndarray()
        self.actions = np.ndarray()
        self.rewards = np.ndarray()
        self.next_states = np.ndarray()
        self.next_edge_indices = np.ndarray()
        self.dones = np.ndarray()

    def get_batch(self, batch_size):
        index = np.random.choice(self.len, batch_size, replace=False)
        return self.states[index], self.edge_indices[index], self.actions[index], self.rewards[index], self.next_states[index], self.next_edge_indices[index]

    def add(self, state, edge_index, action, reward, next_state, next_edge_indices):
        self.states[self.pointer] = state
        self.edge_indices[self.pointer] = edge_index
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.next_edge_indices[self.pointer] = next_edge_indices
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1)%self.buffer_size
        self.len = min(self.len + 1, self.buffer_size)

    def buffer_filled_percentage(self):
        return (self.len / self.buffer_size) * 100

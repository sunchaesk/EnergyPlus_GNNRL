
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from buffer import ReplayBuffer
from model import ThermoGRL
from model import Encoder, Graph_Encoder

import gymnasium as gym

import idf_parse as idf

import base
from base import default_args

# Not all of the arguments are used
parser = argparse.ArgumentParser(description="")
parser.add_argument("-idf", type=str, default="./in.idf", help="IDF file location")
parser.add_argument("-eplus_weather_file", type=str, default='./weather.epw', help="Weather file")
parser.add_argument("-ep", type=int, default=100, help="The amount of training episodes, default is 100")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=5e-4, help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float,default=0.1, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-encoder_hidden", type=int, default=256, help="Dimension of the hidden representation encoded by the MLP encoder layer")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-gcn_hidden", type=int, default=256, help="Dimension of the hidden representation of the GCN layer")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=2, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-n_epoch", type=int, default=5, help="Epoch size, default is 5")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.95, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")

# DQN stuff
parser.add_argument("-epsilon", type=float, default=0.9, help="Load a saved model to perform a test run!")

args = parser.parse_args()
args.replay_memory


'''
FEATURE MATRIX features order:
1. indoor temperature
2. direct solar
3. diffuse solar
4. infrared
5. month
6. day
7. hour
8. day_of_week
'''

HANDLE_TO_INDEX = {
    'var-attic_indoor_temperature': 0,
    'var-core_zn_indoor_temperature': 1,
    'var-perimeter_zn_1-indoor_temperature': 2,
    'var-perimeter_zn_2-indoor_temperature': 3,
    'var-perimeter_zn_3-indoor_temperature': 4,
    'var-perimeter_zn_4-indoor_temperature': 5,
    'var_environment_site_outdoor_air_drybulb_temperature': 6,
    'var_environment_site_direct_solar_radiation_rate_per_area': 7,
    'var_environment_site_horizontal_infrared_radiation_rate_per_area': 8,
    'var_environment_site_diffuse_solar_radiation_rate_per_area': 9,
    'var-environment-time-month': 10,
    'var-environment-time-day': 11,
    'var-environment-time-hour': 12,
    'var-environment-time-day_of_week': 13
}

ZONE_TO_VARIABLES = {
    'Outdoors': ['var_environment_site_outdoor_air_drybulb_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
    'Attic': ['var-attic_indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
    'Core_ZN': ['var-core_zn_indoor_temperature', 0, 0, 0, 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
    'Perimeter_ZN_1': ['var-perimeter_zn_1-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
    'Perimeter_ZN_2': ['var-perimeter_zn_2-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
    'Perimeter_ZN_3': ['var-perimeter_zn_3-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
    'Perimeter_ZN_4': ['var-perimeter_zn_4-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
    'Perimeter_ZN_4': ['var-perimeter_zn_4-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week'],
} # variables handles associated for each zone

ZONE_INDEX = {
    'Outdoors': 0,
    'Attic': 1,
    'Core_ZN': 2,
    'Perimeter_ZN_1': 3,
    'Perimeter_ZN_2': 4,
    'Perimeter_ZN_3': 5,
    'Perimeter_ZN_4': 6,
}# zone names numbered from 0

# for convenience
D_INDEX_TO_ZONE = {
    0: 'Outdoors',
    1: 'Attic',
    2: 'Core_ZN',
    3: 'Perimeter_ZN_1',
    4: 'Perimeter_ZN_2',
    5: 'Perimeter_ZN_3',
    6: 'Perimeter_ZN_4'
}

EDGE_INDEX = [
    [4, 6, 2, 2, 2, 1, 5, 3, 0, 0, 3, 3, 4, 0, 4, 1, 6, 5, 5, 5, 6, 6, 4, 4, 4, 3, 1, 1, 1, 2, 2, 3, 0, 6, 5, 0],
    [0, 1, 4, 5, 3, 0, 6, 4, 4, 5, 5, 2, 1, 3, 6, 6, 2, 4, 3, 2, 5, 4, 2, 5, 3, 0, 2, 4, 3, 1, 6, 1, 1, 0, 0, 6]
]# adjacency matrix in COO format
EDGE_ATTR = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]# edge_attributes for each of the connections in COO format


def main():
    env = base.EnergyPlusEnv(default_args)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.nvec[0] # returns value of discrete. action space is defined in multivariate

    num_features = 8
    num_agents = 5
    agents_index = [2,3,4,5,6] # zone indices with controllable agents

    ##### NOTE: debug stuff
    debug_encoder = Encoder(num_features=num_features,
                            hidden_size=12,
                            handle_to_index=HANDLE_TO_INDEX,
                            zone_to_variables=ZONE_TO_VARIABLES,
                            zone_index=ZONE_INDEX)
    debug_graph_encoder = Graph_Encoder(12, 32, edge_index=EDGE_INDEX, edge_attr=EDGE_ATTR)



    buff = ReplayBuffer(args.replay_memory, state_dim, action_dim, num_agents)
    model = ThermoGRL(
        handle_to_index=HANDLE_TO_INDEX,
        zone_to_variables=ZONE_TO_VARIABLES,
        zone_index=ZONE_INDEX,
        edge_index=EDGE_INDEX,
        edge_attr=EDGE_ATTR,
        num_features=num_features,
        encoder_hidden_dim=args.encoder_hidden,
        gcn_hidden_dim=args.gcn_hidden,
        q_net_hidden_dim=args.layer_size,
        action_dim=action_dim
    )
    model_tar = ThermoGRL(
        handle_to_index=HANDLE_TO_INDEX,
        zone_to_variables=ZONE_TO_VARIABLES,
        zone_index=ZONE_INDEX,
        edge_index=EDGE_INDEX,
        edge_attr=EDGE_ATTR,
        num_features=num_features,
        encoder_hidden_dim=args.encoder_hidden,
        gcn_hidden_dim=args.gcn_hidden,
        q_net_hidden_dim=args.layer_size,
        action_dim=action_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    i_episode = 1
    n_episode = args.ep
    total_steps = 0
    epsilon = args.epsilon

    while i_episode < n_episode:
        if i_episode > 100:
            epsilon -= 0.001
            if episilon < 0.02:
                epsilon = 0.02

        i_episode += 1

        done = False
        state = env.reset()

        while not done:
            total_steps += 1

            action = []
            q = model(state)
            for agent_index in agents_index:
                if np.random.rand() < epsilon:
                    a = np.random.choice(action_dim)
                else:
                    np.argmax(q[i].cpu().detach().numpy())

                #action[INDEX_TO_ZONE[agent_index]] = a
                action.append(a)

            next_state, reward, done, truncated, info = env.step(action=action)
            buff.add(state, EDGE_INDEX, action, reward, next_state, EDGE_INDEX)

            state = next_state

        if not buff.buffer_filled_percentage() >= 30:
            continue

        for epoch in range(args.n_epoch):
            states, edge_indices, actions, rewards, next_states, next_edge_indices = buff.get_batch(args.batch_size)


if __name__ == "__main__":
    main()

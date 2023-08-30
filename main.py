
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from buffer import ReplayBuffer
from model import ThermoGRL

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
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=2, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.95, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
args = parser.parse_args()
print(args.repm)
exit(1)


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

HANDLE_TO_INDEX = {'var-perimeter_zn_1-indoor_temperature': 0,
                   'var-attic-indoor_temperature': 1,
                   'var-core_zn-indoor_temperature': 2,
                   'var-perimeter_zn_3-indoor_temperature': 3,
                   'var-perimeter_zn_2-indoor_temperature': 4,
                   'var-perimeter_zn_4-indoor_temperature': 5,
                   'var_environment_site_outdoor_air_drybulb_temperature': 6,
                   'var_environment_site_direct_solar_radiation_rate_per_area': 7,
                   'var_environment_site_horizontal_infrared_radiation_rate_per_area': 8,
                   'var_environment_site_diffuse_solar_radiation_rate_per_area': 9,
                   'var-environment-time-month': 10,
                   'var-environment-time-day': 11,
                   'var-environemnt-time-hour': 12,
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
    'Perimeter_ZN_3': 3,
    'Perimeter_ZN_4': 4,
    'Perimeter_ZN_2': 5,
    'Perimeter_ZN_1': 6,
}# zone names numbered from 0
EDGE_INDEX = [
    [0, 6, 0, 5, 2, 3, 4, 6, 3, 4, 7, 0, 7, 0, 2, 5, 2, 3, 5, 6, 4, 6, 0, 7, 2, 5, 5, 7, 2, 3, 3, 4, 5, 7, 4, 6],
    [7, 5, 3, 2, 5, 2, 2, 4, 5, 5, 0, 2, 3, 5, 4, 4, 6, 4, 6, 0, 6, 7, 6, 5, 0, 0, 7, 4, 3, 0, 7, 7, 3, 6, 3, 2]
]# adjacency matrix in COO format
EDGE_ATTR = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]# edge_attributes for each of the connections in COO format


def main():
    env = base.EnergyPlusEnv(default_args)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.nvec[0] # returns value of discrete. action space is defined in multivariate

    num_agents = 5
    agents_index = [2,3,4,5,6] # zone indices with controllable agents

    buff = ReplayBuffer()


if __name__ == "__main__":
    main()
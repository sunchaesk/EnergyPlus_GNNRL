#https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py

import base

import argparse
import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Discrete
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


import ray
from ray import air, tune
from ray.rllib.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import (
    ModelV2,
    restore_original_dimensions,
    flatten,
    _unpack_obs,
)
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.torch_ops import one_hot

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv

import pprint
pp = pprint.PrettyPrinter(indent=4)

torch, nn = try_import_torch()

def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algos', type=str, default="PPO", help='RLlib registered algorithms'
    )
    parser.add_argument(
        '--iters', type=int, default=50, help='Number of iterations to train'
    )
    parser.add_argument(
        '--timesteps', type=int, default=100000000, help='Number of timesteps to train'
    )
    parser.add_argument(
        '--episodes', type=int, default=10000, help='Number of episodes to train'
    )
    parser.add_argument(
        '--no-tune', action="store_true", help='Run without Tune using a manual train loop'
    )
    parser.add_argument(
        '--local-mode', action="store_true", help='Init ray in local mode for easier debugging'
    )
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="torch",
        help="The deep learning framework specifier",
    )
    parsed_args = parser.parse_args()
    return parsed_args


def ray_run() -> None:
    args = parse_args()
    eplus_env = base.EnergyPlusEnv
    eplus_env_config = vars(args)

    ray.init()

    ray.shutdown()

if __name__ == "__main__":
    print('hello')
    ray_run()

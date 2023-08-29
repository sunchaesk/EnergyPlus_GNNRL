
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv

from typing import Dict, List

class Encoder(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(Encoder, self).__init__()
        self.handle_to_index = handle_to_index
        self.zone_to_variable = zone_to_variables
        self.zone_index = zone_index

        self.num_features = num_features
        self.hidden_size = hidden_size

        self.fc = nn.Linear(self.num_features, self.hidden_size)

    def generate_feature_matrix(self, eplus_obs_vec):
        feature_matrix = []
        for zone_num in zone_index:
            zone_features = []

            curr_zone = self.zone_index[zone_num]
            variable_handles_to_fetch = self.zone_to_variables[curr_zone]
            variable_handles_index = [self.handle_to_index[handle] for handle in variable_handles_to_fetch]
            for variable_handle in variable_handles_index:
                zone_features.append(eplus_obs_vec[variable_handle])

            feature_matrix.append(np.array(zone_features))

        return feature_matrix

    def forward(self, eplus_obs_vec):
        '''
        dim(feature) = num_nodes x num_features
        feature x fc = (num_nodes x num_features) x (num_features, hidden_size) = num_nodes x hidden_size
        '''
        feature = self.generate_feature_matrix(eplus_obs_vec)
        y = self.fc(feature)
        y = F.relu(y)
        return y

class Graph_Encoder(nn.Module):
    def __init__(self,
                 #num_features,
                 encoder_hidden_dim,
                 hidden_dim,
                 edge_index,
                 edge_attr):
        # self.handle_to_index = handle_to_index
        # self.zone_to_variables = zone_to_variables
        # self.zone_index = zone_index
        self.num_features = num_features
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim

        # Based on PyG Data class
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # GCN layer
        self.gcn1 = GCNConv(encoder_hidden_dim, hidden_dim)

    def forward(self, encoded_feature_matrix):
        gnn_data = Data(x=feature_matrix, edge_index = self.edge_index, edge_attr=self.edge_attr)
        y = self.gcn1(gnn_data)
        y = F.relu(y)
        return y


class Q_Net(nn.Module):
    def __init__(self, gcn_encode_dim, hidden_dim, action_dim):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(gcn_encode_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = F.relu(y)
        y = self.out(y)
        return y

class ThermoGRL(nn.Module):
    def __init__(self,
                 handle_to_index,
                 zone_to_variables,
                 zone_index,

                 edge_index,
                 edge_attr,

                 num_features,
                 encoder_hidden_dim,
                 gcn_hidden_dim,
                 q_net_hidden_dim,
                 action_dim):
        super(ThermoGRL, self).__init__()

        self.handle_to_index = handle_to_index
        self.zone_to_variables = zone_to_variables
        self.zone_index = zone_index
        self.num_features = num_features
        self.encoder_hidden_dim = encoder_hidden_dim
        self.gcn_hidden_ddim = gcn_hidden_dim
        self.q_net_hidden_dim = q_net_hidden_dim
        self.action_dim = action_dim

        self.vec_encoder = Encoder(num_features, encoder_hidden_dim)
        self.graph_encoder = Graph_Encoder(encoder_hidden_dim, gcn_hidden_dim, edge_index=edge_index, edge_attr=edge_attr)
        self.q_net = Q_Net(gcn_hidden_dim, q_net_hidden_dim, action_dim)

    def forward(self, eplus_obs_vec):
        h1 = self.vec_encoder(eplus_obs_vec)
        h2 = self.graph_encoder(h1)
        q = self.q_net(h2)
        return q

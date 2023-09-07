
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


def graph_actuation_time_series(i_episode,
                                outdoor_temperature,
                                price_signal,
                                core_zn_actuation,
                                perimeter_zn_1_actuation,
                                perimeter_zn_2_actuation,
                                perimeter_zn_3_actuation,
                                perimeter_zn_4_actuation,
                                core_zn_indoor_temperature,
                                perimeter_zn_1_indoor_temperature,
                                perimeter_zn_2_indoor_temperature,
                                perimeter_zn_3_indoor_temperature,
                                perimeter_zn_4_indoor_temperature,
                                epsilon,
                                graph=True,):
    start = 100
    end = 1300
    x = list(range(end - start))

    price_signal = [20] * (end - start)

    outdoor_temperature = outdoor_temperature[start:end]
    price_signal = price_signal[start:end]
    core_zn_actuation = core_zn_actuation[start:end]
    perimeter_zn_1_actuation = perimeter_zn_1_actuation[start:end]
    perimeter_zn_2_actuation = perimeter_zn_2_actuation[start:end]
    perimeter_zn_3_actuation = perimeter_zn_3_actuation[start:end]
    perimeter_zn_4_actuation = perimeter_zn_4_actuation[start:end]
    core_zn_indoor_temperature = core_zn_indoor_temperature[start:end]
    perimeter_zn_1_indoor_temperature = perimeter_zn_1_indoor_temperature[start:end]
    perimeter_zn_2_indoor_temperature = perimeter_zn_2_indoor_temperature[start:end]
    perimeter_zn_3_indoor_temperature = perimeter_zn_3_indoor_temperature[start:end]
    perimeter_zn_4_indoor_temperature = perimeter_zn_4_indoor_temperature[start:end]

    figs, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    axs[0].plot(x, outdoor_temperature, label='Outdoor temperature', color='green')
    axs[0].plot(x, [20] * len(x), label='Price signal', color='black')
    axs[0].plot(x, core_zn_actuation, label='Cooling Setpoint', color='blue')
    axs[0].plot(x, core_zn_indoor_temperature, label='Indoor temperature', color='red')
    axs[0].set_title('Core_ZN')
    axs[0].legend()

    axs[1].plot(x, outdoor_temperature, label='Outdoor temperature', color='green')
    axs[1].plot(x, [20] * len(x), label='Price signal', color='black')
    axs[1].plot(x, perimeter_zn_1_actuation, label='Cooling Setpoint', color='blue')
    axs[1].plot(x, perimeter_zn_1_indoor_temperature, label='Indoor temperature', color='red')
    axs[1].set_title('Perimeter_ZN_1')
    axs[1].legend()

    axs[2].plot(x, outdoor_temperature, label='Outdoor temperature', color='green')
    axs[2].plot(x, [20] * len(x), label='Price signal', color='black')
    axs[2].plot(x, perimeter_zn_2_actuation, label='Cooling Setpoint', color='blue')
    axs[2].plot(x, perimeter_zn_2_indoor_temperature, label='Indoor temperature', color='red')
    axs[2].set_title('Perimeter_ZN_2')
    axs[2].legend()

    axs[3].plot(x, outdoor_temperature, label='Outdoor temperature', color='green')
    axs[3].plot(x, [20] * len(x), label='Price signal', color='black')
    axs[3].plot(x, perimeter_zn_3_actuation, label='Cooling Setpoint', color='blue')
    axs[3].plot(x, perimeter_zn_3_indoor_temperature, label='Indoor temperature', color='red')
    axs[3].set_title('Perimeter_ZN_3')
    axs[3].legend()

    axs[4].plot(x, outdoor_temperature, label='Outdoor temperature', color='green')
    axs[4].plot(x, [20] * len(x), label='Price signal', color='black')
    axs[4].plot(x, perimeter_zn_4_actuation, label='Cooling Setpoint', color='blue')
    axs[4].plot(x, perimeter_zn_4_indoor_temperature, label='Indoor temperature', color='red')
    axs[4].set_title('Perimeter_ZN_4')
    axs[4].legend()

    plt.suptitle('I_EPISODE:' + str(i_episode) + ' EPSILON:' + str(epsilon), fontsize=16)

    plt.tight_layout()

    plt.savefig('./logs/save.png')
    if graph and (i_episode % args.print_every == 0) and i_episode != 0:
        plt.show()






# Not all of the arguments are used
parser = argparse.ArgumentParser(description="")
parser.add_argument("-idf", type=str, default="./in.idf", help="IDF file location")
parser.add_argument("-eplus_weather_file", type=str, default='./weather.epw', help="Weather file")
parser.add_argument("-ep", type=int, default=100, help="The amount of training episodes, default is 100")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=1e-5, help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float,default=0.1, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-encoder_hidden", type=int, default=256, help="Dimension of the hidden representation encoded by the MLP encoder layer")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-gcn_hidden", type=int, default=256, help="Dimension of the hidden representation of the GCN layer")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(5e4), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=2, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("--save_every", type=int, default=2, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-n_epoch", type=int, default=5, help="Epoch size, default is 5")
parser.add_argument("-t", "--tau", type=float, default=0.96, help="Softupdate factor tau, default is 1e-2")
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
    'var-environment-time-day_of_week': 13,
    'var-environment-cost_rate': 14,
}

ZONE_TO_VARIABLES = {
    'Outdoors': ['var_environment_site_outdoor_air_drybulb_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
    'Attic': ['var-attic_indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
    'Core_ZN': ['var-core_zn_indoor_temperature', 0, 0, 0, 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
    'Perimeter_ZN_1': ['var-perimeter_zn_1-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
    'Perimeter_ZN_2': ['var-perimeter_zn_2-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
    'Perimeter_ZN_3': ['var-perimeter_zn_3-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
    'Perimeter_ZN_4': ['var-perimeter_zn_4-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
    'Perimeter_ZN_4': ['var-perimeter_zn_4-indoor_temperature', 'var_environment_site_direct_solar_radiation_rate_per_area', 'var_environment_site_diffuse_solar_radiation_rate_per_area', 'var_environment_site_horizontal_infrared_radiation_rate_per_area', 'var-environment-time-month', 'var-environment-time-day', 'var-environment-time-hour', 'var-environment-time-day_of_week', 'var-environment-cost_rate'],
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

    num_features = len(ZONE_TO_VARIABLES[list(ZONE_TO_VARIABLES.keys())[0]])
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
    n_episode = args.ep + 1
    total_steps = 0
    epsilon = args.epsilon



    while i_episode < n_episode:
        outdoor_temperature = []
        price_signal = []
        core_zn_actuation = []
        perimeter_zn_1_actuation = []
        perimeter_zn_2_actuation = []
        perimeter_zn_3_actuation = []
        perimeter_zn_4_actuation = []
        core_zn_indoor_temperature = []
        perimeter_zn_1_indoor_temperature = []
        perimeter_zn_2_indoor_temperature = []
        perimeter_zn_3_indoor_temperature = []
        perimeter_zn_4_indoor_temperature = []

        if i_episode > 1:
            epsilon -= 0.015
            if epsilon < 0.025:
                epsilon = 0.025

        i_episode += 1

        done = False
        state = env.reset()

        score = 0

        while not done:
            total_steps += 1

            action = []
            q = model(state)
            for agent_index in agents_index:
                if np.random.rand() < epsilon:
                    a = np.random.choice(action_dim)
                else:
                    a = np.argmax(q[agent_index].cpu().detach().numpy())

                #action[INDEX_TO_ZONE[agent_index]] = a
                action.append(a)




            next_state, reward, done, truncated, info = env.step(action=action)
            buff.add(state, EDGE_INDEX, action, reward, next_state, EDGE_INDEX, done)

            score += -1 * sum(reward)


            # for graphing time series
            cooling_actuators = env.retrieve_actuators()[0]
            outdoor_temperature.append(state[6])
            price_signal.append(info['cost_signal'])
            core_zn_actuation.append(cooling_actuators[0])
            perimeter_zn_1_actuation.append(cooling_actuators[1])
            perimeter_zn_2_actuation.append(cooling_actuators[2])
            perimeter_zn_3_actuation.append(cooling_actuators[3])
            perimeter_zn_4_actuation.append(cooling_actuators[4])
            core_zn_indoor_temperature.append(state[1])
            perimeter_zn_1_indoor_temperature.append(state[2])
            perimeter_zn_2_indoor_temperature.append(state[3])
            perimeter_zn_3_indoor_temperature.append(state[4])
            perimeter_zn_4_indoor_temperature.append(state[5])

            state = next_state

        # done
        print('----------------------------------')
        print('I_EPISODE:', i_episode, 'SCORE:', score)
        print('----------------------------------')
        with open('./logs/logs.txt', 'a') as f:
            f.write(str(score) + '\n')

        if i_episode % args.save_every == 0 and i_episode != 0:
            # save current model
            print('##################')
            print('MODEL SAVE I_EPISODE:', i_episode)
            print('##################')
            torch.save({
                'episode': i_episode,
                'model_state_dict': model.state_dict(),
                'model_tar_state_dict': model_tar.state_dict(),
                'optimizer': optimizer.state_dict()
            }, './model/checkpoint.pt')

        graph_actuation_time_series(i_episode,
                                    outdoor_temperature,
                                    price_signal,
                                    core_zn_actuation,
                                    perimeter_zn_1_actuation,
                                    perimeter_zn_2_actuation,
                                    perimeter_zn_3_actuation,
                                    perimeter_zn_4_actuation,
                                    core_zn_indoor_temperature,
                                    perimeter_zn_1_indoor_temperature,
                                    perimeter_zn_2_indoor_temperature,
                                    perimeter_zn_3_indoor_temperature,
                                    perimeter_zn_4_indoor_temperature,
                                    epsilon,
                                    graph=False)


        if not buff.buffer_filled_percentage() >= 30:
            print('---------------------------')
            print('BUFFER FILLED PERCENTAGE:', buff.buffer_filled_percentage())
            print('---------------------------')
            continue


        # epoch_losses = []
        # for epoch in range(args.n_epoch):
        #     states, edge_indices, actions, rewards, next_states, next_edge_indices, dones = buff.get_batch(args.batch_size)

        #     # generate mask
        #     num_rows = len(ZONE_INDEX)
        #     mask = torch.zeros(num_rows)
        #     mask[agents_index] = 1

        #     q_values = model(torch.Tensor(states))
        #     target_q_values = model_tar(torch.Tensor(next_states))
        #     target_q_values = target_q_values - 9e15 * mask.unsqueeze(1)
        #     target_q_values = target_q_values.max(dim=2)[0]
        #     target_q_values = np.array(target_q_values.cpu().data)
        #     expected_q = np.array(q_values.cpu().data)
        #     for j in range(args.batch_size):
        #         for i in range(num_agents):
        #             expected_q[j][i][actions[j][i]] = rewards[j] + (1-dones[j]) * args.gamma * target_q_values[j][i]

        #     loss = (q_values - torch.tensor(expected_q)).pow(2).mean()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     epoch_losses.append(loss.item())

        #     with torch.no_grad():
        #         for p, p_targ in zip(model.parameters(), model_tar.parameters()):
        #             p_targ.data.mul_(args.tau)
        #             p_targ.data.add_((1 - args.tau) * p.data)

        # with open('./logs/loss.txt', 'a') as f:
        #     f.write(str(np.mean(epoch_losses))
        # + '\n')


        epoch_losses = []
        for epoch in range(args.n_epoch):
            states, edge_indices, actions, rewards, next_states, next_edge_indices, dones = buff.get_batch(args.batch_size)

            batch_loss = []
            batch_grads = []
            for i in range(args.batch_size):
                curr_state = states[i]
                curr_next_state = next_states[i]
                curr_edge_indices = edge_indices[i]
                curr_rewards = rewards[i]
                curr_actions = actions[i]
                curr_dones = dones[i]

                q_values = model(torch.tensor(curr_state)).detach()
                target_q_values = model_tar(torch.tensor(curr_next_state)).detach()
                masked_target_q_values = target_q_values[agents_index]
                target_q_values = np.array(masked_target_q_values.cpu().data)
                expected_q = np.array(q_values.cpu().data)

                for j in range(len(agents_index)):
                    expected_q[j][curr_actions[j]] = curr_rewards[j] + (1 - curr_dones) * args.gamma * target_q_values[j][curr_actions[j]]

                #print('temp batch _loss:', (q_values - torch.tensor(expected_q)).pow(2), type((q_values - torch.tensor(expected_q, dtype=torch.float32)).pow(2)), (q_values - torch.tensor(expected_q, dtype=torch.float32)).pow(2).shape)
                temp_batch_loss = torch.mean((q_values - torch.tensor(expected_q, dtype=torch.float32)).pow(2))

                temp_batch_loss.requires_grad = True

                temp_batch_loss.backward()

                batch_loss.append(temp_batch_loss)
                batch_grads.append(temp_batch_loss.grad)


            loss = torch.mean(torch.stack(batch_loss))
            average_gradient = torch.mean(torch.stack(batch_grads), dim=0)
            optimizer.zero_grad()
            loss.backward(gradient=average_gradient)
            optimizer.step()

            epoch_losses.append(loss.item())
            # loss = torch.mean(torch.tensor(batch_loss))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        epoch_loss = sum(epoch_losses) / args.n_epoch
        print('I_EPISODE:', i_episode, 'EPOCH LOSS:', epoch_loss)
        with open('./logs/loss.txt', 'a') as f:
            f.write(str(epoch_loss) + '\n')


if __name__ == "__main__":
    main()

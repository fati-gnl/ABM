import json
import os
from collections import defaultdict
import time
from itertools import product, combinations

import numpy as np
from matplotlib import pyplot as plt

from model import Corruption


class Experiment():

    def __init__(self,
                 data_params,
                 default_params,
                 max_steps,
                 cop_params,
                 cit_params,
                 folder):

        self.data_params = data_params
        self.default_params = default_params
        self.max_steps = max_steps
        self.cop_params = cop_params
        self.cit_params = cit_params
        self.folder = folder

        self.pair_combinations = list(
            combinations(self.data_params.keys(), 2))  # 10 unique combinations between parameters

        # self.create_data()

        # cit_data = self.get_cit_data()
        # self.visualize(cit_data)

    def create_data(self):
        for param1, param2 in self.pair_combinations:
            range_param1, range_param2 = self.data_params[param1], self.data_params[param2]
            data_combinations = list(product(range_param1, range_param2))  # 64 combinations for parameters

            for setting_param1, setting_param2 in data_combinations:
                test_params = self.default_params.copy()
                test_params[param1], test_params[param2] = setting_param1, setting_param2
                model = Corruption(test_params= {param1: setting_param1, param2: setting_param2}, *test_params.values())
                start = time.time()
                for _ in range(self.max_steps):
                    model.step()
                end = time.time()
                print("Time elapsed [s]: ",end - start)

    def write_data(self, dictionary, folder, path):
        with open(folder + "/" + path, 'w') as f:
            json.dump(dictionary, f)

    def get_cop_data(self):

        os.makedirs("/content/data", exist_ok=True)

        self.cop_data = {}

        for file_path in os.listdir(self.folder):

            with open("/content/results/" + file_path) as f:
                file = np.array([json.loads(line) for line in f])

            file_path = file_path.split('.json')[0]
            self.cop_data[file_path] = defaultdict(list)

            print(len(file))

            for iteration in range(len(file)):
                iteration_name = 'iteration_' + str(iteration)
                iteration_data = file[iteration][iteration_name]

                nested = {k: v for k, v in iteration_data.items() if isinstance(v, dict)}

                for param in self.cop_params:

                    if param == 'action':
                        iter_freq_bribe = sum([1 for cop in nested['cops'] if nested['cops'][cop][param] == 'bribe'])
                        self.cop_data[file_path]['bribe_frequency'].append(iter_freq_bribe)

                    if param == 'time_left_in_jail':
                        iter_in_jail = sum([1 for cop in nested['cops'] if nested['cops'][cop][param] > 0])
                        self.cop_data[file_path]['in_jail'].append(iter_in_jail)

                    if param == 'estimated_prob_accept':
                        iter_prob_accept = [nested['cops'][cop][param] for cop in nested['cops']]
                        self.cop_data[file_path]['prob_accept'].append(np.mean(iter_prob_accept))

                    if param == 'approximated_prob_caught':
                        iter_prob_caught = [nested['cops'][cop][param] for cop in nested['cops']]
                        self.cop_data[file_path]['prob_caught'].append(np.mean(iter_prob_caught))

        self.write_data(self.cop_data, "/content/data", "cop_data")

        return self.cop_data

    def visualize_array(self, data, agent_param, iteration):

        for param1, param2 in self.pair_combinations:
            range_param1, range_param2 = self.data_params[param1], self.data_params[param2]
            data_combinations = list(product(range_param1, range_param2))
            z = np.zeros((len(range_param1), len(range_param2)))

            for idx1, setting_param1 in enumerate(range_param1):
                setting1 = param1 + "_" + str(setting_param1)
                for idx2, setting_param2 in enumerate(range_param2):
                    setting2 = param2 + "_" + str(setting_param2)
                    experiment_name = setting1 + "-" + setting2

                    if iteration == 'mean':
                        z[idx1, idx2] = round(np.mean(data[experiment_name][agent_param]), 2)

                    if iteration == 'median':
                        z[idx1, idx2] = round(np.median(data[experiment_name][agent_param]), 2)

                    if isinstance(iteration, int):
                        z[idx1, idx2] = round(data[experiment_name][agent_param][iteration], 2)

                    if isinstance(iteration, list):
                        z[idx1, idx2] = round(np.mean(data[experiment_name][agent_param][iteration[0]: iteration[1]]),
                                              2)

            fig, ax = plt.subplots()
            im = ax.imshow(z)

            ax.set_xticks(np.arange(len(range_param2)))
            ax.set_yticks(np.arange(len(range_param1)))
            ax.set_xticklabels([round(item, 2) for item in range_param2])
            ax.set_yticklabels([round(item, 2) for item in range_param1])

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(len(range_param1)):
                for j in range(len(range_param2)):
                    text = ax.text(j, i, z[i, j],
                                   ha="center", va="center", color="b")

            ax.set_title(agent_param)
            ax.set_xlabel(param2)
            ax.set_ylabel(param1)

            fig.tight_layout()
            plt.show()

if __name__ == '__main__':

    data_params = {"team_size": [1, 2, 5, 10, 20, 25, 50, 100],
                   "corruption_among_teams_spread": np.linspace(0.01, 0.99, num=8).tolist(),
                   "jail_time": np.arange(1, 9).tolist(),
                   "prob_of_prosecution": np.linspace(0.01, 0.99, num=8).tolist(),
                   "cost_complain": np.linspace(0.01, 15, num=8).tolist()}

    data_params_T = {"jail_time": np.arange(1, 5).tolist(),
                     "prob_of_prosecution": np.linspace(0.01, 0.99, num=4).tolist()}

    default_params = {"num_citizens": 1000,
                      "num_cops": 100,
                      "team_size": 10,
                      "rationality_of_agents": 10,
                      "jail_time": 4,
                      "prob_of_prosecution": 0.7,
                      "memory_size": 10,
                      "fine_amount": 1.,
                      "cost_complain": 0.4,
                      "penalty_citizen_prosecution": 0.,
                      "jail_cost_factor": 0.5,
                      "cost_accept_mean_std": (0.2, 0.05),
                      "citizen_complain_memory_discount_factor": 0.5,
                      "bribe_amount": 0.5,
                      "moral_commitment_mean_std": (0.25, 0.1),
                      "initial_time_left_in_jail": 0,
                      "initial_indifferent_corruption_honest_rate": (1.0, 0.0, 0.0),
                      "corruption_among_teams_spread": 1.0,
                      "logger": True}

    cop_params = ['action', 'time_left_in_jail', 'estimated_prob_accept', 'approximated_prob_caught']
    cit_params = ['action', 'complain_memory']

    experiment = Experiment(data_params, default_params, 200, cop_params, cit_params, '/content/results')
    experiment.create_data()
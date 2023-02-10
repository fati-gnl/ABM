from model import *
from agents import *
import random
from mesa.batchrunner import BatchRunner, FixedBatchRunner
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
plt.style.use('ggplot')


# Local Sensitivity Analysis

replicates = 20
max_steps = 140
distinct_samples = 15

problem = {
    'num_vars': 10,
    'names': ['team_size', 'rationality_of_agents', 'jail_time', 'prob_of_prosecution', 'memory_size',
              'cost_complain', 'penalty_citizen_prosecution',
              'jail_cost_factor','citizen_complain_memory_discount_factor', 'bribe_amount'],
    'bounds': [[5, 25], [0.01, 1.99], [1, replicates], [0.01, 0.99], [1, 16], [0.01, 15], [0, 20], [0.1, 10], [0.01, 0.99], [0, 2]]
}

integer_vars = ['team_size', 'jail_time', 'memory_size']
# Set the repetitions, the amount of steps, and the amount of distinct values per variable


# Set the outputs
model_reporters = { "Bribing": lambda m: sum([1 for cop in m.schedule_Cop.agents if cop.action == cop.possible_actions(0)]),
             "NoBribing": lambda m: sum([1 for cop in m.schedule_Cop.agents if cop.action == cop.possible_actions(1)])}

data = {}

for i, var in enumerate(problem['names']):
    # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
    samples = np.linspace(*problem['bounds'][i], num=distinct_samples).tolist()
    # Some parameters should have integer values. We change
    # the code to acommodate for this and sample only integers.
    if var in integer_vars[1:]:
        samples = np.linspace(*problem['bounds'][i], num=distinct_samples, dtype=int).tolist()
    if var=='team_size':
        samples = [1, 2, 5, 10, 20, 25, 50, 100]

    # batch = BatchRunner(Corruption,
    #                     max_steps=max_steps,
    #                     iterations=replicates,
    #                     variable_parameters={var: samples},
    #                     model_reporters=model_reporters,
    #                     display_progress=True)
    #
    # batch.run_all()
    # data[var] = batch.get_model_vars_dataframe()


def plot_param_var_conf(df, var, param, i):
    """
    Helper function for plot_all_vars. Plots the individual parameter vs
    variables passed.
    Args:
        ax: the axis to plot to
        df: dataframe that holds the data to be plotted
        var: variables to be taken from the dataframe
        param: which output variable to plot
    """
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]

    replicates = df.groupby(var)[param].count()
    err = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)

    plt.plot(x, y, c='k')
    plt.plot(x, y, '.', color='blue')
    plt.fill_between(x, y - err, y + err, alpha=0.5)

    plt.xlabel(var)
    plt.ylabel(param)

def plot_all_vars(df, param):
    """
    Plots the parameters passed vs each of the output variables.
    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """

    for i, var in enumerate(problem['names']):
        plt.figure(figsize=(6.5,3.5))
        plot_param_var_conf(data[var], var, param, i)

# Visualise the results of Local Sensitivity Analysis

# for param in ('Bribing', 'NoBribing'):
#     plot_all_vars(data, param)
#     plt.show()

# RUNNING THE MODEL USING BASELINE VALUES MULTIPLE TIMES TO GET THE DISTRIBUTION OF THE OUTPUTS
def model_baseline_output(team_size, rationality_of_agents, jail_time, prob_of_prosecution, memory_size,
                          cost_complain, penalty_citizen_prosecution, jail_cost_factor,
                          citizen_complain_memory_discount_factor, bribe_amount, max_steps, model_reporters):


    batch_fixed = FixedBatchRunner(Corruption,
                                   parameters_list=[
                                       {'team_size': team_size, 'rationality_of_agents': rationality_of_agents,
                                        'jail_time': jail_time,
                                        'prob_of_prosecution': prob_of_prosecution,
                                        'memory_size':memory_size,'cost_complain':cost_complain,
                                        'penalty_citizen_prosecution':penalty_citizen_prosecution,
                                        'jail_cost_factor':jail_cost_factor,
                                        'citizen_complain_memory_discount_factor':citizen_complain_memory_discount_factor,
                                        'bribe_amount':bribe_amount,
                                        'logger': False}],
                                   iterations=250,
                                   max_steps=max_steps,
                                   model_reporters=model_reporters)
    batch_fixed.run_all()

    data_fixed = batch_fixed.get_model_vars_dataframe()
    amount_bribe = data_fixed["Bribing"].values
    amount_nobribe = data_fixed["NoBribing"].values

    def plot_dist(data, data_name):
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        # Plot the histogram
        ax.hist(
        data,
        bins=25,
        density=True,
        label="Histogram from samples",
        zorder=5,
        edgecolor="k",
        alpha=0.5,
        )

        kde = sm.nonparametric.KDEUnivariate(data)

        kde.fit()  # Estimate the densities

        # Plot the KDE as fitted using the default arguments
        ax.plot(kde.support, kde.density, lw=3, label="KDE from samples", zorder=10)

        ax.set_ylabel('Density')
        ax.set_xlabel(data_name)
        ax.legend(loc="best")
        ax.grid(True, zorder=-5)
        plt.show()

    def qq_plot(data, data_name):
        sm.qqplot(data, line='s',marker='.', markerfacecolor='k', markeredgecolor='k', alpha=0.3)
        plt.title(data_name)
        plt.show()

# Create the QQ-plots to check the output data for normality.
    for data, label in [(amount_bribe,"amount_bribe"),(amount_nobribe, "amount_nobribe")]:
        plot_dist(data, label)
        qq_plot(data, label)
#Run the baseline model by running the function above
# model_baseline_output(team_size=10, rationality_of_agents=0.75, jail_time=4, prob_of_prosecution=0.7,
#                       memory_size=10,  cost_complain=3, penalty_citizen_prosecution=5, jail_cost_factor=5,
#                       citizen_complain_memory_discount_factor=3, bribe_amount=50, max_steps=max_steps,
#                       model_reporters=model_reporters)

#Global Sensitivity Analysis
# replicates_global = 20
# max_steps_global = 140
# distinct_samples_global = 20
#
# # We get all our samples here
# param_values = saltelli.sample(problem, distinct_samples_global, calc_second_order = False)
# items = [1, 2, 5, 10, 20, 25, 50, 100]
# for values in param_values:
#     values[0] =random.choice(items)
#
# print(param_values)
# batch_global = BatchRunner(Corruption,
#                            max_steps=max_steps_global,
#                            variable_parameters={name:[] for name in problem['names']},
#                            model_reporters=model_reporters)
#
# count = 0
# data_global = pd.DataFrame(index=range(replicates_global*len(param_values)),
#                            columns= problem['names'])
# data_global['Run'], data_global['Bribe'], data_global['NoBribe'] = None, None, None
#
# for i in range(replicates_global):
#     for vals in param_values:
#         # Change parameters that should be integers
#         vals = list(vals)
#         vals[0] = int(vals[0])
#         vals[2] = int(vals[2])
#         vals[4] = int(vals[4])
#         # Transform to dict with parameter names and their values
#         variable_parameters = {}
#         for name, val in zip(problem['names'], vals):
#             variable_parameters[name] = val
#
#         batch_global.run_iteration(variable_parameters, tuple(vals), count)
#         iteration_data = batch_global.get_model_vars_dataframe().iloc[count]
#         iteration_data['Run'] = count
#         data_global.iloc[count, 0:10] = vals
#         data_global.iloc[count, 10:13] = iteration_data
#         count += 1
#
#         print(f'{count / (len(param_values) * (replicates_global)) * 100:.2f}% done')

# Si_bribe = sobol.analyze(problem, data_global['Bribe'].values, calc_second_order = False, print_to_console=True)
# Si_nobribe = sobol.analyze(problem, data_global['NoBribe'].values, calc_second_order = False, print_to_console=True)

def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """


    indices = s['S' + i]
    errors = s['S' + i + '_conf']
    plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')


# for Si in (Si_bribe, Si_nobribe):
#     # First order
#     plot_index(Si, problem['names'], '1', 'First order sensitivity')
#     plt.show()
#
#     # Total order
#     plot_index(Si, problem['names'], 'T', 'Total order sensitivity')
#     plt.show()

import copy
from scipy import stats

# Implementing the PAWN Sensitivity Analysis Technique
#
# # At first, we sample values for every parameter, and run the model.
# # Nu is the number of random samples of inputs 'X'.
Nu = 160
# Nc is the number of the random samples of inputs 'X~i'
Nc = 130
# # M is the number of random samples to be used as conditioning values for 'Xi'.
M = 15


def PAWN_implementation(Nu, Nc, M, problem):
    integer_vars = ['team_size', 'jail_time', 'memory_size']
    Nu_samples = {}
    Nc_samples = {}
    M_samples = {}
    #reset = {}
    bribe_amount = []
    F_y = []
    items_division = [1, 2, 5, 10, 20, 25, 50, 100]
    ks_list = []
    ks_parameters_total = {}
    ks_parameters_lists = {}

    def ecdf(a):
        """
        Function that takes as input argument a list of values,
        and generates as output a list that represents
        the Empirical Cumulative Distribution Function(ECDF)
        of the given initial list.
        """
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1]

    def plot_ecdf(data_dict, unconditional, xlabel):
        """
        Function that takes as input argument a dictionary of values.
        That dictionary will be used to plot the conditional CDFs, while we also
        give as input argument the unconditional CDF.

        The function plots these ECDFs in one figure, per parameter.
        """
        i = 0
        for label, data in data_dict.items():
            i += 1
            # Compute the ECDF of the data.
            x, y = ecdf(data)
            x = np.insert(x, 0, x[0])
            y = np.insert(y, 0, 0.)
            # Plot the computed conditional ECDF
            if i == 1:
                plt.plot(x, y, drawstyle='steps-post', color='black', label='Conditional CDFs')
            else:
                plt.plot(x, y, drawstyle='steps-post', color='black')
        # Plot the unconditional ECDF
        x1, y1 = ecdf(unconditional)
        x1 = np.insert(x, 0, x[0])
        y1 = np.insert(y, 0, 0.)
        plt.plot(x1, y1, drawstyle='steps-post', label='Unconditional CDF', color='red')
        plt.grid(True)
        plt.legend()
        plt.xlabel(xlabel)
        plt.title('Conditional and Unconditional CDFs')
        plt.show()

    # Generate Nu random samples of inputs and evaluate the model.
    # The evaluation of the model results in the unconditional distribution of the model ouputs, F(y)
    for i, var in enumerate(problem['names']):
        Nu_samples[var] = np.random.uniform(*problem['bounds'][i], Nu).tolist()
        Nc_samples[var] = np.random.uniform(*problem['bounds'][i], Nc).tolist()
        M_samples[var] = np.random.uniform(*problem['bounds'][i], M).tolist()
        if var in integer_vars[1:]:
            Nu_samples[var] = np.random.randint(*problem['bounds'][i], Nu).tolist()
            Nc_samples[var] = np.random.randint(*problem['bounds'][i], Nc).tolist()
            M_samples[var] = np.random.randint(*problem['bounds'][i], M).tolist()
        if var == 'team_size':
            Nu_samples[var] = random.choices(items_division, k=Nu)
            Nc_samples[var] = random.choices(items_division,k= Nc)
            M_samples[var] = random.choices(items_division,k= M)


    # Evaluate the model for all these samples.
    # This results to the Unconditional CDF, F(y) - everything is sampled
    for i in range(Nu):
        batch_fy = FixedBatchRunner(Corruption,
                                    parameters_list=[{'team_size': Nu_samples['team_size'][i],
                                                      'rationality_of_agents': Nu_samples['rationality_of_agents'][i],
                                                      'jail_time': Nu_samples['jail_time'][i],
                                                      'prob_of_prosecution': Nu_samples['prob_of_prosecution'][i],
                                                      'memory_size': Nu_samples['memory_size'][i],
                                                      'cost_complain': Nu_samples['cost_complain'][i],
                                                      'penalty_citizen_prosecution': Nu_samples['penalty_citizen_prosecution'][i],
                                                      'jail_cost_factor': Nu_samples['jail_cost_factor'][i],
                                                      'citizen_complain_memory_discount_factor': Nu_samples['citizen_complain_memory_discount_factor'][i],
                                                      'bribe_amount': Nu_samples['bribe_amount'][i]}],
                                    iterations=1,
                                    max_steps=max_steps,
                                    model_reporters=model_reporters)
        batch_fy.run_all()
        intermed = batch_fy.get_model_vars_dataframe()
        F_y.append(intermed["Bribing"].values[0])

    # Create the data for the conditional CDFs
    testi = {}
    for i, var in enumerate(problem['names']):
        testi[f'Variable {var}'] = {}
        reset = copy.deepcopy(Nc_samples)
        for j in range(M):
            reset[var] = np.ones(len(reset[var])) * M_samples[var][j]
            testi[f'Variable {var}'][f'Conditioning value nr {j}'] = {}
            # Evaluate the model.
            # By doing that we get the model outputs, when a parameter is fixed and the others are sampled.
            # Hence we can form the conditional CDFs.
            for k in range(len(reset[var])):
                batch_fixed = FixedBatchRunner(Corruption,
                                               parameters_list=[{'team_size': int(reset['team_size'][k]),
                                                                 'rationality_of_agents':
                                                                     reset['rationality_of_agents'][k],
                                                                 'jail_time': reset['jail_time'][k],
                                                                 'prob_of_prosecution':
                                                                     reset['prob_of_prosecution'][k],
                                                                 'memory_size': reset['memory_size'][k],
                                                                 'cost_complain': reset['cost_complain'][k],
                                                                 'penalty_citizen_prosecution':
                                                                     reset['penalty_citizen_prosecution'][k],
                                                                 'jail_cost_factor': reset['jail_cost_factor'][k],
                                                                 'citizen_complain_memory_discount_factor': reset[
                                                                     'citizen_complain_memory_discount_factor'][k],
                                                                 'bribe_amount': reset['bribe_amount'][k],
                                                                 'logger': False}],
                                               iterations=1,
                                               max_steps=max_steps,
                                               model_reporters=model_reporters,
                                               display_progress=False)
                batch_fixed.run_all()
                data_fixed = batch_fixed.get_model_vars_dataframe()
                bribe_amount.append(data_fixed["Bribing"].values[0])
                if k == (len(reset[var]) - 1):
                    testi[f'Variable {var}'][f'Conditioning value nr {j}'] = bribe_amount
                    bribe_amount = []
            # Fill in the dictionaries needed for visualisation of the results
            ks_stat = stats.ks_2samp(F_y, testi[f'Variable {var}'][f'Conditioning value nr {j}']).statistic
            ks_list.append(ks_stat)
            if j == len(range(M - 1)):
                ks_parameters_total[f'T_i of variable {var}'] = np.mean(ks_list)
                ks_parameters_lists[f'List of variable {var}'] = ks_list
                ks_list = []
        # Plot the KS statistics and the 'Ti'- PAWN index per parameter.
        plt.figure()
        plt.title('KS statistics and T_i index(PAWN Index)')
        plt.plot(ks_parameters_lists[f'List of variable {var}'], 'o', label='KS statistics for each conditioning value')
        plt.plot(ks_parameters_lists[f'List of variable {var}'])
        plt.xlabel(var)
        plt.axhline(ks_parameters_total[f'T_i of variable {var}'], label='mean of KS (index T_i)', color='black')
        plt.ylabel('KS Statistic')
        plt.legend()
        plt.ylim(0, 1)
        plt.show()
        # Plot the conditional and unconditional CDFs
        plt.figure()
        plot_ecdf(testi[f'Variable {var}'], F_y, var)
    plt.figure()
    a = ks_parameters_total['T_i of variable team_size']
    b = ks_parameters_total['T_i of variable rationality_of_agents']
    c = ks_parameters_total['T_i of variable jail_time']
    d = ks_parameters_total['T_i of variable prob_of_prosecution']
    e = ks_parameters_total['T_i of variable memory_size']
    f = ks_parameters_total['T_i of variable cost_complain']
    g = ks_parameters_total['T_i of variable penalty_citizen_prosecution']
    h = ks_parameters_total['T_i of variable jail_cost_factor']
    i = ks_parameters_total['T_i of variable citizen_complain_memory_discount_factor']
    j = ks_parameters_total['T_i of variable bribe_amount']
    plt.bar(problem['names'], [a, b, c, d, e, f, g, h, i, j])
    plt.show()

PAWN_implementation(Nu, Nc, M, problem)

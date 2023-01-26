# We define our variables and bounds
from model import *
from agents import *
from mesa.batchrunner import BatchRunner
import numpy as np
import matplotlib.pyplot as plt

problem = {
    'num_vars': 6,
    'names': ['prob_prosecution', 'cost_of_complaining', 'cost_of_silence', 'reward_citizen', 'penalty_citizen',
              'penalty_cop'],
    'bounds': [[0.01, 1], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100]]
}

# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 2
max_steps = 130
distinct_samples = 10

# Set the outputs
model_reporters = { "Bribing": lambda m: sum([1 for cop in m.schedule.agents if type(cop) == Cop and cop.action == "bribe"])/m.number_of_citizens,
             "NoBribing": lambda m: sum([1 for cop in m.schedule.agents if type(cop) == Cop and cop.action == "not_bribe"])/m.number_of_citizens}

data = {}

for i, var in enumerate(problem['names']):
    # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
    samples = np.linspace(*problem['bounds'][i], num=distinct_samples)

    # Keep in mind that wolf_gain_from_food should be integers. You will have to change
    # your code to acommodate for this or sample in such a way that you only get integers.
    #if var == 'wolf_gain_from_food':
     #   samples = np.linspace(*problem['bounds'][i], num=distinct_samples, dtype=int)

    batch = BatchRunner(CopCitizen,
                        max_steps=max_steps,
                        iterations=replicates,
                        variable_parameters={var: samples},
                        model_reporters=model_reporters,
                        display_progress=True)

    batch.run_all()

    data[var] = batch.get_model_vars_dataframe()


def plot_param_var_conf(ax, df, var, param, i):
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

    ax.plot(x, y, c='k')
    ax.fill_between(x, y - err, y + err)

    ax.set_xlabel(var)
    ax.set_ylabel(param)


def plot_all_vars(df, param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """

    f, axs = plt.subplots(6, figsize=(7, 10))
    plt.subplots_adjust(hspace=0.9)
    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param, i)


for param in ('Bribing', 'NoBribing'):
    plot_all_vars(data, param)
    plt.show()
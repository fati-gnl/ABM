This is an implementation of a agent-based modelling apporach to the corruption model. In this model there are two types
of agents: cops and agents. In this model only cops can offer bribe. Citizens can either accept silently, accept and
complain, reject the bribe and then complain or reject the bribe silently. Agents make actions based on estimated
utilities for each action.

#### Heterogeneity

Each citizen has different moral cost of being silent and accepting the bribe. They also have the memory of successful
complaints which is discounted, running average of complain event results.

Cops have to types of memory: accepting_bribe_memory and prob_caught. Former is the memory of how often did the citizens
accepted the bribe. The latter is the estimated by the rate of collegues in team being caught.

#### Social Network

Cops are connected through social network, which is imitated by teams. Each team has the same size and each cop belongs
to exactly one team. It is used only in estimating probability of prosecution.

#### Installation

#### How to run

Just run `server.py` file. It will open mesa board in your browser. You will see charts there of the simulation.

Regarding the Sensitivity Analysis, run the `sensitivity_analysis.py` file. This will produce the results and plots 
of the sensitivity analysis techniques used. 

In order to see the results of the conducted experiments, run the `run_experiments.py` file. You will see plots 
with the results of the experiments.

#### Files

- `ABM/agents.py`: This file defines the Citizen and Cop agent classes
- `ABM/model.py`: Defines the Corruption model itself
- `ABM/server.py`:  Sets up the interactive visualization server
- `ABM/utils.py`: Additional helping functions and classes
- `ABM/sensitivity_analysis.py`: Performs the sensitivity analysis techniques used in the model.
- `ABM/run_experiments.py`: Sets up the experiments to be conducted and visualises the results.



#### Logger

This project has it's json logger. The data is automatically stored in the `results/` folder and name of the file is
random name plus date. The json structure is as follows:

- `init_params`: stores all the parameters that are initialized at the beginning
    - `bribe_amount`
    - ...
- `iteration_step`: stores all the parameters that could change in each step
    - ... NOTE: The tricky part is that two files are saved, probably because of the server runs the model function
      again. The right file is the one that has more keys than just `init_params`. 

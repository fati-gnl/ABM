This is an implementation of a agent-based modelling apporach to the corruption model. In this model there are two types of agents: cops and agents.
In this model only cops can offer bribe. Citizens can either accept silently, accept and complain, reject the bribe and then complain or reject the bribe silently.
Agents make actions based on estimated utilities for each action. 

####  Heterogeneity
Each citizen has different moral cost of being silent and accepting the bribe. They also have the memory of successful complaints which is discounted, running average of complain event results.

Cops have to types of memory: accepting_bribe_memory and prob_caught. Former is the memory of how often did the citizens accepted the bribe. The latter is the estimated by the rate of collegues in team being caught.

#### Social Network
Cops are connected through social network, which is imitated by teams. Each team has the same size and each cop belongs to exactly one team. It is used only in estimating probability of prosecution.

#### How to run
Just run `server.py` file. It will open mesa board in your browser. You will see charts there of the simulation. 

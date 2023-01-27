import random

import axelrod as axl
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler

from agents import Citizen, Cop

class CopCitizen(Model):
    '''
    Corruption Model: Citizens and Cops
    '''

    def __init__(self, initial_citizens=5000, initial_cops=100, rationality_of_agents=1,
                 moral_commitment_mean_std=(0.3, 0.2), prob_prosecution=1, cost_of_complaining=0.1, cost_of_silence=0.1,
                 fine=1, penalty_citizen=0.1, penalty_cop=0.1, reward_citizen=0.1, bribe_amount=5, cost_of_accepting = 0.1, prob_succesful_complain = 1):
        '''
        :param initial_citizens:
        :param initial_cops:
        :param rationality_of_agents: 0 agents completely random, 1 completely rational
        '''
        super().__init__()

        # This should be 1 so other values are more or less normalized in respect to it
        self.fine = fine
        self.bribe_amount = bribe_amount
        self.cost_of_complaining = cost_of_complaining
        # TODO: Should this be different for each individual? Or dependent on an environment somehow?
        self.cost_of_silence = cost_of_silence
        self.penalty_citizen = penalty_citizen
        self.penalty_cop = penalty_cop
        self.reward_citizen = reward_citizen
        self.cost_of_accepting = cost_of_accepting
        self.prob_succesful_complain = prob_succesful_complain

        self.jail_time = 0

        self.initial_citizens = initial_citizens
        self.number_of_citizens = 1
        self.initial_cops = initial_cops
        self.lambda_ = rationality_of_agents

        # Create a scheduler for all agents
        self.schedule = BaseScheduler(self)

        # Data collector to be able to save the data - cop and citizen count
        # Todo: We should collect other data - maybe mean payoff or total payoff. For sure citizens actions
        # Mean payoffs, Action distribution, Citizen's complaints rate
        # Already implemented: bribing rate
        self.datacollector = DataCollector(
            {"Bribing": lambda m: sum([1 for cop in self.schedule.agents if type(cop) == Cop and cop.action == "bribe"])/self.number_of_citizens,
             "NoBribing": lambda m: sum([1 for cop in self.schedule.agents if type(cop) == Cop and cop.action == "not_bribe"])/self.number_of_citizens,
             "ComplainRate": lambda m: sum([1 for cit in self.schedule.agents if type(cit) == Citizen and (cit.action == "accept_and_complain" or cit.action == "reject_and_complain")])/self.number_of_citizens,
             "NoComplainRate": lambda m: sum([1 for cit in self.schedule.agents if type(cit) == Citizen and (cit.action == "accept_and_silent" or cit.action == "reject_and_silent")])/self.number_of_citizens,
             })

        # Create cops
        for i in range(self.initial_cops):
            cop = Cop(self.next_id(), self, self.lambda_, jail_time = 0, moral_commitment_mean_std=moral_commitment_mean_std)
            self.schedule.add(cop)

        # Create citizens
        for i in range(self.initial_citizens):
            citizen = Citizen(self.next_id(), self, self.lambda_)
            self.schedule.add(citizen)

        # Needed for the Batch run
        self.running = True
        # Needed for the datacollector
        self.datacollector.collect(self)

        # TODO: make it change depending on the environment
        # TODO: decide if should be moved to each agent individually
        self.prob_prosecution = 1

        # This should be 1 so other values are more or less normalized in respect to it
        self.fine = 1.

        # This I think should be here as it's more global?
        self.cost_of_complaining = 0
        # TODO: Should this be different for each individual? Or dependent on an environment somehow?
        self.cost_of_silence = 0.9
        # This is systematic, so I think global is good
        self.penalty_citizen = 0.1
        self.penalty_cop = 60.
        self.reward_citizen = 10

        # Create social groups or networks (groups of cops that will see the actions that their friends took in the past)
        self.cop_list = [agent for agent in self.schedule.agents if isinstance(agent, Cop)]
        self.network = self.social_connections(self.cop_list, 10)

    def step(self):
        '''
        Method that calls the step method for each of the citizens, and then for each of the cops.
        '''

        # Calculate if the cops in each group have been to prision or not
        n_prision_count = []
        for groups in self.network:
            prision_count = []
            for agent in groups:
                if (agent.jail_time > 0):
                    prision_count.append(1)
            n_prision_count.append(sum(prision_count))

        print(n_prision_count)

        # Create list of cops that will be used to keep track of which cops have already served a citizen in that step
        self.available_cops = [agent for agent in self.schedule.agents if isinstance(agent, Cop) and agent.jail_time == 0]

        # Number of citizens == number of cops
        self.number_of_citizens = len(list(filter(lambda a: type(a) == Cop and a.jail_time == 0, self.schedule.agents)))

        self.caught_citizens = random.choices([agent for agent in self.schedule.agents if isinstance(agent, Citizen)], k=self.number_of_citizens)

        self.schedule.step()

        # Save the statistics
        self.datacollector.collect(self)

        # Reduce jail_time
        for groups in self.network:
            for agent in groups:
                if agent.jail_time != 0:
                    agent.jail_time -= 1

    def get_cop(self):
        '''
        Gets a cop from the available cops, this cop is not available anymore.
        If all cops busy, return None
        :return: Cop object or None
        '''
        if len(self.available_cops) > 0:
            cop =  random.sample(self.available_cops, 1)[0]
            self.available_cops.remove(cop)
            return cop
        return None

    def social_connections(self, available_cops, group_size):
        '''
        Groups all the cops into smaller groups of cops (their social network)
        :return:
        '''
        social_groups = []
        for i in range(0, len(available_cops), group_size):
            social_groups.append(available_cops[i:i + group_size])
        return social_groups
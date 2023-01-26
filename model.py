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

    def __init__(self, initial_citizens=5000, initial_cops=100, rationality_of_agents=1, bribe_mean_std=(0.5, 0.1),
                 moral_commitment_mean_std=(0.3, 0.2)):
        '''
        :param initial_citizens:
        :param initial_cops:
        :param rationality_of_agents: 0 agents completely random, 1 completely rational
        '''
        super().__init__()

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
            {"Citizens": lambda m: sum(1 for agent in self.schedule.agents if isinstance(agent, Citizen)),
             "Cops":lambda m: sum(1 for agent in self.schedule.agents if isinstance(agent, Cop)),
             "Bribing": lambda m: sum([1 for cop in self.schedule.agents if type(cop) == Cop and cop.action == "bribe"])/self.number_of_citizens,
             "NoBribing": lambda m: sum([1 for cop in self.schedule.agents if type(cop) == Cop and cop.action == "not_bribe"])/self.number_of_citizens,
             "ComplainRate": lambda m: sum([1 for cit in self.schedule.agents if type(cit) == Citizen and (cit.action == "accept_and_complain" or cit.action == "reject_and_complain")])/self.number_of_citizens,
             "NoComplainRate": lambda m: sum([1 for cit in self.schedule.agents if type(cit) == Citizen and (cit.action == "accept_and_silent" or cit.action == "reject_and_silent")])/self.number_of_citizens,
             })

        # Create cops
        for i in range(self.initial_cops):
            cop = Cop(self.next_id(), self, self.lambda_, bribe_mean_std=bribe_mean_std,
                      moral_commitment_mean_std=moral_commitment_mean_std)
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
        self.prob_prosecution = 0

        # This should be 1 so other values are more or less normalized in respect to it
        self.fine = 1.

        # This I think should be here as it's more global?
        self.cost_of_complaining = 0.1
        # TODO: Should this be different for each individual? Or dependent on an environment somehow?
        self.cost_of_silence = 0.1
        # This is systematic, so I think global is good
        self.penalty_citizen = 0.1
        self.penalty_cop = 0.1
        self.reward_citizen = 0.1

    def step(self):
        '''
        Method that calls the step method for each of the citizens, and then for each of the cops.
        '''

        # Create list of available cops so then caught citizen can be assigned to one cop
        self.available_cops = [agent for agent in self.schedule.agents if isinstance(agent, Cop)]

        # Number of caught citizens should be from 0 to num of cops as not always all cops are busy
        #self.number_of_citizens = random.randint(1, len(list(filter(lambda a: type(a) == Cop, self.schedule.agents))))
        self.number_of_citizens = len(list(filter(lambda a: type(a) == Cop, self.schedule.agents)))

        self.caught_citizens = random.choices([agent for agent in self.schedule.agents if isinstance(agent, Citizen)], k=self.number_of_citizens)

        self.schedule.step()

        # Save the statistics
        self.datacollector.collect(self)

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
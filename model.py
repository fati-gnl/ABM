import random

import axelrod as axl
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

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
        self.initial_cops = initial_cops
        self.lambda_ = rationality_of_agents

        # TODO: check if COp schedule could be removed
        # TODO: maybe other type of scheduler would be better?
        self.schedule_Citizen = RandomActivation(self)
        self.schedule_Cop = RandomActivation(self)

        # Data collector to be able to save the data - cop and citizen count
        # Todo: We should collect other data - I guess total money
        self.datacollector = DataCollector(
            {"Citizens": lambda m: self.schedule_Citizen.get_agent_count(),
             "Cops": lambda m: self.schedule_Cop.get_agent_count(),
             "Bribing": lambda m: sum([1 for cop in self.schedule_Cop.agents if cop.action == "bribe"]),
             "NotBribing": lambda m: self.schedule_Cop.get_agent_count() - sum(
                 [1 for cop in self.schedule_Cop.agents if cop.action == "bribe"]),
             })

        # Create citizens
        for i in range(self.initial_citizens):
            citizen = Citizen(self.next_id(), self, self.lambda_)
            self.schedule_Citizen.add(citizen)

        # Create cops: No two cops should be placed on the same cell
        for i in range(self.initial_cops):
            cop = Cop(self.next_id(), self, self.lambda_, bribe_mean_std=bribe_mean_std,
                      moral_commitment_mean_std=moral_commitment_mean_std)
            self.schedule_Cop.add(cop)

        # Needed for the Batch run
        self.running = True
        # Needed for the datacollector
        self.datacollector.collect(self)

        # TODO: make it change depending on the environment
        # TODO: decide if should be moved to each agent individually
        self.prob_prosecution = 0.2

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
        self.available_cops = self.schedule_Cop.agents.copy()
        # Number of caught citizens should be from 0 to num of cops as not always all cops are busy
        number_of_citizens = random.randint(0, self.schedule_Cop.get_agent_count())
        self.caught_citizens = random.choices(self.schedule_Citizen.agents, k=number_of_citizens)

        self.schedule_Citizen.step()
        # Save the statistics
        self.datacollector.collect(self)

    def run_model(self, step_count=5):
        '''
        Method that runs the model for a specific amount of steps.
        '''
        for i in range(step_count):
            self.step()

    def get_cop(self):
        '''
        Gets a cop from the available cops, this cop is not available anymore.
        If all cops busy, return None
        :return: Cop object or None
        '''
        if len(self.available_cops) > 0:
            return random.sample(self.available_cops, 1)[0]
        return None

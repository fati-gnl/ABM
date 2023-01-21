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

    def __init__(self, initial_citizens=100, initial_cops=0, strategies=[axl.Cooperator, axl.Defector]):

        super().__init__()

        self.initial_citizens = initial_citizens
        self.initial_cops = initial_cops

        # Add a schedule for citizens and cops seperately to prevent race-conditions
        # Check if they can be drawn instead (citizens)
        self.schedule_Citizen = RandomActivation(self)
        self.schedule_Cop = RandomActivation(self)

        # Data collector to be able to save the data - cop and citizen count
        # Todo: We should collect other data - I guess total money
        self.datacollector = DataCollector(
            {"Citizens": lambda m: self.schedule_Citizen.get_agent_count(),
             "Cops": lambda m: self.schedule_Cop.get_agent_count()})

        # Create citizens
        for i in range(self.initial_citizens):
            citizen = Citizen(self.next_id(), self)
            self.schedule_Citizen.add(citizen)

        # Create cops: No two cops should be placed on the same cell
        for i in range(self.initial_cops):
            cop = Cop(self.next_id(), self)
            self.schedule_Cop.add(cop)

        # Needed for the Batch run
        self.running = True
        # Needed for the datacollector
        self.datacollector.collect(self)

        self.prob_prosecution = 0.2

    def remove_agent(self, agent):
        '''
        Method that removes an agent from the grid and the correct scheduler.
        '''
        getattr(self, f'schedule_{type(agent).__name__}').remove(agent)

    def step(self):
        '''
        Method that calls the step method for each of the citizens, and then for each of the cops.
        '''
        self.available_cops = self.schedule_Cop.agents.copy()
        number_of_citizens = random.randint(0, self.schedule_Cop.get_agent_count())
        self.caught_citizens= random.choices(self.schedule_Citizen.agents, k=number_of_citizens)

        self.schedule_Citizen.step()
        # self.schedule_Cop.step()


        print(self.schedule_Citizen.get_agent_count())
        # Save the statistics
        self.datacollector.collect(self)

    def run_model(self, step_count=5):
        '''
        Method that runs the model for a specific amount of steps.
        '''
        for i in range(step_count):
            self.step()

    def get_cop(self):
        if len(self.available_cops) > 0:
            return random.sample(self.available_cops,1)
        return None


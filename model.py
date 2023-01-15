import random

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from agents import Citizen,Cop

class CopCitizen(Model):
    '''
    Corruption Model: Citizens and Cops
    '''

    def __init__(self, height=20, width=20,
                 initial_citizens=7, initial_cops=5,
                 sheep_reproduction_chance=0.05, wolf_death_chance=0.05):

        super().__init__()

        self.height = height
        self.width = width
        self.initial_citizens = initial_citizens
        self.initial_cops = initial_cops
        self.sheep_reproduction_chance = sheep_reproduction_chance
        self.wolf_death_chance = wolf_death_chance

        #Add a schedule for citizens and cops seperately to prevent race-conditions
        self.schedule_Citizen = RandomActivation(self)
        self.schedule_Cop = RandomActivation(self)

        self.grid = MultiGrid(self.width, self.height, torus=True)

        # Data collector to be able to save the data - cop and citizen count
        # Todo: We should collect other data - I guess total money
        self.datacollector = DataCollector(
             {"Citizens": lambda m: self.schedule_Citizen.get_agent_count(),
              "Cops": lambda m: self.schedule_Cop.get_agent_count()})

        # Create citizens
        for i in range(self.initial_citizens):
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            citizen = Citizen(self.next_id(), self, (x,y))
            self.grid.place_agent(citizen, (x, y))
            self.schedule_Citizen.add(citizen)

        # Create cops: No two cops should be placed on the same cell
        # Todo: No two cops should be next to each other
        current_coord = []
        for i in range(self.initial_cops):
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            while (x, y) in current_coord:
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
            current_coord.append((x, y))
            cop = Cop(self.next_id(), self, (x,y))
            self.grid.place_agent(cop, (x, y))
            self.schedule_Cop.add(cop)

        # Needed for the Batch run
        self.running = True
        # Needed for the datacollector
        self.datacollector.collect(self)

    def step(self):
        '''
        Method that calls the step method for each of the citizens, and then for each of the cops.
        '''
        self.schedule_Citizen.step()
        self.schedule_Cop.step()

        # Save the statistics
        self.datacollector.collect(self)

    def run_model(self, step_count=5):
        '''
        Method that runs the model for a specific amount of steps.
        '''
        for i in range(step_count):
            self.step()
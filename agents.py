from mesa import Agent
import random


class Citizen(Agent):
    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy

    def step(self):
        '''
        This method should ...
        Input:
        Output:
        '''
        pass


class Cop(Agent):
    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy

    def step(self):
        '''
        This method should ...
        Input:
        Output:
        '''
        pass
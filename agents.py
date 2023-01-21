import numpy.random
from mesa import Agent
import random

Citizen_actions = ["accept_and_complain", "accept_and_silent", "reject_and_complain", "reject_and_silent"]
Cop_actions = ["bribe", "not_bribe"]

fine = 1.
cost_of_complaining = 0.1
cost_of_rejecting = 0.1

penalty_citizen = 0.1
penalty_cop = 0.1

reward_citizen = 0.1


class Citizen(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.actions = Citizen_actions
        self.payoffs = dict.fromkeys(self.actions, 0)
        self.action = None

    def step(self):
        '''
        This method should ...
        Input:
        Output:
        '''
        if self.unique_id in [citizen.unique_id for citizen in self.model.caught_citizens]:
            self.play()

    def play(self):
        '''
        If Cititzen selected they play
        :return:
        '''
        # all the moves are done here
        cop = self.model.get_cop()
        cop.do_action()
        self.do_action()

        if cop.action == Cop_actions[0]:
            # bribe
            if self.action == Citizen_actions[0]:
                # accept_and_complain
                self.payoffs[self.action] += fine - cop.bribe - cost_of_complaining + self.model.prob_prosecution * (
                        reward_citizen - penalty_citizen)
                cop.payoffs[cop.action] += cop.bribe - self.model.prob_prosecution * (penalty_cop + cop.bribe)
            elif self.action == Citizen_actions[1]:
                # accept_and_silent
                self.payoffs[self.action] += fine - cop.bribe
                cop.payoffs[cop.action] += cop.bribe
            elif self.action == Citizen_actions[2]:
                # reject_and_complain
                self.payoffs[self.action] += fine - cost_of_rejecting + self.model.prob_prosecution * (reward_citizen)
                cop.payoffs[cop.action] += self.model.prob_prosecution * penalty_cop
            elif self.action == Citizen_actions[3]:
                # reject_and_silent
                self.payoffs[self.action] += fine - cost_of_rejecting
                cop.payoffs[cop.action] += 0

        else:
            # no bribe
            self.payoffs[self.action] += fine
            cop.payoffs[cop.action] += cop.moral_commitment

    def do_action(self):
        # categorical distribution over payoffs
        self.action = numpy.random.multinomial(1, self.payoffs.values())

class Cop(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.actions = Cop_actions
        self.payoffs = dict.fromkeys(self.actions, 0)
        self.moral_commitment = random.random()
        self.action = None
        self.bribe = 0.5

    def do_action(self):
        # categorical distribution over payoffs
        self.action = numpy.random.multinomial(1, self.payoffs.values())
        # should be also drawn randomly
        self.bribe = 0.5

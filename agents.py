import numpy.random
from mesa import Agent
import random

CITIZEN_ACTIONS = ["accept_and_complain", "accept_and_silent", "reject_and_complain", "reject_and_silent"]
COP_ACTIONS = ["bribe", "not_bribe"]




class Citizen(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.actions = CITIZEN_ACTIONS.copy()
        # Payoff matrix for the individual citizen. Initialized at 0.01 so there is a chance of playing each action.
        self.payoffs = dict.fromkeys(self.actions, 0.01)
        self.action = None

    def step(self):
        '''
        This method should check first if a citizen is in the caught citizens. Then if so the citizen should play
        '''
        if self.unique_id in [citizen.unique_id for citizen in self.model.caught_citizens]:
            self.play()

    def play(self):
        '''
        If Cititzen selected they play(meets cop).
        The logic of the meeting is implemented here and not in the cop class.
        '''
        # all the moves are done here and not in the cop step function
        # First pick a cop that is assigned to this citizen
        cop = self.model.get_cop()
        # cop makes an action based on payoff matrix
        cop.do_action()
        # citizen makes an action based on payoff matrix
        self.do_action()

        # TODO: change the utilities here according to the new ones
        # depending on the choices different payoff is assigned
        if cop.action == COP_ACTIONS[0]:
            # bribe
            if self.action == CITIZEN_ACTIONS[0]:
                # accept_and_complain
                self.payoffs[self.action] += - cop.bribe - self.model.cost_of_complaining + self.model.prob_prosecution * (
                        self.model.reward_citizen - self.model.penalty_citizen)
                cop.payoffs[cop.action] += cop.bribe - self.model.prob_prosecution * (self.model.penalty_cop + cop.bribe)
            elif self.action == CITIZEN_ACTIONS[1]:
                # accept_and_silent
                self.payoffs[self.action] += - cop.bribe - self.model.cost_of_silence
                cop.payoffs[cop.action] += cop.bribe
            elif self.action == CITIZEN_ACTIONS[2]:
                # reject_and_complain
                self.payoffs[self.action] += -self.model.cost_of_complaining + self.model.prob_prosecution * (self.model.reward_citizen)
                cop.payoffs[cop.action] += self.model.prob_prosecution * self.model.penalty_cop
            elif self.action == CITIZEN_ACTIONS[3]:
                # reject_and_silent
                self.payoffs[self.action] += self.model.cost_of_silence
                cop.payoffs[cop.action] += 0

        else:
            # no bribe
            self.payoffs[self.action] += -self.model.fine
            cop.payoffs[cop.action] += cop.moral_commitment

    def do_action(self):
        '''
        This function sets an action to a chosen action.
        The action is chosen based on the payoff matrix of the agent.
        '''
        # TODO change to a logit function instead
        payoff_sum = sum(list(self.payoffs.values()))
        payoff_sum = 1 if payoff_sum == 0 else payoff_sum
        normalized_payoffs = [x / payoff_sum for x in list(self.payoffs.values())]
        # categorical distribution over payoffs
        self.action = CITIZEN_ACTIONS[ numpy.argmax(numpy.random.multinomial(1, normalized_payoffs))]

class Cop(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.actions = COP_ACTIONS.copy()
        # Payoff matrix for the individual citizen. Initialized at 0.01 so there is a chance of playing each action.
        self.payoffs = dict.fromkeys(self.actions, 0.01)
        # Each cop has a different moral commitment value, drawn from a normal distribution.
        # TODO: change to normal distribution
        self.moral_commitment = random.random()
        self.action = None
        # Each cop has a different bribe value as they're greediness varies, drawn from a normal distribution.
        # TODO: change to normal distribution
        self.bribe = 0.5

    def do_action(self):
        '''
        This function sets an action to a chosen action.
        The action is chosen based on the payoff matrix of the agent.
        '''
        # TODO: change to a logit function instead
        # TODO: extract the function out of this
        # categorical distribution over payoffs
        payoff_sum = sum(list(self.payoffs.values()))
        payoff_sum =1 if payoff_sum==0 else payoff_sum
        normalized_payoffs = [x / payoff_sum for x in list(self.payoffs.values())]
        self.action =COP_ACTIONS[numpy.argmax(numpy.random.multinomial(1, normalized_payoffs))]

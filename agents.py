from typing import List, Type

import numpy as np
import numpy.random
from mesa import Agent
import random

class Functions:

    def softmax(self, x, lambda_):
        x = x * lambda_
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def sample_action(self, utilities, id_act, lambda_):
        distribution = self.softmax(utilities, lambda_)
        action = id_act[np.argmax(np.random.multinomial(1, distribution))]
        return action

class PayoffAgent(Agent):
    def __init__(self, unique_id, model, lambda_):
        super().__init__(unique_id, model)
        self.lambda_ = lambda_

        self.actions = []

        self.action = None

    def init_action_dicts(self):
        # Payoff matrix for the individual citizen.
        self.payoffs = dict.fromkeys(self.actions, 0)
        self.action_count = dict.fromkeys(self.actions, 0)

    def avg_payoffs(self):
        '''
        Take an average of payoffs
        :return: averaged payoff matrix
        '''
        mean_payoffs = [
            self.payoffs[action] / self.action_count[action] if self.action_count[action] > 0 else self.payoffs[action]
            for action in self.actions]
        return mean_payoffs

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def choose_action(self):
        '''
        :return: Action number
        '''
        avg_payoffs = np.array(self.avg_payoffs())
        # logit equilibrium distribution
        distribution = self.softmax(self.lambda_ * avg_payoffs)
        return numpy.argmax(numpy.random.multinomial(1, distribution))


class Citizen(PayoffAgent):
    def __init__(self, unique_id, model, lambda_, reward = 50):
        super().__init__(unique_id, model, lambda_)

        self.actions = Actions.get_actions(Citizen)
        self.init_action_dicts()
        self.action = None

        self.reward = reward

    def step(self):
        '''
        This method should check first if a citizen is in the caught citizens. Then if so the citizen should play
        '''
        self.action = None
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

        # depending on the choices different payoff is assigned
        if cop.action == "bribe":
            payoff_matrix = {}

            payoff_matrix["accept_and_complain"] = -cop.bribe_amount - self.model.cost_of_complaining - self.model.prob_succesful_complain * \
                                             (self.model.reward_citizen - self.model.penalty_citizen) - self.model.cost_of_accepting
            payoff_matrix["accept_and_silent"] = -cop.bribe_amount - self.model.cost_of_silence - self.model.cost_of_accepting
            payoff_matrix["reject_and_complain"] = -cop.fine - self.model.cost_of_complaining - self.model.prob_succesful_complain * self.reward
            payoff_matrix["reject_and_silent"] = -cop.fine - self.model.cost_of_silence

            payoffs = payoff_matrix.values()
            payoff_probabilities = np.argmax(self.softmax(payoffs))

        else:
            pass

            #self.payoffs[self.action] += 0
            #cop.payoffs[cop.action] += cop.moral_commitment

    def do_action(self):
        '''
        This function sets an action to a chosen action.
        The action is chosen based on the payoff matrix of the agent.
        '''
        self.action = Actions.get_actions(Citizen)[self.choose_action()]
        self.action_count[self.action] += 1


class Cop(PayoffAgent):

    def __init__(self, unique_id, model, lambda_, bribe_mean_std=(0.5, 0.1), moral_commitment_mean_std=(0.3, 0.2)):
        super().__init__(unique_id, model, lambda_)
        self.actions = Actions.get_actions(Cop)
        self.action = None

        self.jail_time = random.randint(0,1)
        self.bribe_amount = bribe_amount
        self.fine = fine

        # Each cop has a different moral commitment value, drawn from a normal distribution.
        self.moral_commitment = np.random.normal(loc=moral_commitment_mean_std[0], scale=moral_commitment_mean_std[1])
        # Each cop has a different bribe value as they're greediness varies, drawn from a normal distribution.
        self.bribe = np.random.normal(loc=bribe_mean_std[0], scale=bribe_mean_std[1])

        self.init_action_dicts()

    def step(self):
        self.action = None

    def do_action(self):
        '''
        This function sets an action to a chosen action.
        The action is chosen based on the payoff matrix of the agent.
        '''
        self.action = Actions.get_actions(Cop)[self.choose_action()]
        self.action_count[self.action] += 1
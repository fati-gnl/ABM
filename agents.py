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


class Citizen(Agent, Functions):
    def __init__(self,
                 unique_id,
                 model,
                 lambda_,
                 prob_success_complain,
                 complain_reward,
                 fine_amount,
                 memory_size,
                 cost_accept_mean,
                 cost_accept_std,
                 cost_silence_mean,
                 cost_silence_std):
        super().__init__(unique_id, model)

        self.action = None
        self.cost_complain = self.model.cost_complain
        self.cost_accept = np.random.normal(loc=cost_accept_mean, scale=cost_accept_std)
        self.cost_silence = np.random.normal(loc=cost_silence_mean, scale=cost_silence_std)
        self.prob_success_complain = prob_success_complain
        self.complain_reward = complain_reward
        self.lambda_ = lambda_
        self.fine_amount = fine_amount

        self.memory_size = memory_size
        self.memory = [0] * self.memory_size

        # Hello,
        self.id_act = {0: "accept_complain",
                       1: "accept_silent",
                       2: "reject_complain",
                       3: "reject_silent"}

    def get_action(self, bribe_amount):
        # utility_accept_complain = -self.bribe_amount - self.cost_complain - self.prob_success_complain*(0.5*self.complain_reward) - self.cost_accept
        # utility_accept_reject = -self.bribe_amount - self.cost_silence - self.cost_accept
        # utility_reject_complain = -self.fine_amount - self.cost_complain - self.prob_success_complain*self.complain_reward
        # utility_reject_reject = -self.fine_amount - self.cost_silence

        utilities = np.array([10,
                              10,
                              5,
                              5])

        self.action = self.sample_action(utilities, self.id_act, self.lambda_)

        return self.action

    def approximate_prob_accept(self):
        return sum(self.memory) / self.memory_size

    def step(self):
        self.action = None

    def update_memory(self, update):
        self.memory.pop(0)
        self.memory.append(update)


class Cop(Agent, Functions):
    def __init__(self, unique_id, model, in_jail, lambda_, memory_size, bribe_amount_mean, bribe_amount_std):
        super().__init__(unique_id, model)

        self.in_jail = in_jail
        self.lambda_ = lambda_
        self.memory_size = memory_size
        self.memory = [0] * self.memory_size
        self.action = None
        self.bribe_amount_mean = bribe_amount_mean
        self.bribe_amount_std = bribe_amount_std

        self.id_act = {0: "bribe", 1: "not_bribe"}

    def step(self):
        self.action = None
        # Check whether this cop can play this round
        play = self.validate_play()

        # If cop can play, chose a random citizen from the avialable citizens and sample an action for the cop
        if play:
            cop_action = self.get_action()
            citizen = self.model.get_citizen()

            # If the cop bribes, sample an action for the citizen
            if cop_action == "bribe":
                cit_action = citizen.get_action(self.bribe_amount)

                # If the citizen complains, give the cop a jail time based on the ground truth of the probability of getting caught
                if "complain" in cit_action:
                    self.in_jail = self.model.jail_time * \
                                   np.random.multinomial(1, [self.model.prob_caught, 1 - self.model.prob_caught])[0]

                # If the citizen accepts to bribe, update this in the memory for the cop
                if "accept" in cit_action:
                    self.update_memory(1)

                # If the citizen rejects to bribe, update this in the memory for the cop
                if "reject" in cit_action:
                    self.update_memory(0)

        # If the cop cannot play, check whether the cop is in jail and if the cop is in jail, reduce their sentence by 1
        elif self.in_jail > 0:
            self.in_jail -= 1

    def validate_play(self):
        play = True if self in self.model.cops_playing else False
        return play

    def get_action(self):
        self.bribe_amount = np.random.normal(loc=self.bribe_amount_mean, scale=self.bribe_amount_std)

        approx_prob_caught = self.approximate_prob_caught()
        approx_prob_accept = self.approximate_prob_accept()

        utility_bribe = (1 - approx_prob_caught) * (approx_prob_accept * self.bribe_amount)
        utility_not_bribe = approx_prob_caught * self.model.jail_time

        utilities = np.array([utility_bribe, utility_not_bribe])

        self.action = self.sample_action(utilities, self.id_act, self.lambda_)

        return self.action

    def approximate_prob_accept(self):
        return sum(self.memory) / self.memory_size

    def approximate_prob_caught(self):
        team = self.model.id_team[self.unique_id]
        m = self.model.team_jailed[team]
        return m / self.model.team_size

    def update_memory(self, update):
        self.memory.pop(0)
        self.memory.append(update)
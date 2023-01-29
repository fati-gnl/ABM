from typing import List, Type, Tuple

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
                 cost_accept_mean_std: Tuple[float, float],
                 cost_silence_mean_std: Tuple[float, float],
                 prone_to_complain: float,
                 complain_memory_discount_factor: float,
                 first_action: str = None,
                 ):
        super().__init__(unique_id, model)

        self.action = first_action
        self.cost_complain = self.model.cost_complain

        self.lambda_ = self.model.lambda_
        self.fine_amount = self.model.fine_amount
        self.penalty_citizen_prosecution = self.model.penalty_citizen_prosecution

        self.cost_accept = np.random.normal(loc=cost_accept_mean_std[0], scale=cost_accept_mean_std[1])
        self.cost_silence = np.random.normal(loc=cost_silence_mean_std[0], scale=cost_silence_mean_std[1])

        # Initialize memory, complain_memory ==0.5 means that the beginning state is being indifferent
        self.complain_memory_len = 1
        self.complain_memory = prone_to_complain
        # 0 is easily forgetting, 1 all events important the same
        self.discount_factor = complain_memory_discount_factor

        # Hello,
        self.id_act = {0: "accept_complain",
                       1: "accept_silent",
                       2: "reject_complain",
                       3: "reject_silent"}

    def do_action(self, bribe_amount):
        # complain_reward = bribe to make the # of params smaller
        prob_success_complain = self.approximate_prob_succesful_complain()
        utility_accept_complain = -bribe_amount - self.cost_complain + prob_success_complain * (
                -self.penalty_citizen_prosecution + bribe_amount) - self.cost_accept
        utility_accept_silent = -bribe_amount - self.cost_silence - self.cost_accept
        utility_reject_complain = -self.fine_amount - self.cost_complain
        utility_reject_silent = -self.fine_amount - self.cost_silence

        utilities = np.array([utility_accept_complain,
                              utility_accept_silent,
                              utility_reject_complain,
                              utility_reject_silent])

        self.action = self.sample_action(utilities, self.id_act, self.lambda_)

    def step(self):
        self.action = None

    def approximate_prob_succesful_complain(self):
        return self.complain_memory

    def update_succesful_complain_memory(self, update):
        self.complain_memory_len += 1
        self.complain_memory += self.discount_factor * (self.complain_memory * self.complain_memory_len * (
                self.complain_memory_len - 1)) + update / self.complain_memory_len


class Cop(Agent, Functions):
    def __init__(self, unique_id,
                 model,
                 in_jail,
                 lambda_: float,
                 accepted_bribe_memory_size: int,
                 bribe_amount_mean_std: Tuple[float, float],
                 moral_commitment_mean_std: Tuple[float, float],
                 jail_cost: float,
                 first_action: str = None,
                 accepted_bribe_memory_initial: float = 0.5):
        super().__init__(unique_id, model)

        self.action = first_action

        self.in_jail = in_jail

        self.lambda_ = lambda_
        self.accepted_bribe_memory_size = accepted_bribe_memory_size
        self.accepted_bribe_memory = [accepted_bribe_memory_initial] * self.accepted_bribe_memory_size

        self.bribe_amount_mean_std = bribe_amount_mean_std

        # How much the cop is against taking risks. 0 - likes risk a lot, 1. - doesnt like risk at all
        self.risk_aversion = np.random.uniform(0., 1.)
        self.moral_commitment = np.random.normal(loc=moral_commitment_mean_std[0], scale=moral_commitment_mean_std[1])
        self.jail_cost = jail_cost

        self.id_act = {0: "bribe", 1: "not_bribe"}

    def step(self):
        self.action = None
        # Check whether this cop can play this round
        play = self.validate_play()

        # If cop can play, chose a random citizen from the available citizens and sample an action for the cop
        if play:
            self.do_action()
            citizen = self.model.get_citizen()

            # If the cop bribes, sample an action for the citizen
            if self.action == "bribe":
                citizen.do_action(self.bribe_amount)

                # If the citizen complains, give the cop a jail time based on the ground truth of the probability of getting caught
                if "complain" in citizen.action:
                    cop_goes_to_jail = \
                        np.random.multinomial(1, [self.model.prob_of_prosecution, 1 - self.model.prob_of_prosecution])[
                            0]
                    if cop_goes_to_jail == 1:
                        # complain succesful
                        self.in_jail = self.model.jail_time
                        citizen.update_succesful_complain_memory(1)
                    else:
                        # complain failed, citizen remembers that
                        citizen.update_succesful_complain_memory(0)
                # If the citizen accepts to bribe, update this in the memory for the cop
                if "accept" in citizen.action:
                    self.update_accepting_bribe_memory(1)

                # If the citizen rejects to bribe, update this in the memory for the cop
                if "reject" in citizen.action:
                    self.update_accepting_bribe_memory(0)


        # If the cop cannot play, check whether the cop is in jail and if the cop is in jail, reduce their sentence by 1
        elif self.in_jail > 0:
            self.in_jail -= 1

    def validate_play(self):
        '''
        Checks if the cop is allowed to play. They are allowed if they're not in jail. This is checked in model
        :return:True if allowed to play
        '''
        return True if self in self.model.cops_playing else False

    def do_action(self):
        '''
        Cop is making an action based on utilities. The sampled action is then saved in the self.action field.
        '''
        # Draw the bribe amount from the normal distribution
        self.bribe_amount = np.random.normal(loc=self.bribe_amount_mean_std[0], scale=self.bribe_amount_mean_std[1])

        approx_prob_caught = self.approximate_prob_caught()
        approx_prob_accept = self.approximate_prob_accept()

        # Calculate expected utilities for each action
        utility_bribe = (1 - approx_prob_caught) * (approx_prob_accept * (
                1 - self.risk_aversion) * self.bribe_amount) - approx_prob_caught * self.jail_cost
        utility_not_bribe = self.moral_commitment

        utilities = np.array([utility_bribe, utility_not_bribe])

        self.action = self.sample_action(utilities, self.id_act, self.lambda_)

    def approximate_prob_accept(self):
        return sum(self.accepted_bribe_memory) / self.accepted_bribe_memory_size

    def approximate_prob_caught(self):
        '''
        This function checks how many cops in the network/group are currently in jail. This rate is the estimated probability of probability of prosecution
        :return: estimated probability of getting caught, 0 to 1
        '''
        # TODO: move it teamID be a field of an agent
        team = self.model.id_team[self.unique_id]
        m = self.model.team_jailed[team]
        return m / self.model.team_size

    def update_accepting_bribe_memory(self, update):
        self.accepted_bribe_memory.pop(0)
        self.accepted_bribe_memory.append(update)

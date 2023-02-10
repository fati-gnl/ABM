from copy import deepcopy
from typing import List, Type, Tuple
import numpy as np
import numpy.random
from mesa import Agent

from utils import CitizenActions, sample_action, CopActions


class Citizen(Agent):
    def __init__(self,
                 unique_id,
                 model,
                 cost_accept_mean_std: Tuple[float, float],
                 complain_memory_discount_factor: float,
                 prone_to_complain: float = 0.5,
                 first_action: str = None
                 ):
        super().__init__(unique_id, model)

        self.action = first_action

        # initialize moral costs
        self.cost_accept = np.random.normal(loc=cost_accept_mean_std[0], scale=cost_accept_mean_std[1])

        # Initialize memory, complain_memory ==0.5 means that the beginning state is being indifferent
        self.complain_memory_accumulated_weights = 1
        self.complain_memory = prone_to_complain
        # 0 is easily forgetting, 1 all events important the same
        self.discount_factor = complain_memory_discount_factor

        self.possible_actions = CitizenActions

    def do_action(self, bribe_amount):
        prob_success_complain = self.approximate_prob_successful_complain()

        # complain_reward = bribe to make the # of params smaller
        utility_accept_complain = -bribe_amount + prob_success_complain * (
                bribe_amount - self.model.penalty_citizen_prosecution) - self.model.cost_complain - self.cost_accept
        utility_accept_silent = -bribe_amount - self.cost_accept
        utility_reject_complain = -self.model.fine_amount - self.model.cost_complain
        utility_reject_silent = -self.model.fine_amount

        utilities = np.array([utility_accept_complain,
                              utility_accept_silent,
                              utility_reject_complain,
                              utility_reject_silent])

        self.action = sample_action(utilities, self.possible_actions, self.model.rationality_of_agents)

    def step(self):
        self.action = None

    def approximate_prob_successful_complain(self):
        """
        Estimate how successful complain will be.
        :return: estimated probability of cop being prosecuted.
        """
        return self.complain_memory

    def update_successful_complain_memory(self, update):
        """
        Updates running, discounted average. This way doesn't require remembering each event.
        Discount factor should be in [0,1] range.
         0 is when only last experience is important. 1 - all experiences are weighted the same
        :param update: complain event result, 0 - cop is not caught, 1 - cop is caught
        """
        old_complain_memory_sum_weights = self.complain_memory_accumulated_weights
        self.complain_memory_accumulated_weights = self.complain_memory_accumulated_weights * self.discount_factor + 1
        old_mean_rate = old_complain_memory_sum_weights / self.complain_memory_accumulated_weights

        self.complain_memory = self.discount_factor * old_mean_rate * self.complain_memory + update / self.complain_memory_accumulated_weights
        assert self.complain_memory <= 1.0 or self.complain_memory >= 0.0, (
                "Complain memory is out of proper range! " + str(self.complain_memory))

    def log_data(self, step: int) -> dict:
        """
        Creates a dictionary with all params of this agent
        :return: dict with results
        """

        data = {
            'action': self.action if self.action is None else self.action.name,
            'complain_memory': self.complain_memory
        }
        if step == 0:
            data['cost_accept'] = self.cost_accept
        return data.copy()


class Cop(Agent):
    def __init__(self, unique_id,
                 model,
                 time_left_in_jail: int,
                 accepted_bribe_memory_size: int,
                 bribe_amount: float,
                 moral_commitment_mean_std: Tuple[float, float],
                 first_action: str = None,
                 accepted_bribe_memory_initial: float = 0.5):
        super().__init__(unique_id, model)

        self.action = first_action
        self.bribe_amount = bribe_amount

        self.time_left_in_jail = time_left_in_jail
        self.accepted_bribe_memory_size = accepted_bribe_memory_size
        self.accepted_bribe_memory = [accepted_bribe_memory_initial] * accepted_bribe_memory_size
        # for stats
        self.estimated_prob_accept = self.approximate_prob_accept()

        self.moral_commitment = np.random.normal(loc=moral_commitment_mean_std[0], scale=moral_commitment_mean_std[1])

        self.possible_actions = CopActions

    def step(self):
        self.action = None
        # Check whether this cop can play this round
        play = self.validate_play()

        # If cop can play, chose a random citizen from the available citizens and sample an action for the cop
        if play:
            self.do_action()
            citizen = self.model.get_citizen()

            # If the cop bribes, sample an action for the citizen
            if self.action == CopActions.bribe:
                citizen.do_action(self.bribe_amount)

                # If the citizen complains, give the cop a jail time based on the ground truth of the probability of getting caught
                if citizen.action in [CitizenActions.accept_complain, CitizenActions.reject_complain]:
                    if np.random.multinomial(1,
                                             [self.model.prob_of_prosecution,
                                              1 - self.model.prob_of_prosecution])[0] == 1:
                        # complain succesful -> cop goes to jail
                        self.time_left_in_jail = self.model.jail_time
                        citizen.update_successful_complain_memory(1)
                    else:
                        # complain failed, citizen remembers that
                        citizen.update_successful_complain_memory(0)
                # If the citizen accepts to bribe, update this in the memory for the cop
                if citizen.action in [CitizenActions.accept_complain, CitizenActions.accept_silent]:
                    self.update_accepting_bribe_memory(1)

                # If the citizen rejects to bribe, update this in the memory for the cop
                if citizen.action in [CitizenActions.reject_complain, CitizenActions.reject_silent]:
                    self.update_accepting_bribe_memory(0)


        # If the cop cannot play, check whether the cop is in jail and if the cop is in jail, reduce their sentence by 1
        elif self.time_left_in_jail > 0:
            self.time_left_in_jail -= 1

    def validate_play(self):
        """
        Checks if the cop is allowed to play. They are allowed if they're not in jail. This is checked in model
        :return:True if allowed to play
        """
        return True if self in self.model.cops_playing else False

    def do_action(self):
        """
        Cop is making an action based on utilities. The sampled action is then saved in the self.action field.
        """

        approx_prob_caught = self.approximate_prob_caught()
        approx_prob_accept = self.approximate_prob_accept()

        # Calculate expected utilities for each action
        utility_bribe = (
                                1 - approx_prob_caught) * approx_prob_accept * self.bribe_amount - approx_prob_caught * self.model.jail_cost
        utility_not_bribe = self.moral_commitment

        utilities = np.array([utility_bribe, utility_not_bribe])

        self.action = sample_action(utilities, self.possible_actions, self.model.rationality_of_agents)

    def approximate_prob_caught(self):
        """
        This function checks how many cops in the network/group are currently in jail. This rate is the estimated probability of probability of prosecution
        :return: estimated probability of getting caught, 0 to 1
        """
        team = self.model.id_team[self.unique_id]
        m = self.model.team_jailed[team]
        return m / self.model.team_size

    def update_accepting_bribe_memory(self, update):
        """
        Writes the information about last bribing attempt being successful. Keeps the memory in certain size.
        :param update: last bribing attempt result. 0 - not successful, 1 - successful
        """

        self.accepted_bribe_memory.append(update)
        if len(self.accepted_bribe_memory) > self.accepted_bribe_memory_size:
            self.accepted_bribe_memory.pop(0)

    def approximate_prob_accept(self):
        """
        Takes the average of the attempts.
        :return: estimated probability of accepting the bribe by a citizen
        """
        # saving it here to have it in the logged data
        self.estimated_prob_accept = sum(self.accepted_bribe_memory) / self.accepted_bribe_memory_size
        return self.estimated_prob_accept

    def log_data(self, step: int) -> dict:
        """
        Creates a dictionary with all params of this agent
        :return: dict with results
        """

        data = {'action': self.action if self.action is None else self.action.name,
                'time_left_in_jail': self.time_left_in_jail,
                'estimated_prob_accept': self.estimated_prob_accept,
                'approximated_prob_caught': self.approximate_prob_caught()
                }
        if step == 0:
            data['moral_commitment'] = self.moral_commitment
        return data

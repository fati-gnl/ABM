from enum import Enum
from typing import Type

import numpy as np


def softmax(x: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Calculate softmax of x, weighted by lambda.
    This is logit equlibrium function https://en.wikipedia.org/wiki/Quantal_response_equilibrium#Logit_equilibrium
    This function is implemented so it's numerically stable.
    :param x: estimated payoffs for each action
    :param lambda_: rationality factor. 0 - random
    :return: computed softmax
    """
    x = x * lambda_
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample_action(utilities: np.ndarray, possible_actions: Type[Enum], lambda_: float) -> Enum:
    """
    Sample action taking into considerations estimated utilities per each action.
    :param utilities: per each action estimated utility will be treated as distribution
    :param possible_actions: possible actions for this agent
    :param lambda_: rationality factor. 0 - random
    :return: action
    """
    distribution = softmax(utilities, lambda_)
    action = possible_actions(np.argmax(np.random.multinomial(1, distribution)))
    return action


class CitizenActions(Enum):
    """
    Actions possible for citizen
    """
    accept_complain = 0
    accept_silent = 1
    reject_complain = 2
    reject_silent = 3


class CopActions(Enum):
    """
    Actions possible for cop
    """
    bribe = 0
    not_bribe = 1


class CopMemoryInitial(Enum):
    """
    The value for the Accepting bribe memory initialization. Depending on how Cop is we have different vals.
    """
    Corrupt = 1.0
    Indifferent = 0.5
    Honest = 0.0


class CitizenMemoryInitial(Enum):
    """
    The value for how succesful is complaining. Depending on the system, citizen will have different experiences. Assuming that if system is honest they had nice memories.
    """
    Corrupt = 0.0
    Indifferent = 0.5
    Honest = 1.0

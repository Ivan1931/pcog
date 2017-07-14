import deps.MDP as MDP
import deps.POMDP as POMDP
from typing import Any
from .humanoid import Humanoid
from envconf import Config, Action, DangerState


def build_model(humanoid):
    # type: (Humanoid) -> Any
    """
    Builds a POMDP model for a given humanoid
    :param humanoid: Humanoid is an object representing a human agents
    perception of it's environment
    :return: model that can be passed on to a POMDP solver
    """
    model = POMDP.model(Config.OBSERVATIONS,
                        Config.STATES,
                        Action.N)
    # TODO Instantiate the POMDP model


def best_action(model):
    """
    Runs a POMCP simulation of our approximate world and finds the best
    available action for the humanoid state
    :param model:
    :return:
    """
    pass

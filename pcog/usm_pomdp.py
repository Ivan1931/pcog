import logging

from .deps import MDP
from .deps import POMDP
from .usm import UtileSuffixMemory
from typing import List
import pickle

logging.basicConfig(filename="pcog.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)


def transition_function(usm):
    # type: (UtileSuffixMemory) -> List[List[List[float]]]
    """
    Method extracts the transition funciton information that we can use to form a POMDP.
    The transition function is a 2D array of probability distributions over the sets of all states.
    :param usm: USM node which we extract the transition function
    :return: 3D transition function array
    """
    if not usm.has_actions():
        raise ValueError("USM does not have an action space")
    transitions = [[[0.0 for s1 in usm.get_states()] for r in usm.get_actions()] for s2 in usm.get_states()]
    for i, s1 in enumerate(usm.get_states()):
        for j, action in enumerate(usm.get_actions()):
            transitions[i][j] = usm.transition_for(s1, action)
    return transitions


def observation_function(usm):
    """
    Observation function of the Utile Suffix Memory.
    Returns a 3D array which is the observation function that allows our POMDP planner to operate. 
    """
    if not usm.has_observations():
        raise ValueError("USM does not have an observation space")
    observations = [[[0.0 for _ in usm.get_observations()] for _ in usm.get_actions()] for _ in usm.get_states()]
    for s, state in enumerate(usm.get_states()):
        for a, action in enumerate(usm.get_actions()):
            observation_dist = usm.observation_for(state=state, action=action)
            if sum(observation_dist) != 1.0:
                pass
            observations[s][a] = observation_dist
    return observations


def reward_function(usm):
    """
    Creates a reward function array from UTile Suffix Memory. 
    We are creating the function:
    R(S1, A, S2) -> Reward of moving from S1 to S2 given action A.
    For USM we ignore the incident state and only care about the start state. 
    """
    rewards = [[[0.0 for s1 in usm.get_states()] for r in usm.get_actions()] for s2 in usm.get_states()]
    for i, state in enumerate(usm.get_states()):
        for j, action in enumerate(usm.get_actions()):
            reward = state.reward(action)
            for k in range(len(usm.get_states())):
                rewards[i][j][k] = reward
    return rewards


def belief_state(usm, past_perceptions):
    leaves = usm.traverse(past_perceptions)
    if len(leaves) == 0:
        # Edge case for when we don't find any related leaves (IE we have like one step that matches nothing )
        logger.info("Belief state could not be derived for: %s", str(past_perceptions))
        p = 1.0 / float(len(usm.get_states()))
        return [p for s in usm.get_states()]
    else:
        p = 1.0 / float(len(leaves))
    belief = [0.0 for s in usm.get_states()]
    for idx, state in enumerate(usm.get_states()):
        if state in leaves:
            belief[idx] = p
    return belief


def _function_sanity_test(func):
    offending = []
    for i, x in enumerate(func):
        for j, y in enumerate(x):
            if sum(y) != 1.0:
                return offending.append((i, j))
    return offending


def build_pomdp_model(usm):
    # type: (UtileSuffixMemory) -> POMDP.Model
    """
    Creates a POMDP model from a Utile Suffix Memory
    :param usm: Utile suffix memory to use in POMDP
    :return: POMDP
    """
    S = len(usm.get_states())
    A = len(usm.get_actions())
    O = len(usm.get_observations())
    reward = reward_function(usm)
    transition = transition_function(usm)
    observation = observation_function(usm)
    logger.info("Reward function:\n{}".format(reward))
    logger.info("Transition function:\n{}".format(transition))
    logger.info("Observation function:\n{}".format(observation))
    model = POMDP.Model(O, S, A)
    model.setRewardFunction(reward)
    model.setTransitionFunction(transition)
    model.setObservationFunction(observation)
    model.setDiscount(usm.gamma)
    return model


def solve(usm, model, past_perceptions, belief_nodes=1000, horizon=10, episolon=0.03):
    solver = POMDP.POMCPModel(model, belief_nodes, 1000, 10000.0)
    S = len(usm.get_states())
    A = len(usm.get_actions())
    O = len(usm.get_observations())
    # policy = POMDP.Policy(S, A, O, solution[1])
    a = solver.sampleAction(belief_state(usm, past_perceptions), horizon)
    return a

def plan_off_usm(usm, last_precept):
    model, _, _, _ = build_pomdp_model(usm)
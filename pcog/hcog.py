from .deps import MDP
from .deps import POMDP


class Attrib(object):
    def __init__(self, identifier, allowed_values):
        self._identifier = identifier
        self._allowed_values = allowed_values

    def get_allowed_values(self):
        return self._allowed_values

    def get_identifier(self):
        return self._identifier


class BasicHybridBuilder(object):
    def __init__(self):
        self._perceptions = None
        self._attributes = None
        self._goals = None
        self._actions = None
        self._observations = None
        self._transitions = None
        self._util = None
        self._desire = None
        self._refocus = None

    def finalize(self):
        for prop in vars(self).values():
            if prop is None:
                raise ValueError("Error: Attempted to build Hybrid agent without specifying {}".format(prop))
        return BasicHybrid(
            attributes=self._attributes,
            goals=self._goals,
            actions=self._actions,
            observations=self._observations,
            transitions=self._transitions,
            util=self._util,
            desire=self._desire,
            refocus=self._refocus
        )

    def util(self, util):
        self._util = util

    def observations(self, observations):
        self._observations = observations

    def goals(self, goals):
        self._goals = goals

    def actions(self, actions):
        self._actions = actions

    def transition(self, transitions):
        self._transitions = transitions

    def desire_fn(self, desire_fn):
        self._desire = desire_fn

    def refocus_fn(self, refocus_fn):
        self._refocus = refocus_fn


class BasicHybrid(object):
    def __init__(self, attributes, goals, actions, observations, transitions, util, desire, refocus):
        self._perceptions = []
        self._attributes = attributes
        self._goals = goals
        self._actions = actions
        self._observations = observations
        self._transitions = transitions
        self._satf = util.get_satif()
        self._pref = util.get_pref()
        self._states = self._derive_states()
        self._refocus = refocus
        self._desire = desire

    def _derive_states(self):
        state_space = self._attributes[0].get_allowed_values()
        for attrib in self._attributes[1:]:
            current_allowed = attrib.get_allowed_values()
            iters = len(state_space)
            idx = 0
            while idx < iters:
                state = state_space[idx]
                for item in current_allowed:
                    new_state = state[::]
                    new_state.append(item)
                    state_space.append(new_state)
                idx += 1
            del state_space[:iters]
        return state_space

    def add_perception(self, perception):
        assert(perception in self._observations)
        self._perceptions.append(perception)

    def _derive_reward(self):
        """
        In this function we will derive the agents reward based on it's intention
        :return: [[[float]]] that represents the reward function that we can then feed into a POMDP
        """
        pass

    def _update_desire(self):
        """
        In this function we will update the agents desire using the formula
        D(g) <- D(g) + 1 - Satf_b(g, B)
        """
        pass

    def _derive_intention(self):
        """
        This function gets the current intention.
        The current intention is the most desired goal
        :return: A state configuration that represents the intention
        """
        pass

    def _refocus(self):
        """
        We need to discuss this
        :return:
        """
        pass

    def derive_pomdp(self):
        pass

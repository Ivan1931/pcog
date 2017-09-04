import logging
from random import choice
from datetime import datetime
import pickle

from .usm import UtileSuffixMemory, Instance
from .envconf import Action
from .perception import perceive, perception_reward, Perceptor
from .usm_draw import update_usm_drawing
from .usm_pomdp import build_pomdp_model, solve

logger = logging.getLogger(__name__)


class ModelLearnAgent(object):
<<<<<<< HEAD
    def __init__(self, usm=None, perceptor=Perceptor, max_exploration_iterations=40, perceptive_window=2):
        # type: (UtileSuffixMemory, Perceptor, int, int) -> None
=======
    def __init__(self,
                 usm=None,
                 perceptor=Perceptor,
                 max_exploration_iterations=60,
                 perceptive_window=2,
                 save_perceptions=False):
        # type: (UtileSuffixMemory, Perceptor, int, int, bool) -> None
>>>>>>> improved_logging
        if usm:
            self.usm = usm
        else:
            self.usm = UtileSuffixMemory()
        self.max_exploration_iterations = max_exploration_iterations
        self._iterations = 0
        self._perceptions = []
        self._actions = []
        self._rewards = []
        self._perception_window = perceptive_window
        self._perception_factory = perceptor
        self.model = None
<<<<<<< HEAD
=======
        self._should_save_perceptions = save_perceptions
        self._has_saved_perceptions = False
>>>>>>> improved_logging
        usm._observation_space = self.possible_observations()

    def possible_observations(self):
        return self._perception_factory.possible_observations()

    def should_explore(self):
        return self._iterations < self.max_exploration_iterations

    def visualise(self):
        update_usm_drawing(self.usm)

    def add_perception(self, perception):
        self._perceptions.append(perception)
        if 1 < len(self._perceptions):
<<<<<<< HEAD
            perceptor = self._perception_factory(self._perceptions[-1], self._perceptions[-2])
=======
            perceptor = self._perception_factory(
                current=self._perceptions[-1],
                previous=self._perceptions[-2]
            )
>>>>>>> improved_logging
            action = self._actions[-1]
            reward = perceptor.perception_reward(action)
            observation = perceptor.perception_observation()
            self._rewards.append(reward)
            self.usm.insert(
                Instance(
                    observation=observation,
                    action=action,
                    reward=reward,
                )
            )
            logger.info(" Inserted perceptions: %s action: %s reward: %s into learning agent "
                        , str(perception)
                        , str(action)
                        , str(reward))
        self._iterations += 1

    def _repeating_actions(self):
        if 0 == len(self._actions):
            return True
        else:
            return all([a == self._actions[0] for a in self._actions[-7:]])

    def _save_perceptions(self):
        file_name = "perceptions-of-{}".format(datetime.now())
        with open(file_name, 'w') as save_file:
            logger.info("Saving perceptions to %s", file_name)
            pickle.dump(self._perceptions, save_file)
            self._has_saved_perceptions = True

    def get_decision(self):
        if self.max_exploration_iterations < self._iterations and not self._repeating_actions():
            if not self._has_saved_perceptions and self._should_save_perceptions:
                self._save_perceptions()
            if not self.model:
                self.model = build_pomdp_model(self.usm)
            if len(self.usm.get_instances()) == 0:
                raise ValueError("Attempting to plan with a model that has no perceptions")
            logger.info("Making decision No. %d with POMDP model", self._iterations)
            action = solve(self.usm,
                           self.model,
                           self.usm.get_instances()[-self._perception_window:])
            self._actions.append(action)
            return action
        else:
            if 1 < len(self._perceptions):
                logger.info("Making decision using smart exploration")
                perceptor = self._perception_factory(self._perceptions[-1], self._perceptions[-2])
                action = perceptor.smart_explore()
            else:
                logger.info("Making decision No. %d with random action", self._iterations)
                # Bellow is a hack to get the agent to spend more time choosing the explore action than before
                actions_to_choose = self.usm.get_actions() + [Action.EXPLORE]
                action = choice(actions_to_choose)
            self._actions.append(action)
            return action

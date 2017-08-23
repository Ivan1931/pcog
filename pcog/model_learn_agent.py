import logging
from random import choice

from .usm import UtileSuffixMemory, Instance
from .perception import perceive, perception_reward
from .usm_draw import update_usm_drawing
from .usm_pomdp import build_pomdp_model, solve

logger = logging.getLogger(__name__)

class ModelLearnAgent(object):
    def __init__(self, 
                usm=None,
                max_exploration_iterations=30,
                perceptive_window=2):
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
        self.model = None


    def should_explore(self):
        return self._iterations < self.max_exploration_iterations


    def visualise(self):
        update_usm_drawing(self.usm)


    def add_perception(self, perception):
        self._perceptions.append(perception)
        if 1 < len(self._perceptions):
            reward = perception_reward(self._perceptions[-1], 
                                       self._perceptions[-2])
            action = self._actions[-1]
            self._rewards.append(reward)
            self.usm.insert(
                Instance(
                    observation=perceive(self._perceptions[-1], 
                                         self._perceptions[-1]),
                    action=action,
                    reward=reward,
                )
            )
            logger.info(
                """
                Inserted perceptions: %s
                action: %s
                reward: %s 
                into learning agent
                """.replace("\n", " ").strip()
                , str(perception)
                , str(action)
                , str(reward)
            )
        self._iterations += 1

    
    def get_decision(self):
        if self.max_exploration_iterations < self._iterations:
            if not self.model:
                self.model = build_pomdp_model(self.usm)
            return solve(self.usm, 
                         self.model, 
                         self.usm.get_instances()[-self._perception_window:])
        else:
            action = choice(self.usm.get_actions())
            self._actions.append(action)
            return action
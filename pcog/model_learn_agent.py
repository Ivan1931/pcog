import logging

from .usm import UtileSuffixMemory, Instance
from .perception import perceive
from .usm_draw import update_usm_drawing
from .usm_pomdp import build_pomdp_model, solve

logger = logging.getLogger(__name__)

class ModelLearnAgent(object):
    def __init__(self, 
                usm=None,
                max_exploration_iterations=30):
        if usm:
            self.usm = usm
        else:
            self.usm = UtileSuffixMemory()

        self.max_exploration_iterations = max_exploration_iterations
        self.iterations = 0
        self.perceptions = []
        self.model = None


    def should_explore(self):
        return self.iterations < self.max_exploration_iterations


    def visualise(self):
        update_usm_drawing(self.usm)


    def add_perception(self, perception, action, reward):
        self.perceptions.append(perception)
        self.iterations += 1
        self.usm.insert(
            Instance(
                observation=perception,
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

    
    def get_decision(self):
        if self.model is None:
            self.model = build_pomdp_model(self.usm)
        return solve(self.usm, self.model)
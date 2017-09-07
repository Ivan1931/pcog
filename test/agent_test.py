import unittest
from json import loads
from bunch import bunchify
from pcog.agent import simulate, GridAgent
from pcog.envconf import Action

humoid_json = """
{
      "lastWolfPosition": [29.02450942993164,0.0,27.91568946838379],
      "lastFoodPosition":[0.0,0.0,0.0],
      "position":[29.140689849853516,0.0,26.8988094329834],
      "entityId":0,
      "stamina":100.0,
      "health":7.0
}
"""
humanoid_null_wolf = """
{
    "position":[5.669702053070068,0.0,1.7854390144348145],
    "entityId":0,
    "stamina":100.0,
    "health":10.0,
    "lastWolfPosition":null,
    "lastFoodPosition":null
}
"""
class BasicStressTest(unittest.TestCase):
    def test_solution(self):
        action = simulate(humoid_json)
        self.assertIn(action, Action.SET, msg="The action selected was {}".format(action))
    def test_null_wolf(self):
        action = simulate(humanoid_null_wolf)
        self.assertIn(action, Action.SET, msg="The action selected was {}".format(action))


class GridAgentTest(unittest.TestCase):
    def test_construction(self):
        agent = GridAgent()
        agent.derive_model()
        observation = bunchify({
            "position": [2.0, 0.0, 1.0],
            "health": 10.0,
            "lastWolfPosition": [1.0, 0.0, 1.0],
            "lastFoodPosition": [10.0, 0.0, 10.0],
        })
        agent.update_belief(observation)
        action = agent.plan()
        self.assertEqual(action, Action.ATTACK)

if __name__ == "__main__a":
    unittest.main()

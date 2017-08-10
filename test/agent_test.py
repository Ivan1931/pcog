import unittest
from json import loads
import pcog
from pcog.agent import simulate
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

if __name__ == "__main__a":
    unittest.main()

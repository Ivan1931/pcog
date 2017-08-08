from typing import Tuple, Optional
from json import loads
from .envconf import HealthState, DangerObservation

import math

def dist(a, b):
    return math.sqrt(math.sqrt(a[0] - b[0]) +
                     math.sqrt(a[1] - b[1]) +
                     math.sqrt(a[2] - b[2]))

class Humanoid(object):
    """
    Humanoid is a python class that wraps around json data
    that describes the current state of the agent in the QCOG test bed environment.

    An example of the json data used to create this class.

    {
      "lastWolfPosition": [29.02450942993164,0.0,27.91568946838379],
      "lastFoodPosition":[0.0,0.0,0.0],
      "position":[29.140689849853516,0.0,26.8988094329834],
      "entityId":0,
      "stamina":100.0,
      "health":7.0
    }
    """
    def __init__(self, position, health, stamina):
        # type: (Tuple[float, float, float] , float, float) -> None
        """

        :param entity_id: Agents id - we may need this
        :param position: Agents position - discertised for the POMDP
        :param health: Agents health - we discretise this for the POMDP
        :param stamina: Agents stamina - discretised for the POMDPO
        """
        self.position = position
        self.health = health
        self.stamina = stamina
        self.wolf_position = None  # type: Optional[Tuple[float, float, float]]
        self.food_position = None  # type: Optional[Tuple[float, float, float]]

    def get_danger_level(self):
        if self.wolf_position is None:
            return DangerObservation.UNKNOWN
        d = dist(self.wolf_position, self.position)
        if 7.0 < d:
            return DangerObservation.FAR
        elif 4.0 < d < 7.0:
            return DangerObservation.CLOSE
        else:
            return DangerObservation.UNDER_ATTACK

    def get_health_state(self):
        if 7.0 < self.health:
            return DangerObservation.GOOD
        elif 4.0 < self.health < 7.0:
            return DangerObservation.AVERAGE
        else:
            return DangerObservation.BAD

    def get_hunger_level(self):
        pass

    @staticmethod
    def from_dict(data):
        # type: (dict) -> Humanoid
        """
        Correct types are assumed in the dictionary
        :param data: dict containing fields of the humoid
        :return: a Humanoid derived from the dict
        """
        x, y, z = data["position"]
        humanoid = Humanoid((x, y, z), data["health"], data["stamina"])
        humanoid.wolf_position = tuple(data["lastWolfPosition"])
        humanoid.food_position = tuple(data["lastFoodPosition"])
        return humanoid

    @staticmethod
    def from_json(json_string):
        # type: (str) -> Humanoid
        """

        :param json_string: Json string describing the various parameters of the game state
        :return: Humanoid based on the game state
        """
        data = loads(json_string)
        if type(data) == dict:
            return Humanoid.from_dict(data)
        else:
            raise ValueError("Error, unrecognised json parse type")

from typing import Tuple, Optional
from json import loads


class Humanoid(object):
    def __init__(self, entity_id, position, health, stamina):
        # type: (int, Tuple[float, float, float] , float, float) -> None
        """

        :param entity_id: Agents id - we may need this
        :param position: Agents position - discertised for the POMDP
        :param health: Agents health - we discretise this for the POMDP
        :param stamina: Agents stamina - discretised for the POMDPO
        """
        self.entity_id = entity_id
        self.position = position
        self.health = health
        self.stamina = stamina
        self.wolf_position = None  # type: Optional[Tuple[float, float, float]]
        self.food_position = None  # type: Optional[Tuple[float, float, float]]

    @staticmethod
    def from_dict(data):
        # type: (dict) -> Humanoid
        """
        Correct types are assumed in the dictionary
        :param data: dict containing fields of the humoid
        :return: a Humanoid derived from the dict
        """
        x, y, z = data["position"]
        return Humanoid(data["id"], (x, y, z), data["health"], data["stamina"])

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
        if type(data) == list:
            humanoid = Humanoid.from_dict(data["humanoid"])
            x, y, z = data["wolf"]
            humanoid.wolf_position = (float(x), float(y), float(z))

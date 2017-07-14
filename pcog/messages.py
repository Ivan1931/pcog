from typing import List


class MessageType(object):
    EXIT = -1
    POSITION_UPDATE = 0
    HEALTH_UPDATE = 1

    ENTITY_WATER_SOURCE = 2
    ENTITY_PREDATOR = 3
    ENTITY_PREY = 4
    WORLD_ENTITY = 15

    NEW_PERCEIVED_AGENT = 5
    REMOVE_PERCEIVED_AGENT = 6

    STAMINA_UPDATE = 7
    HUNGER_UPDATE = 8
    THIRST_UPDATE = 9

    INVENTORY_UPDATE = 10
    ITEM_EDIBLE = 11
    NON_EDIBLE_ITEM = 12

    SOUND_HEARD = 13
    DAMAGE_TAKEN = 14
    FACING_DIRECTION_UPDATE = 16
    REINFORCEMENT = 17
    RESTART = 18
    TEMPERATURE_UPDATE = 19

    NOOP = 1000  # The dummy message of the super class


class Message(object):
    def __init__(self, message_text):
        # type: (str) -> None
        """
        Creates a new message from some input string
        :param message_text: str text of the message received
        """
        self.message_raw = Message.parse_message(message_text)
        self.id = int(self.message_raw[0])

    def get_id(self):
        return self.id

    def get_type(self):
        raise NotImplementedError("Message type for base message class not implemented")

    @staticmethod
    def parse_message(message_text):
        # type: (str) -> List[str]
        """
        Parses message text into a list of strings.
        Also trims their output
        :param message_text:
        :return:
        """
        parsed = []
        for m in message_text.split("$"):
            parsed.append(m.strip())
        return parsed


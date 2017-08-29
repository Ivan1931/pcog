from random import choice


class Change(object):
    N = 3
    LESS, SAME, MORE = range(N)
    SET = set(range(N))


class Action(object):
    """
    Represents the possible policies that the agent can execute.
    They are listed bellow along with their action ID codes given by QCog:

    FLEE = 8
    ATTACK = 9
    EXPLORE = 0
    EAT = 6
    """
    N = 3
    EXPLORE, ATTACK, EAT = range(N)
    SET = set(range(N))

    @staticmethod
    def qcog_action(a):
        if a == Action.ATTACK:
            return 9
        if a == Action.EXPLORE:
            return 0
        if a == Action.EAT:
            return 6
        raise ValueError("Unrecognised action")

    @staticmethod
    def qcog_action_name(a):
        if a == 8:
            return "FLEE"
        elif a == 9:
            return "ATTACK"
        elif a == 0:
            return "EXPLORE"
        elif a == 6:
            return "EAT"
        raise ValueError("Unkown action")

    @staticmethod
    def action_name(a):
        if a == Action.ATTACK:
            return "ATTACK"
        elif a == Action.EXPLORE:
            return "EXPLORE"
        elif a == Action.EAT:
            return "EAT"
        raise ValueError("Unknown action")

    @staticmethod
    def random_action():
        """
        Chooses a random action from set of available actions
        """
        return choice(list(Action.SET))

    @staticmethod
    def valid_action(a):
        return a in Action.SET


class HealthObservation(object):
    N = 3
    GOOD, OK, BAD = range(N)
    SET = set(range(N))

    @staticmethod
    def health_level(health):
        # type: (float) -> int
        """
        From an integer returns the current health level of the agent
        :param health: Current agent health level
        :return: discretized health level
        """
        if 7.0 < health:
            return HealthObservation.GOOD
        elif 4.0 < health < 7.0:
            return HealthObservation.OK
        else:
            return HealthObservation.BAD


class MovementObservation(object):
    N = 3
    STATIONARY, FAR, SUPER_FAR = range(N)
    SET = set(range(N))

    @staticmethod
    def movement_level(movement):
        if 4.0 < movement:
            return MovementObservation.SUPER_FAR
        elif 1.0 < movement <= 4.0:
            return MovementObservation.FAR
        else:
            return MovementObservation.STATIONARY


class FoodObservation(object):
    N = 4
    UNKNOWN, FAR, NEAR, VERY_CLOSE = range(N)
    SET = set(range(N))


class DistanceObservation(object):
    N = 4
    UNKNOWN, FAR, CLOSE, VERY_CLOSE = range(N)
    SET = set(range(N))

    @staticmethod
    def proximity_level(distance):
        if 15.0 < distance:
            proximity = DistanceObservation.UNKNOWN
        elif 5.0 <= distance <= 15.0:
            proximity = DistanceObservation.FAR
        elif 2.0 < distance <= 5.0:
            proximity = DistanceObservation.CLOSE
        else:
            proximity = DistanceObservation.VERY_CLOSE
        return proximity

    @staticmethod
    def as_string(o):
        if o == DistanceObservation.UNKNOWN:
            return "UNKNOWN"
        elif o == DistanceObservation.FAR:
            return "FAR"
        elif o == DistanceObservation.CLOSE:
            return "CLOSE"
        elif o == DistanceObservation.VERY_CLOSE:
            return "UNDER_ATTACK"


class DangerState(object):
    N = 4

    LOW, MEDIUM, HIGH, SEVERE = range(N)

    SET = set(range(N))


class StaminaState(object):
    N = 3
    GOOD, AVERAGE, BAD = range(N)
    SET = set(range(N))

    @staticmethod
    def valid_state(s):
        return s in StaminaState.SET

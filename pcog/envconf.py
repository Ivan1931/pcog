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
    FLEE, EXPLORE, ATTACK = range(N)
    SET = set(range(N))

    @staticmethod
    def qcog_action(a):
        if a == Action.FLEE:
            return 8
        if a == Action.ATTACK:
            return 9
        if a == Action.EXPLORE:
            return 0
        raise ValueError("Unrecognised action")

    @staticmethod
    def action_name(a):
        if a == Action.FLEE:
            return "FLEE"
        elif a == Action.ATTACK:
            return "ATTACK"
        elif a == Action.EXPLORE:
            return "EXPLORE"
        raise ValueError("Unknown action")


    @staticmethod
    def valid_action(a):
        return a in Action.SET


class HealthObservation(object):
    N = 3
    GOOD, OK, BAD = range(N)
    SET = set(range(N))


class FoodObservation(object):
    N = 4
    UNKNOWN, FAR, NEAR, VERY_CLOSE = range(N)
    SET = set(range(N))


class WolfObservation(object):
    N = 4
    UNKNOWN, FAR, CLOSE, UNDER_ATTACK = range(N)
    SET = set(range(N))


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

class Action(object):
    FLEE = 8
    ATTACK = 9
    EXPLORE = 0
    EAT = 6

    SET = set([
            FLEE,
            ATTACK,
            EXPLORE,
            EAT,
        ])

    N = len(SET)

    @staticmethod
    def valid_action(a):
        return a in Action.SET


class HealthState(object):
    GOOD = 1
    AVERAGE = 2
    BAD = 3

    SET = set([
            GOOD,
            BAD,
            AVERAGE
        ])

    N = len(SET)

    @staticmethod
    def valid_state(h):
        return h in HealthState.SET

class FoodObservation(object):
    FAR = 1
    NEAR = 2
    VERY_CLOSE = 3
    UNKNOWN = 4

    SET = set([
        FAR,
        NEAR,
        VERY_CLOSE,
        UNKNOWN,
    ])

    N = len(SET)

class DangerObservation(object):
    FAR = 1
    CLOSE = 2
    UNDER_ATTACK = 3
    UNKNOWN = 4

    SET = set([
        FAR,
        CLOSE,
        UNDER_ATTACK,
        UNKNOWN,
    ])

    N = len(SET)

    @staticmethod
    def valid_state(h):
        return h in DangerObservation.SET


class StaminaState(object):
    GOOD = 1
    AVERAGE = 2
    BAD = 3

    SET = set([
        GOOD,
        BAD,
        AVERAGE
    ])

    N = len(SET)

    @staticmethod
    def valid_state(s):
        return s in StaminaState.SET


class Config(object):
    STATES = StaminaState.N * HealthState.N * DangerObservation.N
    OBSERVATIONS = 2

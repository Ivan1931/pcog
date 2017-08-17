from .envconf import HealthObservation, DangerState, Action, WolfObservation
from .humanoid import Humanoid
from json import loads
from .deps import MDP
from .deps import POMDP

import logging

logging.basicConfig(filename="pcog.log", filemode="w", level=logging.INFO)

def oi(wolf_proximity, health_observation):
    return wolf_proximity * HealthObservation.N + health_observation


def danger(s, w, h, a):
    d = (w + h + a) / (WolfObservation.N + HealthObservation.N + Action.N)
    if 0.0 < d < 0.25:
        if s == DangerState.LOW:
            return 1.0
        elif s == DangerState.MEDIUM:
            return 0.2
        elif s == DangerState.HIGH:
            return 0.1
        else:
            return 0.05
    elif 0.25 < d < 0.5:
        if s == DangerState.LOW:
            return 0.2
        elif s == DangerState.MEDIUM:
            return 0.8
        elif s == DangerState.HIGH:
            return 0.2
        else:
            return 0.1
    elif 0.5 < d < 0.75:
        if s == DangerState.LOW:
            return 0.1
        elif s == DangerState.MEDIUM:
            return 0.3
        elif s == DangerState.HIGH:
            return 0.3
        else:
            return 0.2
    else:
        if s == DangerState.LOW:
            return 0.05
        elif s == DangerState.MEDIUM:
            return 0.3
        elif s == DangerState.HIGH:
            return 0.3
        else:
            return 0.7


def find_transition(sf, a, si):
    d = (si - sf) / DangerState.N
    if 0.0 < d: # Chance of moving high danger to low danger
        if a == Action.FLEE:
            x = 0.2
        elif a == Action.EXPLORE:
            x = 0.5
        else:
            x = 0.1
        return x / float(d)
    elif d == 0:
        if a == Action.EXPLORE:
            return 0.5
        else:
            return 0.1
    else: # Chance of moving from low danger to high danger
        if a == Action.FLEE:
            x = 0.1
        elif a == Action.EXPLORE:
            x = 0.5
        else:
            x = 1.0
        return x / float(abs(d))


def find_reward(sf, a, si):
    if si < sf: # Penalise transitioning to higher danger
        return -10.0
    elif si == sf: # Reward staying in the same state
        return 5.0
    else: # Heavily reward reducing danger level
        return 20.0


def make_pcog_simulation():
    # Actions are: 0-listen, 1-open-left, 2-open-right
    S = DangerState.N
    A = Action.N
    O = WolfObservation.N * HealthObservation.N
    model = POMDP.Model(O, S, A)
    transitions = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    rewards = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    observations = [[[0 for x in xrange(O)] for y in xrange(A)] for k in xrange(S)]
    for w in xrange(WolfObservation.N):
        for h in xrange(HealthObservation.N):
            o = oi(w, h)
            for a in xrange(A):
                for s in xrange(S):
                    try:
                        d = danger(s, w, h, a)
                        observations[s][a][o] = d
                    except IndexError as e:
                        print("{} {}".format(w, h))
                        print("{} {} {}".format(s, a, o))
                        print("{} {} {}".format(S, A, O))
                        raise e

    for a in xrange(A):
        for s in xrange(S):
            total_o = 0.0
            for o in xrange(O):
                total_o += observations[s][a][o]
            for o in xrange(O):
                observations[s][a][o] /= total_o

    total_t = 0.0
    for si in xrange(DangerState.N):
        for a in xrange(Action.N):
            for sf in xrange(DangerState.N):
                t = find_transition(sf, a, si)
                transitions[sf][a][si] = t
                total_t += t

    for sf in xrange(DangerState.N):
        for a in xrange(Action.N):
            total_s = 0.0
            for si in xrange(DangerState.N):
                total_s += transitions[sf][a][si]
            for si in xrange(DangerState.N):
                transitions[sf][a][si] /= total_s

    for si in xrange(DangerState.N):
        for a in xrange(Action.N):
            for sf in xrange(DangerState.N):
                rewards[sf][a][si] = find_reward(sf, a, si)


    model.setRewardFunction(rewards)
    model.setObservationFunction(observations)
    model.setTransitionFunction(transitions)
    return model, observations, transitions, rewards


def belief_state(humanoid, observations):
    health = humanoid.get_health()
    wolf = humanoid.get_wolf_proximity()
    belief = [0.0 for s in xrange(DangerState.N)]
    for s in xrange(DangerState.N):
        belief[s] += observations[s][Action.EXPLORE][oi(wolf, health)]
    total_b = sum(belief)
    return map(lambda state: state / total_b, belief)


def run_pcog_simulation(humanoid):
    model, observations, transitions, rewards = make_pcog_simulation()
    model.setDiscount(0.95)
    horizon = 10 # 10 seconds in real time
    solver = POMDP.IncrementalPruning(horizon, 0.0)
    solution = solver(model)
    policy = POMDP.Policy(DangerState.N, Action.N, HealthObservation.N * WolfObservation.N, solution[1])
    b = belief_state(humanoid, observations)
    a, ID = policy.sampleAction(b, horizon)
    return a


def simulate(humanoid_string):
    humanoid = Humanoid.from_json(humanoid_string)
    action = run_pcog_simulation(humanoid)
    return Action.qcog_action(action)

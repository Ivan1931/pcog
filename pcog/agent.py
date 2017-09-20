from .envconf import HealthObservation, DangerState, Action, DistanceObservation
from .humanoid import Humanoid
from .deps import MDP
from .deps import POMDP
from .usm import normalise_to_one

import logging

logging.basicConfig(filename="pcog.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)


class GridAgent(object):
    def __init__(self):
        self.width_blocks = 5
        self.height_blocks = 5
        self.grid_width = 30.0
        self.grid_height = 30.0
        self.L = self.height_blocks * self.width_blocks + 1
        self.number_wolf = 1
        self.number_food = 1
        self.health_level = HealthObservation.GOOD
        self.UNKNOWN = None
        self.model = None
        self.gamma = 0.3
        self.states = self._get_all_states()
        self.belief = normalise_to_one([1.0 / len(self.states) for _ in self.states])

    def get_states(self):
        pass

    def get_rewards(self):
        pass

    def _grid_coordinates(self):
        return [(x, y) for x in range(self.height_blocks) for y in range(self.height_blocks)]

    def _idx(self, humanoid, wolf, food, health):
        if food is None:
            food_idx = self.height_blocks * self.width_blocks
        else:
            food_x, food_y= food
            food_idx = food_x * self.width_blocks + food_y
        if wolf is None:
            wolf_idx = self.width_blocks * self.height_blocks
        else:
            wolf_x, wolf_y = wolf
            wolf_idx = wolf_x * self.width_blocks + wolf_y
        humanoid_x, humanoid_y = humanoid
        humanoid_idx = humanoid_x * self.width_blocks + humanoid_y
        return humanoid_idx * self.L * self.L * HealthObservation.N + wolf_idx * self.L * HealthObservation.N + food_idx * HealthObservation.N + health

    def _get_all_states(self):
        coordinates = self._grid_coordinates()
        coordinates.append(None)
        return [(humanoid, wolf, food, health)
                for humanoid in coordinates
                for wolf in coordinates
                for food in coordinates
                for health in HealthObservation.SET]

    @staticmethod
    def _is_neighbour(a, b):
        if a is None or b is None:
            return False
        ax, ay = a
        bx, by = b
        return abs(ax - bx) <= 1 or abs(ay - by) <= 1

    def get_transitions(self):
        """
        T : S x A -> S
        :return:
        """
        N = self.L * self.L * self.L * HealthObservation.N
        states = self._get_all_states()
        T = [[[0.0 for _ in range(N)] for _ in Action.N] for _ in range(N)]
        for statei in states:
            for action in Action.SET:
                for statef in states:
                    humanoidi, wolfi, foodi, healthi = statei
                    humanoidf, wolff, foodf, healthf = statef
                    iidx = self._idx(humanoidi, wolfi, foodi, healthi)
                    fidx = self._idx(humanoidf, wolff, foodf, healthf)
                    # food position is immutable
                    if foodf != foodi:
                        T[iidx][action][fidx] = 0.0
                        break
                    if action == Action.EXPLORE:
                        if wolff == wolfi:
                            T[iidx][action][fidx] += 1.0
                        if self._is_neighbour(wolfi, wolff):
                            T[iidx][action][fidx] += 1.0
                        if self._is_neighbour(humanoidi, humanoidf):
                            T[iidx][action][fidx] += 1.0
                        if humanoidi == humanoidf:
                            T[iidx][action][fidx] += 2.0
                    elif action == Action.ATTACK:
                        # Attack and wolf neighbour - more likely to be in wolf position
                        if wolff == wolfi:
                            T[iidx][action][fidx] += 1.0
                        if self._is_neighbour(wolfi, wolff):
                            T[iidx][action][fidx] += 1.0
                        if self._is_neighbour(humanoidi, wolfi) and wolff == humanoidf:
                            T[iidx][action][fidx] += 2.0
                    elif action == Action.EAT:
                        # Eat and food position is neighbour more likely to be in food position
                        if wolff == wolfi:
                            T[iidx][action][fidx] += 1.0
                        if self._is_neighbour(wolfi, wolff):
                            T[iidx][action][fidx] += 1.0
                        if self._is_neighbour(humanoidi, foodi) and foodf == humanoidi:
                            T[iidx][action][fidx] += 2.0
        for transition_state in T:
            for state_action in transition_state:
                t_norm = self.normalise(state_action)
                for i in len(state_action):
                    state_action[i] = t_norm[i]
        return T

    @staticmethod
    def normalise(array):
        total = sum(array)
        return normalise_to_one([i / total for i in array])

    def make_value_discrete(self, value):
        return int(self.width_blocks * value / self.grid_width)

    def make_discrete(self, data):
        if data is None:
            return self.UNKNOWN
        if len(data) == 3:
            x = data[0]
            z = data[2]
        elif len(data) == 2:
            x = data[0]
            z = data[1]
        else:
            raise ValueError("{data} does not have 2 or 3 arguments".format(data))
        x = self.make_value_discrete(x)
        z = self.make_value_discrete(z)
        return x, z

    def derive_observation(self, humanoid_state):
        # Wolf state
        wolf = self.make_discrete(humanoid_state.wolf)
        position = self.make_discrete(humanoid_state.position)
        food = self.make_discrete(humanoid_state.food)
        health = HealthObservation.health_level(humanoid_state.health)
        return position, wolf, food, health

    def get_observations(self):
        states = self._get_all_states()
        O = [[[0.0 for _ in states] for a in Action.SET] for _ in states]
        for state in states:
            for action in Action.SET:
                for obs in states:
                    humanoid, wolf, food, health = state
                    idx = self._idx(humanoid, wolf, food, health)
                    ohumanoid, owolf, ofood, ohealth = obs
                    oidx = self._idx(ohumanoid, owolf, ofood, ohealth)
                    if owolf == self.UNKNOWN or ofood == self.UNKNOWN:
                        O[idx][action][oidx] = len(states) / 1.0
                    if ohumanoid == humanoid:
                        O[idx][action][oidx] += 1.0
                    if food == ofood:
                        O[idx][action][oidx] += 1.0
                    if wolf == owolf:
                        O[idx][action][oidx] += 1.0
                    if ohealth == health:
                        O[idx][action][oidx] += 1.0
        for observation_state in O:
            for state_action in observation_state:
                o_norm = self.normalise(state_action)
                for i in len(state_action):
                    state_action[i] = o_norm[i]
        return O

    def update_belief(self, observation):
        humanoid, wolf, food, health = observation
        for idx in range(len(self.states)):
            shumanoid, swolf, sfood, shealth = observation
            if shumanoid == humanoid:
                self.belief[idx] += 1.0
            if swolf == wolf:
                self.belief[idx] += 1.0
            if sfood == food:
                self.belief[idx] += 1.0
            if shealth == health:
                self.belief[idx] += 1.0
        self.belief = self.normalise(self.belief)

    def construct_reward(self):
        states = self._get_all_states()
        R = [[[0.0 for _ in states] for a in Action.SET] for _ in states]
        for statei in states:
            for action in Action.SET:
                for statef in states:
                    humanoidi, wolfi, foodi, healthi = statei
                    humanoidf, wolff, foodf, healthf = statef
                    iidx = self._idx(humanoidi, wolfi, foodi, healthi)
                    fidx = self._idx(humanoidf, wolff, foodf, healthf)
                    if action == Action.ATTACK:
                        if humanoidf == wolff:
                            R[iidx][action][fidx] += 2.0
                    elif action == Action.EAT:
                        if humanoidf == foodf:
                            R[iidx][action][fidx] += 4.0
                    elif action == Action.EXPLORE:
                        R[iidx][action][fidx] = 0.2
        return R

    def set_reward(self, reward_fn):
        pass

    def derive_model(self):
        logger.info("Deriving POMDP model")
        import ipdb; ipdb.set_trace()
        rewards = self.construct_reward()
        transitions = self.get_transitions()
        observations = self.get_observations()
        S = len(self._get_all_states())
        O = len(self._get_all_states())
        A = len(Action.SET)
        model = POMDP.Model(O, S, A)
        model.setRewardFunction(rewards)
        model.setTransitionFunction(transitions)
        model.setObservation(observations)
        model.setDiscount(self.gamma)
        self.model = model

    def plan(self):
        if self.model is not None:
            solver = POMDP.POMCPModel(self.model, 1000, 10, 1000.0)
            action = solver.sampleAction(self.belief, 10)
            logger.info("Manual POMDP agent chose action %s",
                        Action.qcog_action_name(action))
            return action

class Hybrid(object):
    @staticmethod
    def _state_space():
        return [(food_seen, enemy_seen, health, danger_level)
                for food_seen in [True, False]
                for enemy_seen in [True, False]
                for health in HealthObservation.SET
                for danger_level in DangerState.SET]

    def __init__(self, is_hybrid=False):
        self._states = Hybrid._state_space()
        self._intention = None

    def plan(self):
        pass

    def update_belief(self):
        pass

    def derive_model(self):
        pass

    def derive_transitions(self):
        pass

    def derive_reward(self):
        pass

    def derive_observations(self):
        pass

    def set_belief(self):
        pass

    def set_rewards(self):
        pass

def oi(wolf_proximity, health_observation):
    return wolf_proximity * HealthObservation.N + health_observation


def danger(s, w, h, a):
    d = (w + h + a) / (DistanceObservation.N + HealthObservation.N + Action.N)
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
    O = DistanceObservation.N * HealthObservation.N
    model = POMDP.Model(O, S, A)
    transitions = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    rewards = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    observations = [[[0 for x in xrange(O)] for y in xrange(A)] for k in xrange(S)]
    for w in xrange(DistanceObservation.N):
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
    policy = POMDP.Policy(DangerState.N, Action.N, HealthObservation.N * DistanceObservation.N, solution[1])
    b = belief_state(humanoid, observations)
    a, ID = policy.sampleAction(b, horizon)
    return a


def simulate(humanoid_string):
    humanoid = Humanoid.from_json(humanoid_string)
    action = run_pcog_simulation(humanoid)
    return Action.qcog_action(action)
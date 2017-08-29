from json import loads
import math
from bunch import bunchify
from .envconf import Change, DistanceObservation, HealthObservation, Action, MovementObservation
from .humanoid import dist


def process(state_perception_string):
    return bunchify(loads(state_perception_string))


def _change(current, previous):
    current_distance = 0.0
    for v in current.predators:
        current_distance += dist(v, current.position)
    if 0.0 < len(current.predators): 
        current_distance /= float(len(current.predators))
    previous_distance = 0.0
    for v in previous.predators:
        previous_distance += dist(v, previous.position)
    return (
        current.health - previous.health,
        current_distance - previous_distance,
        len(current.predators) - len(previous.predators)
    )


def perceive(current, previous):
    """
    This method takes the previous known state of the world and the current known state
    and compares the difference between them and creates a discetised difference between them
    :param current: The most recent state
    :param previous: The state recorded before it
    :return: A perceptive difference between the state
    """
    if current.health < previous.health:
        health = Change.LESS
    elif current.health == previous.health:
        health = Change.SAME
    else:
        health = Change.MORE
    current_distance = 0.0
    for v in current.predators:
        current_distance += dist(v, current.position)
    if 0.0 < len(current.predators):
        current_distance /= float(len(current.predators))
    previous_distance = 0.0
    for v in previous.predators:
        previous_distance += dist(v, previous.position)
    if 0.0 < len(previous.predators):
        previous_distance /= float(len(previous.predators))

    if current_distance < previous_distance:
        distance = Change.LESS
    elif previous_distance < current_distance:
        distance = Change.MORE
    else:
        distance = Change.SAME

    if len(current.predators) < len(previous.predators):
        predators = Change.LESS
    elif len(current.predators) == len(previous.predators):
        predators = Change.SAME
    else:
        predators = Change.MORE
    return distance, health, predators


def sigmoid(x):
    return math.exp(x) / (math.exp(x) + 1.0)

def perception_reward(current, previous, conf=dict(max_health=10.0)):
    health_change, distance_change, visible_predators_change = _change(current, previous)
    return health_change / conf["max_health"] + sigmoid(distance_change) + sigmoid(visible_predators_change)


class Perceptor(object):
    def __init__(self, current, previous):
        self.current = current
        self.previous = previous

    @staticmethod
    def possible_observations():
        return [(w, f) for w in DistanceObservation.SET
                          for f in DistanceObservation.SET]

    def perception_reward(self, action):
        stationary_penalty = -0.5
        wolf_proximity, food_proximity, health, movement = self._perception_observation()
        if action == Action.FLEE:
            if DistanceObservation.CLOSE == wolf_proximity and HealthObservation.OK < health:
                return 10.0
            else:
                return stationary_penalty
        elif action == Action.EXPLORE:
            # Punish agent for loosing health during exploration
            if self.current.health < self.previous.health:
                return -10.0
            elif DistanceObservation.CLOSE <= food_proximity and health < HealthObservation.OK:
                # Punish agent for exploring when close to food and with less than good health
                return -10.0
            else:
                return 0.5
        elif action == Action.ATTACK:
            if self.current.health < self.previous.health:
                return 10.0
            else:
                return stationary_penalty
        else:
            # action == Action.EAT
            if self.previous.health < self.current.health and DistanceObservation.CLOSE <= food_proximity:
                return 20.0
            elif self.current.health < self.previous.health or food_proximity < DistanceObservation.CLOSE:
                # Punish agent for eating when there is no food in site and it's health is bad
                return -10.0
            else:
                return stationary_penalty

    def perception_observation(self):
        wolf_proximity, food_proximity, _, _ = self._perception_observation()
        return wolf_proximity, food_proximity

    def _perception_observation(self):
        d = 0.0
        wolf_proximity = DistanceObservation.UNKNOWN
        for predator in self.current.predators:
            d += dist(predator, self.current.position)
            wolf_proximity = DistanceObservation.proximity_level(d)
        food_proximity = DistanceObservation.UNKNOWN
        for food in self.current.foodSources:
            d += dist(food, self.current.position)
            food_proximity = DistanceObservation.proximity_level(d)
        health = HealthObservation.health_level(self.current.health)
        movement = dist(self.current.position, self.previous.position)
        movement = MovementObservation.movement_level(movement)
        return wolf_proximity, food_proximity, health, movement

    @staticmethod
    def create(current, previous):
        return Perceptor(current, previous)

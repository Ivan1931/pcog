from json import loads
import math
from random import choice
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
        return [(w, f, m) for w in DistanceObservation.SET
                          for f in DistanceObservation.SET
                          for m in MovementObservation.SET]

    def perception_reward(self, action):
        stationary_penalty = -0.5
        wolf, food, movement = self._perception_observation()
        if action == Action.EAT:
            if movement == MovementObservation.STATIONARY:
                return stationary_penalty
            if DistanceObservation.CLOSE <= food:
                return 10.0
            return 0.0
        if action == Action.ATTACK:
            if movement == MovementObservation.STATIONARY:
                return stationary_penalty
            if DistanceObservation.CLOSE <= wolf:
                return 10.0
            return 0.0
        if action == Action.EXPLORE:
            if movement == MovementObservation.STATIONARY:
                return 2.0
            return 0.2

    def perception_observation(self):
        return self._perception_observation()

    def _perception_observation(self):
        if self.current.lastWolfPosition is None:
            wolf_proximity = DistanceObservation.UNKNOWN
        else:
            wolf_proximity = DistanceObservation.proximity_level(dist(self.current.position, self.current.lastWolfPosition))
        if self.current.lastFoodPosition is None:
            food_proximity = DistanceObservation.UNKNOWN
        else:
            food_proximity = DistanceObservation.proximity_level(dist(self.current.position, self.current.lastFoodPosition))
        if self.current.position == self.previous.position:
            movement = MovementObservation.STATIONARY
        else:
            movement = MovementObservation.movement_level(dist(self.current.position, self.previous.position))
        return wolf_proximity, food_proximity, movement

    def smart_explore(self):
        wolf, food, movement = self._perception_observation()
        if DistanceObservation.CLOSE <= food and movement != MovementObservation.STATIONARY:
            return Action.EAT
        elif DistanceObservation.CLOSE <= wolf:
            return Action.ATTACK
        else:
            return Action.EXPLORE

    @staticmethod
    def create(current, previous):
        return Perceptor(current, previous)

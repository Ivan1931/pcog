from json import loads
import math
from random import choice
import sys
from bunch import bunchify
from .envconf import Change, DistanceObservation, HealthObservation, Action, MovementObservation
from .humanoid import dist


def process(state_perception_string):
    return bunchify(loads(state_perception_string))


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
        stationary_penalty = -10.0
        wolf, food, movement = self._perception_observation()
        if action == Action.EAT:
            if movement == MovementObservation.STATIONARY:
                return stationary_penalty
            if DistanceObservation.CLOSE <= food:
                return 10.0
            return 0.0
        if action == Action.ATTACK:
            if DistanceObservation.FAR <= wolf:
                return 10.0
            return stationary_penalty
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
            wolf_proximity = DistanceObservation.proximity_level(dist(self.current.position,
                                                                      self.current.lastWolfPosition))
        if self.current.lastFoodPosition is None:
            food_proximity = DistanceObservation.UNKNOWN
        else:
            food_proximity = DistanceObservation.proximity_level(dist(self.current.position,
                                                                      self.current.lastFoodPosition))
        if self.current.position == self.previous.position:
            movement = MovementObservation.STATIONARY
        else:
            movement = MovementObservation.movement_level(dist(self.current.position, self.previous.position))
        return wolf_proximity, food_proximity, movement

    def smart_explore(self):
        max_reward = -sys.maxint
        max_action = Action.EXPLORE
        for action in Action.SET:
            reward = self.perception_reward(action)
            if max_reward < reward:
                max_action = action
                max_reward = reward
        return max_action

class SimplePerceptor(Perceptor):
    @staticmethod
    def possible_observations():
        return [(wolf, food, health, movement, died)
                for wolf in [True, False]
                for food in [True, False]
                for died in [True, False]
                for health in HealthObservation.SET
                for movement in MovementObservation.SET]

    def perception_observation(self):
        wolf = 0 < self.current.numberOfSeenPredators
        food = 0 < self.current.numberOfSeenEdibles
        food = food != DistanceObservation.UNKNOWN
        current = self.current.position
        previous = self.previous.position
        movement = MovementObservation.movement_level(dist(current, previous))
        health = HealthObservation.health_level(self.current.health)
        died = self.current.deaths != self.previous.deaths
        return wolf, food, health, movement, died

    def perception_reward(self, action):
        wolf, food, health, movement, died = self.perception_observation()
        if died:
            return -50.0
        if action == Action.ATTACK:
            if wolf and HealthObservation.BAD != health:
                return 10.0
            elif movement == MovementObservation.STATIONARY:
                return -10.0
            else:
                return -5.0
        elif action == Action.EXPLORE:
            if not wolf and not food:
                if movement == MovementObservation.STATIONARY:
                    return 7.0
                else:
                    return 2.0
            else:
                if health == HealthObservation.GOOD:
                    return 3.0
                else:
                    return -4.0
        elif action == Action.EAT:
            if food and health != HealthObservation.GOOD:
                return 15.0
            elif health == HealthObservation.GOOD:
                return -20.0
            elif movement == MovementObservation.STATIONARY:
                return -10.0
            else:
                return -3.0
        elif action == Action.FLEE:
            if wolf and health == HealthObservation.BAD:
                return 5.0
            else:
                return -20.0


class ComplexPerceptor(SimplePerceptor):
    def score_action(self, action):
        wolf = self.current.lastWolfPosition is not None
        food = self.current.lastFoodPosition is not None
        health = HealthObservation.health_level(self.current.health)
        d = dist(self.current.position, self.previous.position)
        movement = MovementObservation.movement_level(d)
        if action == Action.ATTACK:
            if wolf and health != HealthObservation.BAD:
                return 4.0
        elif action == Action.EAT:
            if food and health != HealthObservation.GOOD:
                return 5.0
            elif wolf:
                return -3.0
            else:
                return -2.0
        elif action == Action.FLEE:
            if wolf and HealthObservation == HealthObservation.BAD:
                return 4.0
            else:
                return -2.0
        else:
            if not wolf and not food and movement == MovementObservation.STATIONARY:
                return 2.0
            else:
                return 0.0

    def smart_explore(self):
        max_action = Action.EXPLORE
        max_reward = -sys.maxint
        for action in Action.SET:
            reward = self.score_action(action)
            if max_reward < reward:
                max_reward = reward
                max_action = action
        return max_action

    def perception_reward(self, action):
        kill_increase = self.current.kills != self.previous.kills
        wolf = self.previous.lastWolfPosition is not None
        food = self.previous.lastFoodPosition is not None
        current_health = HealthObservation.health_level(self.current.health)
        past_health = HealthObservation.health_level(self.previous.health)
        death = self.previous.deaths != self.current.deaths
        d = dist(self.current.position, self.previous.position)
        movement = MovementObservation.movement_level(d)
        if death:
            return -20.0
        if action == Action.ATTACK:
            if wolf:
                if kill_increase:
                    return 15.0
                else:
                    return 3.0
            elif movement == MovementObservation.STATIONARY:
                return -7.0
            else:
                return -5.0
        elif action == Action.EAT:
            if food:
                if past_health < current_health:
                    return 10.0
                else:
                    return -5.0
            else:
                return -5.0
        elif action == Action.FLEE:
            if past_health == HealthObservation.BAD and wolf:
                return 5.0
            else:
                return -7.0
        else:
            if wolf and past_health == HealthObservation.GOOD:
                return -5.0
            elif food and past_health != HealthObservation.GOOD:
                return -5.0
            else:
                return 1.0

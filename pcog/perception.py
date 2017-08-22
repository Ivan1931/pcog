from json import loads
import math
from bunch import bunchify
from .envconf import Change
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


def preceive_instance(current, previous):
    distance, health, predators = perceive(current, previous)
    reward = perception_reward(current, previous)


def sigmoid(x):
    return math.exp(x) / (math.exp(x) + 1.0)

def perception_reward(current, previous, conf=dict(max_health=10.0)):
    health_change, distance_change, visible_predators_change = _change(current, previous)
    return health_change / conf["max_health"] + sigmoid(distance_change) + sigmoid(visible_predators_change)
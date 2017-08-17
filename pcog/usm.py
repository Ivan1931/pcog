from collections import deque, defaultdict

class StorageType(object):
    OBSERVATION = 0
    ACTION = 1


class Instance(object):
    def __init__(self, action, observation, reward):
        self.action = action
        self.observation = observation
        self.reward = reward

    def usm_nodes(self):
        return ActionNode(self.action), ObservationNode(self.observation)


class USMNode(object):
    def __init__(self):
        self.is_fringe = True
        self.children = {}

    def __str__(self):
        return "r"


class ActionNode(USMNode):
    def __init__(self, action):
        USMNode.__init__(self)
        self.action = action

    def __str__(self):
        return "{}".format(self.action)

class ObservationNode(USMNode):
    def __init__(self, observation):
        USMNode.__init__(self)
        self.observation = observation

    def __str__(self):
        return "{}".format(self.observation)

class UtileSuffixMemory(object):
    def __init__(self):
        self.root = USMNode()
        self.previous = []

    def insert(self, instance):
        self.previous.append(instance)
        self._insert()

    def _insert(self):
        current = self.root
        for i in reversed(self.previous):
            action, observation = i.usm_nodes()
            if action.action in current.children:
                current = current.children[action.action]
            else:
                current.children[action.action] = action
                current = action
            if observation.observation in current.children:
                current = current.children[observation.observation]
            else:
                current.children[observation.observation] = observation
                current = observation


    def qval(self, action):
        raise NotImplementedError()


    def reward(self):
        raise NotImplementedError()


    def is_leaf(self):
        return len(self.children) == 0


    def display(self):
        import pdb; pdb.set_trace()
        levels = defaultdict(list)
        queue = deque([])
        queue.append((self.root, 0))
        max_level = 0
        while len(queue) != 0:
            node, level = queue.popleft()
            levels[level].append(node)
            max_level = max(level, max_level)
            children = deque([])
            for child in node.children.values():
                queue.append((child, level+1))
        result = ""
        for i in range(max_level+1):
            level = levels[i]
            result += " ".join(map(str, level)) + "\n"
        return result

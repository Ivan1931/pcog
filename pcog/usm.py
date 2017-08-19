from collections import deque, defaultdict

class StorageType(object):
    OBSERVATION = 0
    ACTION = 1


class Instance(object):
    def __init__(self, action, observation, reward):
        self.action = action
        self.observation = observation
        self.reward = reward
        self.leaf = None
        self.next = None
        self.previous = None

    def usm_nodes(self):
        act, obs = ActionNode(self.action), ObservationNode(self.observation)
        act.add_instance(self)
        obs.add_instance(self)
        return act, obs


    def add_front(self, instance):
        self.next = instance
        instance.previous = self


class USMNode(object):
    def __init__(self):
        self.is_fringe = True
        self.children = {}
        self.instances = []
        self.parent = None


    def __str__(self):
        return "r"


    def child(self, key):
        return self.children.get(key)


    def add_child(self, key, child_node):
        child_node.parent = self
        self.children[key] = child_node


    def add_instance(self, instance):
        self.instances.append(instance)


    def is_leaf(self):
        return len(self.children) == 0


    def _reward(self, total_reward, instances):
        return total_reward, instances


    def reward(self):
        total, instances = self._reward(0.0, 0)
        return total / float(instances)


class ActionNode(USMNode):
    def __init__(self, action):
        USMNode.__init__(self)
        self.action = action

    def __str__(self):
        return "{}".format(self.action)

    def _instances_reward(self):
        return sum([i.reward for i in self.instances])

    def _reward(self, total_reward, instances):
        return self.parent._reward(total_reward + self._instances_reward(), 
                                   instances + len(self.instances))


    def observation(self, o):
        return self.child(o)


class ObservationNode(USMNode):
    def __init__(self, observation):
        USMNode.__init__(self)
        self.observation = observation

    def __str__(self):
        return "{}".format(self.observation)

    def action(self, a):
        return self.child(a)

    def _reward(self, total_reward, instances):
        return self.parent._reward(total_reward, instances)


class UtileSuffixMemory(object):
    def __init__(self, window_size=5):
        self.root = USMNode()
        self.instances = []
        self.states = set()
        self.window_size = window_size

    def insert(self, instance):
        if len(self.instances) != 0: 
            self.instances[-1].add_front(instance)
        self.instances.append(instance)
        state = self._insert()
        if state.is_leaf():
            self.states.add(state)
        elif current in self.states:
            self.states.remove(state)
        instance.leaf = state
        return state

    def _insert(self):
        current = self.root
        for i in reversed(self.instances[-self.window_size:]):
            action, observation = i.usm_nodes()
            if action.action in current.children:
                current = current.children[action.action]
            else:
                current.children[action.action] = action
                current.add_child(action.action, action)
                current = action
            if observation.observation in current.children:
                current = current.children[observation.observation]
            else:
                current.add_child(observation.observation, observation)
                current = observation
        return current


    def pr(self, s1, s2, action):
        count = 0
        total = 0
        current = s1
        while current is not self.root:
            if type(current) is ActionNode:
                if current.action == action:
                    for i in current.instances:
                        total += 1
                        successor = i.next
                        if successor and successor.leaf is s2:
                            count += 1
            current = current.parent
        if total == 0.0: return 0.0
        return float(count) / float(total)

    def display(self):
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

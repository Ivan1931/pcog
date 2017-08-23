from collections import deque, defaultdict


class Instance(object):
    def __init__(self, action, observation, reward):
        self.action = action
        self.observation = observation
        self.reward = reward
        self._tree_node = None
        self.next = None
        self.previous = None

    def usm_nodes(self):
        act, obs = ActionNode(self.action), ObservationNode(self.observation)
        act.add_instance(self)
        obs.add_instance(self)
        return act, obs
    

    def get_node(self):
        return self._tree_node


    def set_node(self, usm_node):
        self._tree_node = usm_node


    def add_front(self, instance):
        self.next = instance
        instance.previous = self

    def eql_i(self, that):
        return (isinstance(that, Instance)
            and self.observation == that.observation
            and self.action      == that.action)


class USMNode(object):
    def __init__(self):
        self.is_fringe = False
        self.children = {}
        self.instances = []
        self.parent = None

    def __str__(self):
        return "r"


    def set_fringe(self, fringe):
        self.is_fringe = fringe
        
    def child(self, key):
        return self.children.get(key)


    def add_child(self, key, child_node):
        child_node.parent = self
        self.children[key] = child_node


    def add_instance(self, instance):
        self.instances.append(instance)


    def is_leaf(self):
        if self.is_fringe: 
            return False
        return (len(self.children) == 0 
                or 
                all([child.is_fringe for child in self.children.values()]))


    def _reward(self, total_reward, instances, action):
        return total_reward, instances


    def reward(self, action):
        total, instances = self._reward(0.0, 0, action)
        return total / float(instances)


class ActionNode(USMNode):
    def __init__(self, action):
        USMNode.__init__(self)
        self.action = action

    def __str__(self):
        return "{}".format(self.action)

    def _instances_reward(self, action):
        if self.action == action:
            return sum([i.reward for i in self.instances])
        else:
            return 0.0

    def _reward(self, total_reward, instances, action):
        return self.parent._reward(total_reward + self._instances_reward(action), 
                                   instances + len(self.instances),
                                   action)


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

    def _reward(self, total_reward, instances, action):
        return self.parent._reward(total_reward, instances, action)


class UtileSuffixMemory(object):
    def __init__(self,
                 window_size=5,
                 fringe_depth=2,
                 gamma=0.3,
                 known_actions=None,
                 known_observations=None):
        self._root = USMNode()
        self.instances = []
        self._states = set()
        self.window_size = window_size
        self.fringe_depth = fringe_depth
        self.gamma = gamma
        self._action_space = known_actions
        self._observation_space = known_observations

    def _correct_fringe(self, node):
        """
        Walks up the parents of a none-fringe node
        and changes all of it's ancestors non-fringe elements.
        This is necessary if we promote an element to non-fringe
        or a leaf is inserted bellow the fringe.
        """
        assert(not node.is_fringe)
        while node is not self._root:
            node.set_fringe(True)
            node = node.parent

    def insert(self, instance):
        if instance in self.instances:
            raise ValueError("Inserting an instance that has already been used")
        if 0 < len(self.instances): 
            self.instances[-1].add_front(instance)
        self.instances.append(instance)
        state = self._insert()
        if state.is_leaf():
            self._states.add(state)
            self._correct_fringe(state)
        instance.set_node(state)
        return state

    def _insert_instances(self, start_node, instances, fringe=False):
        current = start_node
        for i in instances:
            action, observation = i.usm_nodes()
            action.set_fringe(fringe)
            observation.set_fringe(fringe)
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

    def _insert_leaf(self, suffix):
        return self._insert_instances(self._root, reversed(suffix), False)


    def _insert_fringe(self, suffix):
        presuffix = suffix[0:self.fringe_depth]
        post_suffix = suffix[self.fringe_depth:]
        for state in self._states:
            # Match prefix of the suffix with the suffix of each state
            # If we can do the match then add the instance suffix to the fringe
            # With the suffix node as root
            count = 0
            instance = state.instances[0]
            equal = True
            while equal and instance and count < self.fringe_depth:
                equal = instance.eql_i(presuffix[count])
                instance = instance.previous
                count += 1
            if equal:
                self._insert_instances(state, post_suffix, True)
    

    def _insert(self):
        suffix = self.instances[-self.window_size:]
        leaf = self._insert_leaf(suffix)
        self._insert_fringe(suffix)
        return leaf


    def has_actions(self):
        return 0 < len(self._action_space)


    def has_observations(self):
        return 0 < len(self._observation_space)

    def get_actions(self):
        return self._action_space


    def get_observations(self):
        return self._observation_space


    def get_states(self):
        return self._states


    def get_root(self):
        return self._root


    def get_instances(self):
        return self.instances


    def _leaves(self, internal_node):
        return [i.get_node() for i in internal_node.instances if i.get_node().is_leaf()]


    def traverse(self, instances):
        current = self.get_root()
        for i in instances:
            if current.is_fringe:
                return [current.parent]
            if current.is_leaf():
                return [current]
            if i.action in current.children:
                current = current.children[i.action]
            else:
                return self._leaves(current)
            if i.observation in current.children:
                current = current.children[i.observation]
            else:
                return self._leaves(current)
                

    
    def _tau(self, state, action):
        instances = []         
        current = state
        while current is not self._root:
            if isinstance(current, ActionNode) and current.action == action:
                instances += current.instances
            current = current.parent
        return instances


    def pr(self, s1, s2, action):
        """
        Probability of transitioning from `s1` to `s2` given `action`
        The following invariant must be maintained:
        ```
        sum([Pr(s1, a, s2) for s2 in states]) == 1.0
        ```
        """
        tau = self._tau(s1, action)
        if len(tau) == 0:
            return 1.0 / float(len(self.get_states()))
        total = 0.0
        for i in tau:
            if i.get_node() is s2:
                total += 1.0
        return total / float(len(tau))



    def observation_fn(self, state, action, observation):
        if len(self.instances) == 0:
            raise ValueError("Attempted to find the observation function on USM with no instances")
        count = 0.0
        observed_count = 0.0
        for i in self.instances:
            if action == i.action and i.get_node() is state:
                observed_count += 1.0
                if i.observation == observation:
                    count += 1.0
        if observed_count == 0.0:
            return 1.0 / float(len(self.get_observations()))
        return count / observed_count


    def display(self):
        levels = defaultdict(list)
        queue = deque([])
        queue.append((self._root, 0))
        max_level = 0
        while 0 < len(queue):
            node, level = queue.popleft()
            levels[level].append(node)
            max_level = max(level, max_level)
            for child in node.children.values():
                queue.append((child, level+1))
        result = ""
        for i in range(max_level+1):
            level = levels[i]
            result += " ".join(map(str, level)) + "\n"
        return result
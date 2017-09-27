from collections import deque, defaultdict
from random import randint
from scipy.stats import ks_2samp
import sys

EPSILON = 0.000000001


def normalise_to_one(numbers):
    s = sum(numbers)
    i = randint(0, len(numbers)-1)
    if s == 1.0:
        return numbers
    elif s < 1.0:
        d = 1.0 - s
        placed = False
        while not placed:
            if 0.0 <= numbers[i] + d <= 1.0:
                numbers[i] += d
                placed = True
            else:
                i = randint(0, len(numbers) - 1)
        return numbers
    else:
        placed = False
        d = s - 1.0
        while not placed:
            if 0.0 <= numbers[i] - d <= 1.0:
                numbers[i] -= d
                placed = True
            else:
                i = randint(0, len(numbers) - 1)
        return numbers


class USMStats(object):
    def __init__(self):
        self.belief_distribution = {}

    def belief_distribution(self):
        pass

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

    def get_children(self):
        return self.children.values()


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
                 window_size=4,
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
            node.set_fringe(False)
            node = node.parent
            if node is not self._root:
                # If the node was a leaf node before the fringe insertion
                # we should remove it since it's no longer a valid state
                if node in self._states:
                    self._states.remove(node)

    def insert(self, instance):
        if instance in self.instances:
            raise ValueError("Inserting an instance that has already been used")
        if 0 < len(self.instances): # Add new instance to end of instance linked list
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
            if i.action in current.children:
                current = current.children[action.action]
                current.add_instance(i)
            else:
                current.children[action.action] = action
                current.add_child(action.action, action)
                current = action
            if i.observation in current.children:
                current = current.children[observation.observation]
                current.add_instance(i)
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

    def utility(self, state):
        # type: (UtileSuffixMemory, USMNode) -> float
        best = -sys.maxint
        for action in self.get_actions():
            best = max(state.reward(action), best)
        return best

    def tree_leaves(self):
        # BFS of the tree for all leaves
        children = []
        leaves = []
        children.append(self._root)
        while len(children) != 0:
            n = children.pop()
            if len(n.get_children()) == 0:
                leaves.append(n)
            else:
                kids = n.get_children()
                children += kids
        return leaves

    def unfringe(self, alpha=0.05):
        """
        This method decides whether to expand the suffix trie onto the fringe.
        It does this by constructing a distribution of the utilities of all the current states and from one
        that includes the current state as well fringe nodes that are on the unofficial leaf of the tree.
        If the two distributions are sufficiently different then the state space is expanded to include all leaf nodes
        currently in the tree. The distribution comparison is performed using a KS test.
        """
        all_leaves = self.tree_leaves()
        all_leaves_dist = list(map(lambda leaf: self.utility(leaf), all_leaves))
        print(all_leaves_dist)
        current_leaves = self.get_states()
        current_dist = list(map(lambda leaf: self.utility(leaf), current_leaves))
        print(current_dist)
        D, p_value = test_result = ks_2samp(all_leaves_dist, current_dist)
        print(D)
        print(p_value)
        if p_value < alpha or alpha < D:
            for leaf in all_leaves:
                if leaf.is_fringe:
                    leaf.set_fringe(False)
                    self._correct_fringe(leaf)
            return True
        else:
            return False

    def _leaves(self, internal_node):
        return [i.get_node() for i in internal_node.instances if i.get_node().is_leaf()]

    def traverse(self, instances):
        if len(instances) < 1:
            raise ValueError("You cannot traverse the tree with empty instances")
        current = self.get_root()
        for i in instances:
            if current.is_leaf():
                return [current]
            if i.action in current.children:
                current = current.children[i.action]
            else:
                return self._leaves(current.parent)
            if i.observation in current.children:
                current = current.children[i.observation]
            else:
                return self._leaves(current)
        if current.is_leaf():
            return [current]
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

    def transition_for(self, incident_state, action):
        """
        Calculates a probability distribution over the state space which
        represents the likelyhood of moving from indicent state to any other
        state from a given action.
        :param incident_state: State that we are travelling from
        :param action: Action that we are considering
        :return: Array which represents a probability distribution - always sums to 1
        """
        l = len(self.get_states())
        transitions = [None for _ in range(l)]
        tau = self._tau(incident_state, action)
        idx = 0
        leaf_count = 0.0
        for i in tau:
            if i.next and i.next.get_node().is_leaf():
                leaf_count += 1.0
        if leaf_count <= 0.0:
            for idx, state in enumerate(self.get_states()):
                if state is incident_state:
                    transitions[idx] = 1.0
                else:
                    transitions[idx] = 0.0
            return transitions
        for arrival_state in self.get_states():
            equal_count = 0.0
            for i in tau:
                """
                Within _tau there are two possibilities:
                arrival_state is equal to the leaf of the successor instance to the instance
                It is not
                Since some instances may be associated with internal nodes it actually makes sense to only
                consider instances that map to a leaf node which is a another state
                """
                if i.next and i.next.get_node().is_leaf():
                    if i.next.get_node() is arrival_state:
                        equal_count += 1.0
            transitions[idx] = equal_count / leaf_count
            idx += 1
        x = sum(transitions)
        if abs(x - 1.0) >= EPSILON:
            raise ValueError("Transition function is not close enough to one")
        return normalise_to_one(transitions)

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
        count = 0.0
        for i in tau:
            if i.next and i.next.get_node().is_leaf():
                count += 1.0
                if i.next.get_node() is s2:
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

    def observation_for(self, state, action):
        observations = [None for _ in self.get_observations()]
        total_possible = 0.0
        for i in self.instances:
            if i.get_node() is state and i.action == action:
                total_possible += 1.0
        # Handle edge case where there is no state or action recorded for this observation
        # Also handles cases where the sum of the distribution does not add up to zero
        # Hacky I know :(
        if total_possible == 0.0:
            p = 1.0 / len(self.get_observations())
            return normalise_to_one([p for _ in self.get_observations()])
        for idx, observation in enumerate(self.get_observations()):
            observation_count = 0.0
            for i in self.instances:
                if i.observation == observation and i.get_node() is state and i.action == action:
                    observation_count += 1.0
            observations[idx] = observation_count / total_possible
        x = sum(observations)
        if EPSILON <= abs(x - 1.0):
            raise ValueError("Observation function distribution is not close enough to one")
        return observations

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

    def derive_new(self, memodict={}):
        usm = UtileSuffixMemory(
            window_size=self.window_size,
            gamma=self.gamma,
            known_actions=self.get_actions(),
            known_observations=self.get_observations(),
        )
        for i in self.get_instances():
            usm.insert(Instance(
                action=i.action,
                observation=i.observation,
                reward=i.reward
            ))
        return usm

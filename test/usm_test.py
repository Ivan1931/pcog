import unittest
import random
import pickle
from pcog.usm import *
from pcog.usm_draw import *
from pcog.usm_pomdp import build_pomdp_model, solve, belief_state


class USMTest(unittest.TestCase):
    def _generate_test_usm(self):
        aba = [
            Instance("a1", "o1", 1.0),
            Instance("a2", "o2", 1.0),
            Instance("a2", "o1", 1.0),
            Instance("a1", "o3", 1.0),
        ]
        tree = UtileSuffixMemory(known_actions=["a1", "a2"], 
                                 known_observations=["o1", "o2", "o3"])
        for i in aba:
            tree.insert(i)
        return tree

    def _generate_random_usm(self, n):
        obs = ["o1", "o2", "o3", "o4"]
        acts = ["a1", "a2", "a3"]
        tree = UtileSuffixMemory(10)
        for i in range(n):
            o = random.choice(obs)
            a = random.choice(acts)
            tree.insert(Instance(a, o, 1.0))
        return tree

    def test_insertion(self):
        tree = self._generate_test_usm()
        print(tree.display())
        a1 = tree._root.child('a1')
        self.assertEqual(a1.reward('a1'), 1.0)


    def test_pr(self):
        tree = self._generate_test_usm()
        for s1 in tree.get_states():
            for s2 in tree.get_states():
                if s1 is not s2:
                    print(tree.pr(s1, s2, "a1"))
                    print(tree.pr(s1, s2, "a2"))
        tree = self._generate_random_usm(25)
        draw_usm(tree)
        print("*" * 10)
        for s1 in tree.get_states():
            t1 = 0.0
            t2 = 0.0
            for s2 in tree.get_states():
                if s1 is not s2:
                    p1 = tree.pr(s1, s2, "a1")
                    t1 += p1
                    print("{} -> {} given a1: {}".format(s1, s2, p1))
                    p2 = tree.pr(s1, s2, "a2")
                    t2 += p2
                    print("{} -> {} given a2: {}".format(s1, s2, p2))
            print(('#' * 5) + ' ' + str(t1))
            print(('#' * 5) + ' ' + str(t2))


    def test_drawing(self):
        tree = self._generate_test_usm()
        draw_usm(tree)


    def _make_instance_pattern(self):
        return [ Instance("a1","o1", 1.0), Instance("a2","o2", 1.0), Instance("a1","o3", 1.0) ]


    def test_derive_pomdp_with_duplicate_instance_sequence(self):
        pattern1 = self._make_instance_pattern()
        pattern2 = self._make_instance_pattern()
        actions = ["a1", "a2"]
        observations = ["o1", "o2","o3"]
        usm = UtileSuffixMemory(
            known_actions=actions,
            known_observations=observations,
            window_size=3
        )
        for p in pattern1:
            usm.insert(p)
        for p in pattern2:
            usm.insert(p)
        # draw_usm(usm)
        pomdp = build_pomdp_model(usm)

    def test_live_usm(self):
        with open('test/data/test_usm', 'r') as f:
            usm = pickle.load(f)
            # draw_usm(usm)
            pomdp = build_pomdp_model(usm)
            self.assertEqual(True, True)

    def test_derive_pomdp(self):
        tree = self._generate_test_usm()
        tree._action_space = ["a1", "a2"]
        tree._observation_space = ["o1", "o2", "o3"]
        pomdp = build_pomdp_model(tree)
        action = solve(tree, pomdp, [
            Instance("a1", "o2", 1.0),
            Instance("a2", "o1", 1.0)
        ])
        self.assertIn(action, range(len(tree.get_actions())))

    def test_derive_belief(self):
        usm = self._generate_test_usm()
        bs = belief_state(usm, [
            Instance("a1", "o3", 1.0),
        ])
        self.assertAlmostEqual(sum(bs), 1.0)

if __name__ == "__main__": 
    unittest.main()

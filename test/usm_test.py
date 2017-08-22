import unittest
import random
from pcog.usm import *
from pcog.usm_draw import *

class USMTest(unittest.TestCase):
    def _generate_test_usm(self):
        aba = [
            Instance("a1", "o1", 1.0),
            Instance("a2", "o2", 1.0),
            Instance("a2", "o1", 1.0),
            Instance("a1", "o3", 1.0),
        ]
        tree = UtileSuffixMemory()
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
        self.assertEqual(a1.reward(), 1.0)


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


if __name__ == "__main__": unittest.main()

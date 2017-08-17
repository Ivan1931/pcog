import unittest
import random
from pcog.usm import *

class USMTest(unittest.TestCase):
    def test_insertion(self):
        """
        
          a1             a2           a3
        o1    o2      o1   o2      o1   o2
     a1 a3 a2   a1  a2  a1   a3  a1  a3
        """
        

    def test_removal(self):
        actions = ["a1", "a2", "a3"]
        obs = ["o1", "o2", "o3"] 
        aba = [
            Instance("a1", "o1", 1.0),
            Instance("a2", "o2", 1.0),
            Instance("a2", "o1", 1.0),
            Instance("a1", "o3", 1.0),
        ]
        tree = UtileSuffixMemory()
        for i in aba:
            tree.insert(i)
        print(tree.display())


    def test_split(self):
        pass


if __name__ == "__main__": unittest.main()

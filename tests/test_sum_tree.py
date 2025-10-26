# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/sum_tree_test.py

import unittest

from slimdqn.sample_collection import sum_tree


class SumTreeTest(unittest.TestCase):

    def setUp(self):
        super(SumTreeTest, self).setUp()
        self.tree = sum_tree.SumTree(capacity=100)

    def test_set_small_capacity(self):
        tree = sum_tree.SumTree(capacity=1)
        tree.set(0, 1.5)
        self.assertEqual(tree.root, 1.5)

    def test_set_and_get_value(self):
        self.tree.set(0, 1.0)
        self.assertEqual(self.tree.get(0), 1.0)

        # Validate that all nodes on the leftmost branch have value 1.
        leaf_index = self.tree.first_leaf_offset
        while leaf_index > 0:
            leaf_index = leaf_index // 2
            self.assertEqual(self.tree.nodes[leaf_index], 1.0)

    def test_set_and_get_values_vectorized(self):
        self.tree.set([1, 2], [3.0, 4.0])
        self.assertEqual(self.tree.get(1), 3.0)
        self.assertEqual(self.tree.get(2), 4.0)
        self.assertEqual(self.tree.root, 7.0)

    def test_set_with_duplicates(self):
        self.tree.set([1, 1, 1, 2, 2], [3.0, 3.0, 3.0, 4.0, 4.0])
        self.assertEqual(self.tree.get(1), 3.0)
        self.assertEqual(self.tree.get(2), 4.0)
        self.assertEqual(self.tree.root, 7.0)

    def test_query_value(self):
        self.tree.set(5, 1.0)
        self.assertEqual(self.tree.query(0.99), 5)

    def test_max_recorded_priority(self):
        self.tree.set(0, 0)
        self.assertEqual(self.tree.max_recorded_priority, 1)
        for i in range(1, 32):
            self.tree.set(i, i)
            self.assertEqual(self.tree.max_recorded_priority, i)

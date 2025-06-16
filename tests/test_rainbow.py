import unittest
import numpy as np
import jax
import jax.numpy as jnp
import optax

from slimdqn.networks.rainbow import Rainbow
from tests.utils import Generator


class TestRainbow(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_bins, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(
            self.key, 6
        )
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.n_bins = int(jax.random.randint(key_bins, (), minval=1, maxval=10))
        self.q = Rainbow(
            self.key,
            self.observation_dim,
            self.n_actions,
            self.n_bins,
            [
                jax.random.randint(key_feature_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_2, (), minval=1, maxval=10),
                jax.random.randint(key_feature_3, (), minval=1, maxval=10),
                jax.random.randint(key_feature_4, (), minval=1, maxval=10),
            ],
            "impala",
            0.001,
            0.94,
            1,
            1,
            1,
            -10,
            10,
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_target_support, computed_target_prob = self.q.compute_target(self.q.params, sample)

        next_q_logits = self.q.network.apply_fn(self.q.params, sample.next_state)
        target_support = sample.reward + (1 - sample.is_terminal) * self.q.gamma * self.q.support
        target_prob = jax.nn.softmax(next_q_logits[jnp.argmax(jax.nn.softmax(next_q_logits) @ self.q.support)])

        self.assertEqual(next_q_logits.shape, (self.n_actions, self.n_bins))
        self.assertEqual(target_support.tolist(), computed_target_support.tolist())
        self.assertEqual(target_prob.tolist(), computed_target_prob.tolist())

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_loss = self.q.loss(self.q.params, self.q.params, sample)[0]

        target_support, target_prob = self.q.compute_target(self.q.params, sample)
        prediction = self.q.network.apply_fn(self.q.params, sample.state)[sample.action]
        loss = optax.softmax_cross_entropy(prediction, self.q.project_target_on_support(target_support, target_prob)[0])

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_values = self.q.network.apply_fn(self.q.params, state)
        best_action = jnp.argmax(jax.nn.softmax(q_values) @ self.q.support)

        self.assertEqual(q_values.shape, (self.n_actions, self.n_bins))
        self.assertEqual(best_action, computed_best_action)

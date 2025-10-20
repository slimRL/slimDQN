from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp

from slimdqn.sample_collection.replay_buffer import ReplayElement


class Generator:
    def __init__(self, batch_size: int, observation_dim: Tuple[int], n_actions: int) -> None:
        self.batch_size = batch_size
        self.observation_dim = observation_dim
        self.n_actions = n_actions

    @partial(jax.jit, static_argnames="self")
    def sample(
        self,
        key: jax.random.PRNGKey,
    ) -> ReplayElement:
        state = jax.random.uniform(key, self.observation_dim)
        action = jax.random.randint(key, (), minval=0, maxval=self.n_actions, dtype=jnp.int8)
        _, key_ = jax.random.split(key)
        reward = jax.random.uniform(key_)
        terminal = jax.random.randint(key_, (), 0, 2)
        next_state = jax.random.uniform(key_, self.observation_dim)
        return ReplayElement(
            state,  # state
            action,  # action
            reward,  # reward
            next_state,  # next_state
            terminal,  # terminal
        )

    @partial(jax.jit, static_argnames="self")
    def samples(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray]:
        return jax.vmap(self.sample)(jax.random.split(key, self.batch_size))

    @partial(jax.jit, static_argnames="self")
    def state(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.uniform(key, self.observation_dim)

    @partial(jax.jit, static_argnames="self")
    def states(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.uniform(key, (self.batch_size,) + self.observation_dim)

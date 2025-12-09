import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, TransitionElement


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_fn, n_training_steps):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_training_steps),  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


def collect_single_sample(key, env, agent, rb: ReplayBuffer, p, epsilon_schedule, n_training_steps: int):
    action = select_action(
        agent.best_action, agent.params, env.state, key, env.n_actions, epsilon_schedule, n_training_steps
    ).item()

    obs = env.observation
    reward, absorbing = env.step(action)

    episode_end = absorbing or env.n_steps >= p["horizon"]
    rb.add(
        TransitionElement(
            observation=obs,
            action=action,
            reward=reward if rb.clipping is None else rb.clipping(reward),
            is_terminal=absorbing,
            episode_end=episode_end,
        )
    )

    if episode_end:
        env.reset()

    return reward, episode_end


def define_boxes(env, n_states_x: int, n_states_v: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    states_x = np.linspace(-env.max_pos, env.max_pos, n_states_x)
    boxes_x_size = (2 * env.max_pos) / (n_states_x - 1)
    states_x_boxes = np.linspace(-env.max_pos, env.max_pos + boxes_x_size, n_states_x + 1) - boxes_x_size / 2
    states_v = np.linspace(-env.max_velocity, env.max_velocity, n_states_v)
    boxes_v_size = (2 * env.max_velocity) / (n_states_v - 1)
    states_v_boxes = np.linspace(-env.max_velocity, env.max_velocity + boxes_v_size, n_states_v + 1) - boxes_v_size / 2

    return states_x, states_x_boxes, states_v, states_v_boxes


class TwoDimesionsMesh:
    def __init__(
        self, dimension_one, dimension_two, sleeping_time: float, axis_equal: bool = True, zero_centered: bool = False
    ) -> None:
        self.dimension_one = dimension_one
        self.dimension_two = dimension_two
        self.grid_dimension_one, self.grid_dimension_two = np.meshgrid(self.dimension_one, self.dimension_two)

        self.sleeping_time = sleeping_time
        self.axis_equal = axis_equal
        self.zero_centered = zero_centered

        self.values = np.zeros((len(self.dimension_one), len(self.dimension_two)))

    def set_values(self, values: np.ndarray, zeros_to_nan: bool = False) -> None:
        assert values.shape == (
            len(self.dimension_one),
            len(self.dimension_two),
        ), f"given shape values: {values.shape} don't match with environment values: {(len(self.dimension_one), len(self.dimension_two))}"

        self.values = values
        if zeros_to_nan:
            self.values = np.where(self.values == 0, np.nan, self.values)

    def show(
        self,
        title: str = "",
        xlabel: str = "States",
        ylabel: str = "Actions",
        plot: bool = True,
        ticks_freq: int = 1,
        clear: bool = True,
    ) -> None:
        if clear:
            clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(5.7, 5))
        plt.rc("font", size=18)
        plt.rc("lines", linewidth=3)

        if self.zero_centered:
            abs_max = np.max(np.abs(self.values))
            kwargs = {"cmap": "PRGn", "vmin": -abs_max, "vmax": abs_max}
        else:
            kwargs = {}

        colors = ax.pcolormesh(
            self.grid_dimension_one, self.grid_dimension_two, self.values.T, shading="nearest", **kwargs
        )

        ax.set_xticks(self.dimension_one[::ticks_freq])
        ax.set_xticklabels(np.around(self.dimension_one[::ticks_freq], 1), rotation="vertical")
        ax.set_xlim(self.dimension_one[0], self.dimension_one[-1])
        ax.set_xlabel(xlabel)

        ax.set_yticks(self.dimension_two[::ticks_freq])
        ax.set_yticklabels(np.around(self.dimension_two[::ticks_freq], 1))
        ax.set_ylim(self.dimension_two[0], self.dimension_two[-1])
        ax.set_ylabel(ylabel)

        if self.axis_equal:
            ax.set_aspect("equal", "box")
        if title != "":
            ax.set_title(title)

        fig.colorbar(colors, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        if plot:
            plt.show()
        time.sleep(self.sleeping_time)

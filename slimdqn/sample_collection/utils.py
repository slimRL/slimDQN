import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def define_boxes(p: dict, env):
    env_name = p.get("env_name")
    n_states_1 = p.get("n_states_1")
    n_states_2 = p.get("n_states_2")
    target_func_name = f"define_boxes_{env_name}"

    if target_func_name in globals():
        return globals()[target_func_name](env, n_states_1, n_states_2)
    else:
        raise ValueError(f"No definition found for: {target_func_name}")

def define_boxes_lunar_lander(env, n_states_x: int, n_states_y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_max, y_max = env.env.observation_space.high[:2]
    states_x = np.linspace(-x_max, x_max, n_states_x)
    boxes_x_size = (2 * x_max) / (n_states_x - 1)
    states_x_boxes = np.linspace(-x_max, x_max + boxes_x_size, n_states_x + 1) - boxes_x_size / 2


    states_y = np.linspace(-y_max, y_max, n_states_y)
    boxes_y_size = (2 * y_max) / (n_states_y - 1)
    states_y_boxes = np.linspace(-y_max, y_max + boxes_y_size, n_states_y + 1) - boxes_y_size / 2

    return states_x, states_x_boxes, states_y, states_y_boxes

def define_boxes_car_on_hill(env, n_states_x: int, n_states_v: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

        plot_values = self.values.copy()
        plot_values = np.ma.masked_equal(plot_values, 0)

        if self.zero_centered:
            #abs_max = np.max(np.abs(self.values))
            #kwargs = {"cmap": "PRGn", "vmin": -abs_max, "vmax": abs_max}
            v_min = np.nanmin(self.values)
            v_max = np.nanmax(self.values)

            if v_min > 0: v_min = 0
            if v_max < 0: v_max = 0

            if v_min == 0 and v_max == 0:
                v_min, v_max = -1, 1

            norm = mcolors.TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
            kwargs = {"cmap": "PRGn", "norm": norm}
        else:
            kwargs = {}

        colors = ax.pcolormesh(
            self.grid_dimension_one, self.grid_dimension_two, plot_values.T, shading="nearest", **kwargs
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

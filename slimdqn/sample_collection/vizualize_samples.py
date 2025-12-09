import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.utils import TwoDimesionsMesh


def count_samples(
    dimension_one: np.ndarray,
    dimension_two: np.ndarray,
    discrete_dim_one_boxes: np.ndarray,
    discrete_dim_two_boxes: np.ndarray,
    rewards: np.ndarray,
) -> tuple:
    # for each element of dimension one, get the index where it is located in the discrete dimension.
    dimension_one = np.array(dimension_one).reshape(-1)
    indexes_dim_one_boxes = np.searchsorted(discrete_dim_one_boxes, dimension_one) - 1

    # for each element of dimension two, get the index where it is located in the discrete dimension.
    dimension_two = np.array(dimension_two).reshape(-1)
    indexes_dim_two_boxes = np.searchsorted(discrete_dim_two_boxes, dimension_two) - 1

    # only count the element pairs that are in the boxes
    dim_one_inside_boxes = np.logical_and(
        dimension_one >= discrete_dim_one_boxes[0],
        dimension_one <= discrete_dim_one_boxes[-1],
    )
    dim_two_inside_boxes = np.logical_and(
        dimension_two >= discrete_dim_two_boxes[0],
        dimension_two <= discrete_dim_two_boxes[-1],
    )
    dimensions_inside_boxes = np.logical_and(dim_one_inside_boxes, dim_two_inside_boxes)

    pruned_rewards = rewards.reshape(-1)[dimensions_inside_boxes]

    samples_count = np.zeros(
        (len(discrete_dim_one_boxes) - 1, len(discrete_dim_two_boxes) - 1)
    )
    rewards_count = np.zeros(
        (len(discrete_dim_one_boxes) - 1, len(discrete_dim_two_boxes) - 1)
    )

    indexes_dim = np.vstack(
        (
            indexes_dim_one_boxes[dimensions_inside_boxes],
            indexes_dim_two_boxes[dimensions_inside_boxes],
        )
    ).T

    for idx_in_list, (idx_dim_one, idx_dim_two) in enumerate(tqdm(indexes_dim)):
        samples_count[idx_dim_one, idx_dim_two] += 1
        rewards_count[idx_dim_one, idx_dim_two] += pruned_rewards[idx_in_list]

    return samples_count, (~dimensions_inside_boxes).sum(), rewards_count


def get_states(rb: ReplayBuffer):
    if not rb.memory:
        return np.array([])
    unpacked_states = [element.unpack().state for element in rb.memory.values()]
    unpacked_rewards = [element.unpack().reward for element in rb.memory.values()]
    return np.stack(unpacked_states), np.stack(unpacked_rewards)


def count_samples_and_plot(
    rb: ReplayBuffer,
    p: dict,
    states_x: np.ndarray,
    states_x_boxes: np.ndarray,
    states_v: np.ndarray,
    states_v_boxes: np.ndarray,
):
    states, rewards = get_states(rb)

    samples_count, _, _ = count_samples(
        states[:, 0],
        states[:, 1],
        states_x_boxes,
        states_v_boxes,
        rewards,
    )

    path = f"figures/{p['env_name']}/figures/{p['experiment_name']}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    (
        np.save(os.path.join(path,"samples_count.npy"), samples_count)
        if p["save_sample_mask"]
        else print("-------- Save Sample Counts False --------\n")
    )

    q_visu_mesh = TwoDimesionsMesh(
        states_x, states_v, sleeping_time=0, axis_equal=False, zero_centered=True
    )
    q_visu_mesh.set_values(samples_count)
    q_visu_mesh.show("", xlabel="x", ylabel="v", plot=False, ticks_freq=2)
    plt.savefig(
        f"figures/{p['env_name']}/figures/{p['experiment_name']}/samples_count.pdf",
        bbox_inches="tight",
    )

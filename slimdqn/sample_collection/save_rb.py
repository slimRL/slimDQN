import gzip
import os
import numpy as np
from tqdm import tqdm
import random


def _generate_filename(checkpoint_dir, attr, idx_iteration, extension):
    return os.path.join(checkpoint_dir, str(idx_iteration), f"{attr}.{extension}")


def save_rb(p, rb, epoch_idx, save_ratio=0.1):

    algo_name = p.get("algo_name", "dqn")
    capacity = p["replay_buffer_capacity"]
    seed = p["seed"]
    env_name = p["env_name"]
    exp_name = p["experiment_name"]

    base_dir = os.path.join(os.getcwd(), "data", env_name, f"{exp_name}/{algo_name}/rb_capacity_{capacity}/{str(seed)}")
    save_dir = os.path.join(base_dir, f"epoch_{epoch_idx}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    observations = []
    o_tm1_indices = []
    o_tm1_stack_sizes = []
    o_t_indices = []
    o_t_stack_sizes = []
    actions = []
    rewards = []
    is_terminals = []

    #randomly select a subset of keys based on save_ratio
    all_keys = list(rb.memory.keys())
    n_samples_to_save = int(len(all_keys) * save_ratio)
    selected_keys = sorted(random.sample(all_keys, n_samples_to_save)) # use fixed seed for exact reproducability
    current_obs_ptr = 0

    for k in tqdm(selected_keys, desc=f"Saving Epoch {epoch_idx}"):
        element = rb.memory[k].unpack()

        state = element.state
        next_state = element.next_state

        # Online Buffer stores (2, 1). Offline Dataset needs (2,).
        # Squeeze the last dimension if it is 1 (Standard for CarOnHill in this codebase)
        if state.ndim == 2 and state.shape[1] == 1:
            state = state.squeeze(1)
        if next_state.ndim == 2 and next_state.shape[1] == 1:
            next_state = next_state.squeeze(1)

        observations.append(state)
        o_tm1_indices.append(current_obs_ptr)
        o_tm1_stack_sizes.append(1)
        current_obs_ptr += 1

        observations.append(next_state)
        o_t_indices.append(current_obs_ptr)
        o_t_stack_sizes.append(1)
        current_obs_ptr += 1

        actions.append(element.action)
        rewards.append(element.reward)
        is_terminals.append(element.is_terminal)

    dataset_components = {}
    dataset_components["o_tm1_indices"] = np.array(o_tm1_indices, dtype=np.int32)
    dataset_components["o_tm1_stack_sizes"] = np.array(o_tm1_stack_sizes, dtype=np.int8)

    dataset_components["o_t_indices"] = np.array(o_t_indices, dtype=np.int32)
    dataset_components["o_t_stack_sizes"] = np.array(o_t_stack_sizes, dtype=np.int8)

    dataset_components["actions"] = np.array(actions, dtype=np.int32)
    dataset_components["rewards"] = np.array(rewards, dtype=np.float32)
    dataset_components["is_terminals"] = np.array(is_terminals, dtype=np.int8)

    # Shape (total N_samples * 2, 2)
    dataset_components["observations"] = np.array(observations)

    for attr in [
        "o_tm1_indices",
        "o_tm1_stack_sizes",
        "o_t_indices",
        "o_t_stack_sizes",
        "actions",
        "rewards",
        "is_terminals",
    ]:
        filename = os.path.join(save_dir, f"{attr}.gz")
        with open(filename, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb") as outfile:
                np.save(outfile, dataset_components[attr])

    filename = os.path.join(save_dir, "observations.gz")
    with open(filename, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb") as outfile:
            np.save(outfile, dataset_components["observations"])


import gzip
import os
import numpy as np
from tqdm import tqdm


def save_rb(p, rb, epoch_idx):
    save_dir = os.path.join(
        "data",
        p["env_name"],
        p["experiment_name"],
        p["algo_name"],
        f"rb_capacity_{p['replay_buffer_capacity']}",
        str(p["seed"]),
        str(epoch_idx),
    )
    print(f"Saving RB to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    observations = []
    o_tm1_indices = []
    o_tm1_stack_sizes = []
    o_t_indices = []
    o_t_stack_sizes = []
    actions = []
    rewards = []
    is_terminals = []

    # randomly select a subset of keys based on save_ratio
    current_obs_ptr = 0

    for k in tqdm(rb.memory.keys(), desc=f"Saving Epoch {epoch_idx}"):
        element = rb.memory[k].unpack()

        # squeeze to remove the stack size
        observations.append(element.state.squeeze())
        o_tm1_indices.append(current_obs_ptr)
        o_tm1_stack_sizes.append(1)
        current_obs_ptr += 1

        observations.append(element.next_state.squeeze())
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

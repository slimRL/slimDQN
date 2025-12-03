from slimdqn.sample_collection.replay_buffer import ReplayBuffer
import numpy as np
import os
import tqdm
import gzip

def save_rb(p: dict, rb: ReplayBuffer):


    algo_name = p.get("algo_name", "dqn")
    capacity = p["replay_buffer_capacity"]
    seed = p["seed"]
    env_name = p["env_name"]
    # Assuming 'data' folder is at the project root relative to this script
    base_dir = os.path.join(os.getcwd(), "data", env_name, f"{algo_name}_{capacity}", str(seed))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        print(f"Saving Offline Dataset to: {base_dir} ...")

    sorted_keys = sorted(rb.memory.keys())

    actions = []
    rewards = []
    is_terminals = []

    observations = []
    o_tm1_indices = []
    o_t_indices = []

    current_obs_index = 0

    for k in tqdm(sorted_keys, desc="Packing Replay Buffer"):
        # Unpack decompresses the snappy compression
        element = rb.memory[k].unpack()

        # 1. Standard Fields
        actions.append(element.action)
        rewards.append(element.reward)
        is_terminals.append(element.is_terminal)

        # 2. Observations & Indices
        # Note: element.state shape is usually (Obs, Stack).
        # For CarOnHill Stack=1, we verify and squeeze if necessary for 'raw' storage,
        # or keep it as is. Usually offline loaders expect (Obs, Stack) or (Obs,).
        # We will save exactly what is in the buffer.

        # Store Current State (s_t)
        observations.append(element.state)
        o_tm1_indices.append(current_obs_index)
        current_obs_index += 1

        # Store Next State (s_t+1)
        observations.append(element.next_state)
        o_t_indices.append(current_obs_index)
        current_obs_index += 1

    # 3. Convert to Numpy Arrays
    # Stack list into arrays
    actions_np = np.array(actions, dtype=np.int32)
    rewards_np = np.array(rewards, dtype=np.float32)
    is_terminals_np = np.array(is_terminals, dtype=np.int8)  # bool to int

    observations_np = np.array(observations)
    o_tm1_indices_np = np.array(o_tm1_indices, dtype=np.int32)
    o_t_indices_np = np.array(o_t_indices, dtype=np.int32)

    # Metadata: Stack size of the SAVED data (which is 1 based on your DQN config)
    # The Offline loader will likely load this and then restack to 4.
    stack_size_np = np.array([1], dtype=np.int32)

    # 4. Helper to save GZ
    def save_gz(name, array):
        path = os.path.join(base_dir, name)
        with open(path, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb") as outfile:
                np.save(outfile, array)

    # 5. Save Files
    save_gz("actions.gz", actions_np)
    save_gz("rewards.gz", rewards_np)
    save_gz("is_terminals.gz", is_terminals_np)
    save_gz("observations.gz", observations_np)
    save_gz("o_tm1_indices.gz", o_tm1_indices_np)
    save_gz("o_t_indices.gz", o_t_indices_np)
    save_gz("o_t_stack_size.gz", stack_size_np)

    print("Dataset saved successfully.")
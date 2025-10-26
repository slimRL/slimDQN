import argparse
import json
import os
import pickle
import time
from typing import List
import wandb

from experiments.base import parser_argument


def prepare_logs(env_name: str, algo_name: str, argvs: List[str]):
    print(
        f"---- Train {algo_name} on {env_name} {time.strftime('%d-%m-%Y %H:%M:%S')} ----",
        flush=True,
    )

    parser = argparse.ArgumentParser(f"Train {algo_name} on {env_name}.")
    shared_params = parser_argument.__dict__["add_base_arguments"](parser)
    agent_params = parser_argument.__dict__[f"add_{algo_name}_arguments"](parser)
    p = vars(parser.parse_args(argvs))
    p["env_name"] = env_name
    if env_name == "atari":
        p["game_name"] = p["experiment_name"].split("_")[-1]
    p["algo_name"] = algo_name
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo_name']}",
    )

    check_experiment(p)
    store_params(p, shared_params, agent_params)

    p["wandb"] = wandb.init(
        project="slimdqn",
        config=p,
        name=str(p["seed"]),
        group=f"{p['algo_name']}_{p['experiment_name']}",
        mode="online" if not p["disable_wandb"] else "disabled",
        settings=wandb.Settings(_disable_stats=True),
    )

    return p


def check_experiment(p: dict):
    # check if the experiment has been run already
    returns_path = os.path.join(p["save_path"], "episode_returns_and_lengths", str(p["seed"]) + ".npy")
    model_path = os.path.join(p["save_path"], "models", str(p["seed"]))
    assert not (
        os.path.exists(returns_path) or os.path.exists(model_path)
    ), "Same algorithm with same seed results already exists. Delete them and restart, or change the experiment name."

    # parameters.json is outside the algorithm folder (in the experiment folder)
    params_path = os.path.join(os.path.split(p["save_path"])[0], "parameters.json")

    # check if some parameters are different than the existing ones
    if os.path.exists(params_path):
        # PS: when many seeds are launched at the same time, the params exist but they are still being dumped.
        #     This is why a json.JSONDecodeError might be raised, in which case no check need to be made.
        try:
            loaded_old_params = json.load(open(params_path, "r"))
            old_params = loaded_old_params["shared_parameters"]
            if p["algo_name"] in list(loaded_old_params.keys()):
                old_params.update(loaded_old_params[p["algo_name"]])

            for param in p:
                if param in old_params:
                    assert (
                        old_params[param] == p[param]
                    ), f"The same experiment has been run with {param} = {old_params[param]} instead of {p[param]}. Change the experiment name."
        except json.JSONDecodeError:
            pass
    # if the folder has no parameters.json file and exists for a long time then raise an error
    else:
        if (
            os.path.exists(os.path.join(p["save_path"], ".."))
            and (time.time() - os.path.getmtime(os.path.join(p["save_path"], ".."))) > 4
        ):
            assert (
                False
            ), f"{p['save_path']} exists but has no parameters.json. Delete the folder and restart, or change the experiment name."


def store_params(p: dict, shared_params: List[str], agent_params: List[str]):
    os.makedirs(p["save_path"], exist_ok=True)
    # parameters.json are stored outside the algorithm folder (in the experiment folder)
    params_path = os.path.join(p["save_path"], "..", "parameters.json")

    # collect the existing parameters
    if os.path.exists(params_path):
        # PS: when many seeds are launched at the same time, the params exist but they are still being dumped.
        #     This is why a json.JSONDecodeError might be raised, in which case we wait until the parameters are dumped.
        loaded = False
        n_attempts = 0
        while not loaded and n_attempts <= 10:
            n_attempts += 1
            try:
                params_dict = json.load(open(params_path, "r"))
                loaded = True
            except json.JSONDecodeError:
                pass
        if not loaded:  # the file might be corrupted
            print("!!!! The file parameters.json might be corrupted. It has been deleted. A new file will be created.")
            os.remove(params_path)
            return store_params(p, shared_params, agent_params)
    else:
        params_dict = {}

        params_dict["shared_parameters"] = {}
        for shared_param in shared_params:
            if shared_param not in ["seed", "disable_wandb"]:
                params_dict["shared_parameters"][shared_param] = p[shared_param]

    # if the algorithms parameters were not stored in a previous run, we store them.
    if p["algo_name"] not in params_dict.keys():
        params_dict[p["algo_name"]] = {}
        for agent_param in agent_params:
            params_dict[p["algo_name"]][agent_param] = p[agent_param]

    # sort keys in a ordered manner
    ordered_params_dict = {
        algo_name: params_dict[algo_name] for algo_name in ["shared_parameters"] + sorted(list(params_dict.keys())[1:])
    }

    json.dump(ordered_params_dict, open(params_path, "w"), indent=4)


def save_data(p: dict, episode_returns: list, episode_lengths: list, model):
    os.makedirs(os.path.join(p["save_path"], "episode_returns_and_lengths"), exist_ok=True)
    episode_returns_and_lengths_path = os.path.join(p["save_path"], f"episode_returns_and_lengths/{p['seed']}.json")
    os.makedirs(os.path.join(p["save_path"], "models"), exist_ok=True)
    model_path = os.path.join(p["save_path"], f"models/{p['seed']}")

    json.dump(
        {"episode_lengths": episode_lengths, "episode_returns": episode_returns},
        open(episode_returns_and_lengths_path, "w"),
        indent=4,
    )
    pickle.dump(model, open(model_path, "wb"))

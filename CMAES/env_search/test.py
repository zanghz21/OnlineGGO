import os
import json
import gin
from env_search.utils import (
    load_pibt_default_config
)
import copy
import numpy as np
from simulators.wppl import py_driver as wppl_py_driver

map_path = "maps/competition/expert_baseline/pibt_maze-32-32-4_flow_baseline.json"
with open(map_path, 'r') as f:
    map_json = json.load(f)
    
all_weights = map_json["weights"]

kwargs = {
    "weights":
        json.dumps(all_weights),
    "map_json_path": "maps/competition/expert_baseline/pibt_maze-32-32-4_flow_baseline.json",
    "simulation_steps": 1000,
    # Defaults
    "gen_random":
        True,
    "num_tasks":
        100000,
    "plan_time_limit":
        1,
    "preprocess_time_limit":
        1800,
    "file_storage_path":
        "large_files",
    "task_assignment_strategy":
        "roundrobin",
    "num_tasks_reveal":
        1,
    "config":
        load_pibt_default_config(),  # Use PIBT default config
}

def single_simulation(seed, agent_num, kwargs, results_dir):
    num_task_finished = 0
    # kwargs["output"] = output_dir
    kwargs["num_agents"] = agent_num
    
    interval = 100
    left_timestep = kwargs["simulation_steps"]
    
    seed_rng = np.random.default_rng(seed)

    all_results = []    
    run_kwargs = copy.deepcopy(kwargs)
    while left_timestep > 0:
        run_kwargs["seed"] = seed_rng.integers(100000)
        run_kwargs["simulation_steps"] = min(interval, left_timestep)
        print("t=", run_kwargs["simulation_steps"])
        
        result_jsonstr = wppl_py_driver.run(**run_kwargs)
        result_json = json.loads(result_jsonstr)
        print("tp=", result_json["throughput"])
        
        run_kwargs["init_agent"] = True
        run_kwargs["init_agent_pos"] = str(result_json["final_pos"])
        run_kwargs["init_task"] = True
        run_kwargs["init_task_ids"] = str(result_json["final_tasks"])
        
        num_task_finished += result_json["throughput"] * run_kwargs["simulation_steps"]
        left_timestep -= run_kwargs["simulation_steps"]
        
        all_results.append(result_json)
        
    print(num_task_finished, result_json.keys())
    print(type(result_json["edge_pair_usage"]))
    # print(result_json)
    
    
    return num_task_finished

if __name__ == "__main__":
    single_simulation(seed=0, agent_num=10, kwargs=kwargs, results_dir='.')
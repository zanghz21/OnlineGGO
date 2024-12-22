import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
import gin.config
import numpy as np
import time
import gin
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import logging
from datetime import datetime
import csv
import copy
import multiprocessing
from itertools import repeat
from simulators.sum_ovc import py_driver as base_py_driver
from simulators.base_lns import py_driver as lns_py_driver

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])
logger = logging.getLogger()

def get_time_str():
    now = datetime.now()
    time_str = now.strftime('%y%m%d_%H%M%S')
    return time_str


EXP_AGENTS={
    "game": [2000, 4000, 6000, 8000, 10000, 12000], 
    # "game": [12000], 
    # "room": [100, 500, 1000, 1500, 2000, 2500, 3000], 
    "room": [100, 200, 300, 400, 500, 600], 
    "sortation_small": [200, 400, 600, 800, 1000, 1200, 1400],
    "warehouse_large": [2000, 4000, 6000, 8000, 10000, 12000], 
    "ggo33x36": [200, 300, 400, 500, 600, 700, 800], 
    "random": [100, 200, 300, 400, 500, 600, 700, 800], 
    "warehouse_small_narrow": [200, 300, 400, 500, 600, 700, 800, 900, 1000], 
    "warehouse_60x100": [200, 600, 1000, 1400, 1800, 2200, 2600, 3000, 3400],
    "empty": [200, 300, 400, 500, 600, 700, 800, 900]
}

EXP_AGENTS_RUNTIME = {
    "sortation_small": [800],
    "random": [400], 
    "warehouse_small_narrow": [600], 
    "empty": [400], 
    "warehouse_60x100": [1800], 
    "game": [6000]
}

EXP_MAPPATHS={
    # "room": "guided-pibt/benchmark-lifelong/maps/room-64-64-8.map", 
    "game": "guided-pibt/benchmark-lifelong/maps/ost003d.map", 
    "room": "guided-pibt/benchmark-lifelong/maps/room-32-32-4.map", 
    "sortation_small": "guided-pibt/benchmark-lifelong/maps/sortation_small_kiva.map", 
    "ggo33x36": "guided-pibt/benchmark-lifelong/maps/ggo_maps/33x36.map", 
    "random": "guided-pibt/benchmark-lifelong/maps/pibt_random_unweight_32x32.map",
    "warehouse_small_narrow": "guided-pibt/benchmark-lifelong/maps/warehouse_small_narrow_kiva.map", 
    "warehouse_60x100": "guided-pibt/benchmark-lifelong/maps/warehouse_60x100_kiva.map", 
    "empty": "guided-pibt/benchmark-lifelong/maps/empty-32-32.map"
}

WAREHOUSE_MAPS = ["sortation_small", "warehouse_small_narrow", "warehouse_60x100"]

def get_default_kwargs(rand_task_dist, exp_name, rand_type = "Gaussian"):
    kwargs = {
        "gen_tasks": True,
        "num_agents": 400, 
        "num_tasks": 100000,
        "seed": 0,  
        "task_assignment_strategy": "roundrobin", 
        "simu_time": 1000, 
    }
    if rand_task_dist:
        kwargs["task_dist_change_interval"] = 200
        kwargs["task_random_type"] = rand_type
        kwargs["task_assignment_strategy"] = "online_generate"
        if exp_name in WAREHOUSE_MAPS:
            kwargs["dist_sigma"] = 1.0
        else:
            kwargs["dist_sigma"] = 0.5
            kwargs["dist_K"] = 3
    return kwargs

def single_experients(base_kwargs, num_agents, seed, base_save_dir, timestr):
    save_dir = os.path.join(base_save_dir, f"ag{num_agents}", f"{seed}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestr}.json")
    
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["seed"] = seed
    kwargs["num_agents"] = num_agents
    simulator = lns_py_driver if kwargs["use_lns"] else base_py_driver
    kwargs["gen_tasks"] = True
    
    t = time.time()
    result_json_s = simulator.run(**kwargs)
    result_json = json.loads(result_json_s)
    sim_time = time.time()-t
    tp = result_json["throughput"]
    
    save_data = {
        "sim_time": sim_time, 
        "throughput": tp
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f)
    return save_data
    
    
def main(use_lns, exp_name, rand_task_dist, n_workers, n_evals, is_runtime, all_results_dir):
    if not (exp_name in WAREHOUSE_MAPS):
        rand_type = "GaussianMixed"
    else:
        rand_type = "Gaussian"
    
    kwargs = get_default_kwargs(rand_task_dist, exp_name, rand_type=rand_type)
    if use_lns:
        algo = "baseLNS"
    else:
        algo = "SUM_OVC"
        
    kwargs["use_lns"] = use_lns
    kwargs["map_path"] = EXP_MAPPATHS[exp_name]
    
    if all_results_dir is None:
        all_results_dir = "../results_multi_process" if not rand_task_dist else "../results_multi_process_dist"
    save_dir = os.path.join(all_results_dir, "Guided-PIBT", algo, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    timestr = get_time_str()
    
    if not is_runtime:
        agent_ls = EXP_AGENTS[exp_name]
    else:
        agent_ls = EXP_AGENTS_RUNTIME[exp_name]
    
    pool = multiprocessing.Pool(n_workers)
    
    exp_agent_ls = []
    exp_seed_ls = []
    for a in agent_ls:
        for s in range(n_evals):
            exp_agent_ls.append(a)
            exp_seed_ls.append(s)
    
    n_simulations = len(exp_agent_ls)
    all_data = pool.starmap(
        single_experients, 
        zip(
            repeat(kwargs, n_simulations), 
            exp_agent_ls, 
            exp_seed_ls, 
            repeat(save_dir, n_simulations), 
            repeat(timestr, n_simulations)
        )
    )
    
    if is_runtime:
        collected_data = {"sim_time": [], "throughput": []}
        for data in all_data:
            for key, v in data.items():
                if key not in collected_data.keys():
                    print(key)
                    raise NotImplementedError
                collected_data[key].append(v)
                
        for key, v in collected_data.items():
            print(f"key={key}, avg={np.mean(v)}, std={np.std(v, ddof=1)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--use_lns', action="store_true", default=False)
    p.add_argument("--exp_name", type=str, choices=["room", "random", "ggo33x36", "sortation_small", "warehouse_small_narrow", "empty", "warehouse_60x100", "game"])
    p.add_argument("--rand_task_dist", action="store_true", default=False)
    p.add_argument('--n_workers', type=int, required=True)
    p.add_argument('--n_evals', type=int, required=True)
    p.add_argument('--is_runtime', action="store_true", default=False)
    p.add_argument('--all_results_dir', type=str, default=None)
    cfg = p.parse_args()

    main(use_lns=cfg.use_lns, exp_name=cfg.exp_name, rand_task_dist=cfg.rand_task_dist, n_workers=cfg.n_workers, n_evals=cfg.n_evals, is_runtime=cfg.is_runtime, all_results_dir=cfg.all_results_dir)
    
    
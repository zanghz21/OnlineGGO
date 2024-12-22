import argparse
import json
import os
import shutil
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.module import TrafficMAPFModule
from env_search.utils.logging import get_current_time_str
from env_search.utils.plot_utils import parse_save_file

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import numpy as np
import gin
import logging
import copy
import time

from simulators.trafficMAPF_lns import py_driver as lns_py_driver
from simulators.trafficMAPF import py_driver as base_py_driver
from simulators.trafficMAPF_off import py_driver as off_py_driver
from simulators.trafficMAPF_off_lns import py_driver as off_py_driver_lns


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])

def single_experients(base_kwargs, num_agents, seed, logdir, save_files):
    save_dir = os.path.join(logdir, "vis")
    os.makedirs(save_dir, exist_ok=True)
    
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["seed"] = seed
    kwargs["num_agents"] = num_agents
    simulator = lns_py_driver if kwargs["use_lns"] else base_py_driver
    kwargs["gen_tasks"] = True
    kwargs["simu_time"] = 10000
    
    save_path = os.path.join(save_dir, f"results_{seed}.json")
    if save_files:
        kwargs["save_path"] = save_path
    
    t = time.time()
    result_json_s = simulator.run(**kwargs)
    result_json = json.loads(result_json_s)
    sim_time = time.time()-t
    tp = result_json["throughput"]
    
    parse_save_file(file_path=save_path, timestep_start=1, 
                    timestep_end=kwargs["simu_time"]+1, 
                    seed=seed, 
                    save_dir=save_dir)
    save_data = {
        "sim_time": sim_time, 
        "throughput": tp
    }
    return save_data

def main(log_dir, seed, save_files, eval_lns):
    log_dir = log_dir[:-1] if log_dir.endswith('/') else log_dir
    
    net_file = os.path.join(log_dir, "optimal_update_model.json")
    with open(net_file, "r") as f:
        weights_json = json.load(f)
    network_params = weights_json["params"]
    
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    eval_config = TrafficMAPFConfig()
    eval_module = TrafficMAPFModule(eval_config)
    kwargs = eval_module.gen_sim_kwargs(nn_weights_list=network_params)
    kwargs["seed"] = 0
    kwargs["use_lns"] = eval_config.use_lns or eval_lns
    
    single_experients(kwargs, eval_config.num_agents, seed, log_dir, 
                        save_files)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--save_files', action="store_true", default=False)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval_lns', action="store_true", default=False)
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir, seed=cfg.seed, save_files=cfg.save_files, eval_lns=cfg.eval_lns)

    
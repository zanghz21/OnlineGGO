import json
import os
import gin.config
import numpy as np
import time
import gin
from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.module import TrafficMAPFModule
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import logging
from env_search.traffic_mapf.utils import get_map_name
from datetime import datetime
import csv

from simulators.trafficMAPF_lns import py_driver as lns_py_driver
from simulators.trafficMAPF import py_driver as base_py_driver

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])
logger = logging.getLogger()

def get_time_str():
    now = datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    return time_str


EXP_AGENTS={
    "game": [2000, 4000, 6000, 8000, 10000, 12000], 
    "room": [500, 1000, 1500, 2000, 2500, 3000], 
    "sortation_small": [200, 400, 600, 800, 1000, 1200, 1400],
    "warehouse_large": [2000, 4000, 6000, 8000, 10000, 12000]
}

def experiments(base_kwargs, exp_name, exp_dir):
    time_str = get_time_str()
    
    exp_logger_dir = os.path.join(exp_dir, "logs")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_logger_dir, exist_ok=True)
    
    file_hd = logging.FileHandler(os.path.join(exp_logger_dir, f'{time_str}.log'))
    file_hd.setLevel(logging.INFO)
    logger.addHandler(file_hd)
    
    simulator = lns_py_driver if base_kwargs["use_lns"] else base_py_driver
    base_kwargs["gen_tasks"] = True
    agent_ls = EXP_AGENTS[exp_name]
    for ag in agent_ls:
        agent_log_dir = os.path.join(exp_dir, f"ag{ag}")
        os.makedirs(agent_log_dir, exist_ok=True)
        
        file_name = os.path.join(agent_log_dir, f"{time_str}.csv")
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['exp_time', 'tp', 'sim_time'])
            
        for i in range(51):
            if i == 24:
                continue
            base_kwargs["seed"] = i
            base_kwargs["num_agents"] = ag
            t = time.time()
            result_json_s = simulator.run(**base_kwargs)
            result_json = json.loads(result_json_s)
            sim_time = time.time()-t
            tp = result_json["throughput"]
            
            with open(file_name, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow([i, tp, sim_time])
            
            print("sim_time = ", sim_time)
            print(result_json_s)
            logger.info(f"ag={ag}, \texp={i}, \ttp={tp}, \tsim_time={sim_time}")

    
def main(log_dir, eval_lns=False):
    log_dir = log_dir[:-1] if log_dir.endswith('/') else log_dir
    net_file = os.path.join(log_dir, "optimal_update_model.json")
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    eval_config = TrafficMAPFConfig()
    eval_module = TrafficMAPFModule(eval_config)

    with open(net_file, "r") as f:
        weights_json = json.load(f)
    network_params = weights_json["params"]

    if eval_config.use_lns:
        algo = "NN_train_lns"
    elif eval_lns:
        algo = "NN_eval_lns"
    else:
        algo = "NN_no_lns"
    
    kwargs = eval_module.gen_sim_kwargs(nn_weights_list=network_params)
    kwargs["seed"] = 0
    kwargs["use_lns"] = eval_config.use_lns
    
    train_map = kwargs["map_path"]
    if "ost" in train_map:
        exp_name = "game"
    elif "sortation" in train_map:
        exp_name = "sortation_small"
    elif "warehouse" in train_map:
        exp_name = "warehouse_large"
    elif "room" in train_map:
        exp_name = "room"
    else:
        print(train_map)
        raise NotImplementedError
    
    log_name = log_dir.split('/')[-1]
    save_dir = os.path.join("../results/Guided-PIBT", algo, exp_name, log_name)
    os.makedirs(save_dir, exist_ok=True)
    
    experiments(kwargs, exp_name, save_dir)

    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir)
    
    
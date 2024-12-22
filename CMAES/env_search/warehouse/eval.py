import argparse
import os
import gin
import json
from env_search.warehouse.config import WarehouseConfig
from env_search.utils import get_n_valid_edges, get_n_valid_vertices, read_in_kiva_map, kiva_env_str2number
from env_search.warehouse.module import WarehouseModule
from env_search.warehouse.warehouse_manager import WarehouseManager
from env_search.iterative_update.envs.rhcr_online_env import WarehouseOnlineEnv
import numpy as np

def base_exp(cfg: WarehouseConfig, model_params, log_dir): 
    domain = "kiva"
    base_map_path = gin.query_parameter("WarehouseManager.base_map_path")
    num_agents = gin.query_parameter("WarehouseManager.agent_num")
    with open(base_map_path, 'r') as f:
        base_map_json = json.load(f)
        
    cfg.overallCutoffTime = 300
    module = WarehouseModule(cfg)
    
    base_map_str, _ = read_in_kiva_map(base_map_path)
    base_map_np = kiva_env_str2number(base_map_str)
    
    n_v = get_n_valid_vertices(base_map_np, domain)
    n_e = get_n_valid_edges(base_map_np, True, domain)
    eval_logdir = os.path.join(log_dir, "eval") # no use here
    
    tp_list = []
    for seed in range(5):
        res, _ = module.evaluate_online(base_map_np, base_map_json, num_agents, model_params, eval_logdir, n_e, n_v, seed)
        tp = res['throughput']
        print(f"seed = {seed}, tp = {tp}")
        tp_list.append(tp)
    print(np.mean(tp_list), np.std(tp_list))
        

def base_offline_exp(cfg: WarehouseConfig, weights, log_dir):
    domain = "kiva"
    base_map_path = gin.query_parameter("WarehouseManager.base_map_path")
    num_agents = gin.query_parameter("WarehouseManager.agent_num")
    with open(base_map_path, 'r') as f:
        base_map_json = json.load(f)
        
    cfg.overallCutoffTime = 300
    cfg.update_gg_interval = 1000
    cfg.simulation_time = 1000
    
    base_map_str, _ = read_in_kiva_map(base_map_path)
    base_map_np = kiva_env_str2number(base_map_str)
    
    n_v = get_n_valid_vertices(base_map_np, domain)
    n_e = get_n_valid_edges(base_map_np, True, domain)
    eval_logdir = os.path.join(log_dir, "eval") # no use here
    
    tp_list = []
    for seed in range(5):
        env = WarehouseOnlineEnv(
            base_map_np, base_map_json, num_agents, eval_logdir, n_v, n_e, cfg, seed
        )
        obs, info = env.reset()
        done = False
        while not done:
            obs, rew, terminated, truncated, info = env.step(weights)
            done = terminated or truncated
        tp = info["result"]['throughput']
        print(f"seed = {seed}, tp = {tp}")
        tp_list.append(tp)
    print(np.mean(tp_list), np.std(tp_list))


if __name__ == "__main__":
    cfg_file = "config/warehouse/online_update/33x36_dist.gin"
    gin.parse_config_file(cfg_file, skip_unknown=True)
    cfg = WarehouseConfig()
    
    model_params_file = ""
    with open(model_params_file, "r") as f:
        model_params_json = json.load(f)
    model_params = model_params_json["params"]
    model_params = np.array(model_params)
    
    base_exp(cfg, model_params, log_dir="test")
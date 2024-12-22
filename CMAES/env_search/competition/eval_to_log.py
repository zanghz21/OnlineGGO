import argparse
import json
import os
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule
from env_search.competition.competition_manager import CompetitionManager
from env_search.utils.logging import get_current_time_str

import numpy as np
import gin
import logging
import csv

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])

EXP_AGENTS = {
    "sortation_small": [200, 400, 600, 800, 1000, 1200, 1400],
    "ggo33x36": [200, 300, 400, 500, 600, 700, 800], 
    "random": [200, 300, 400, 500, 600, 700, 800], 
    "room": [100, 500, 1000, 1500, 2000, 2500, 3000]
}
def parse_config(log_dir):
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    config = CompetitionConfig()
    try:
        is_online = gin.query_parameter("CompetitionManager.online_update")
    except ValueError:
        is_online = False
    return config, is_online
    
def get_update_model(log_dir):
    update_model_file = os.path.join(log_dir, "optimal_update_model.json")
    with open(update_model_file, "r") as f:
        model_json = json.load(f)
        
    params = model_json["params"]
    model_params = np.array(params)
    return model_params

def get_offline_weights(log_dir):
    weights_file = os.path.join(log_dir, "optimal_weights.json")
    with open(weights_file, "r") as f:
        weights_json = json.load(f)
    weights = weights_json["weights"]
    return np.array(weights)

def parse_map(map_path):
    comp_map = Map(map_path)
    n_e = get_n_valid_edges(comp_map.graph, True, "competition")
    n_v = get_n_valid_vertices(comp_map.graph, "competition")
    return n_e, n_v


def vis_arr(arr_, mask=None, name="test"):
    arr = arr_.copy()
    save_dir = "env_search/competition/plots"
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    if mask is not None:
        arr = np.ma.masked_where(mask, arr)
    cmap = plt.cm.Reds
    # cmap.set_bad(color='black')
    plt.imshow(arr, cmap=cmap, interpolation="none")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()
    

def base_offline_exp(cfg: CompetitionConfig, log_dir, save_dir, map_type):
    time_str = get_current_time_str()
    
    cfg.h_update_late = False
    cfg.past_traffic_interval = 20
    cfg.simulation_time = 1000
    cfg.update_interval = 1000
    cfg.warmup_time = 100
    
    optimal_weights = get_offline_weights(log_dir)
    n_e, n_v = parse_map(cfg.map_path)
    
    from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
    
    ag_list = EXP_AGENTS[map_type]
    for ag in ag_list:
        cfg.num_agents = ag
        
        agent_log_dir = os.path.join(save_dir, f"ag{ag}")
        os.makedirs(agent_log_dir, exist_ok=True)
        file_name = os.path.join(agent_log_dir, f"{time_str}.csv")
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['exp_time', 'tp'])
            
        tp_list = []
        for seed in range(50):
            env = CompetitionOnlineEnvNew(n_v, n_e, cfg, seed=seed)
            obs, info = env.reset()
            done = False
            while not done:
                obs, rew, terminated, truncated, info = env.step(optimal_weights)
                done = terminated or truncated
            tp = info["result"]["throughput"]
            print(tp)
            tp_list.append(tp)
            with open(file_name, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow([seed, tp])
            
    print(np.mean(tp_list), np.std(tp_list))


def base_exp(cfg: CompetitionConfig, model_params, save_dir, map_type):
    time_str = get_current_time_str()
    
    n_e, n_v = parse_map(cfg.map_path)
   
    os.makedirs(save_dir, exist_ok=True)
    
    ag_list = EXP_AGENTS[map_type]
    for ag in ag_list:
        cfg.num_agents = ag
        module = CompetitionModule(cfg)
        
        agent_log_dir = os.path.join(save_dir, f"ag{ag}")
        os.makedirs(agent_log_dir, exist_ok=True)
        file_name = os.path.join(agent_log_dir, f"{time_str}.csv")
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['exp_time', 'tp'])
        
        tp_list = []   
        for seed in range(5):
            # res, _ = module.evaluate_iterative_update(model_params, eval_logdir, n_e, n_v, seed)
            res, _ = module.evaluate_online_update(model_params, "", n_e, n_v, seed, env_type="new")
            tp = res['throughput']
            print(f"seed = {seed}, tp = {tp}")
            tp_list.append(tp)
            with open(file_name, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow([seed, tp])
            
        print(np.mean(tp_list), np.std(tp_list))



def main(log_dir):
    log_dir = log_dir[:-1] if log_dir.endswith('/') else log_dir
    
    cfg, is_online = parse_config(log_dir)
    print("is_online:", is_online)
    
    base_save_dir = "../results/PIBT"
    map_type = "random"
    log_name = log_dir.split('/')[-1]
    if is_online:
        algo = "online"
        model_params = get_update_model(log_dir)
        save_dir = os.path.join(base_save_dir, algo, map_type, log_name)
        base_exp(cfg, model_params, save_dir, map_type)
    else:
        algo = "offline"
        save_dir = os.path.join(base_save_dir, algo, map_type, log_name)
        base_offline_exp(cfg, log_dir, save_dir, map_type)

    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir)
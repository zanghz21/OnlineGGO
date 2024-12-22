import argparse
import json
import os
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule
from env_search.utils.logging import get_current_time_str

import numpy as np
import gin
import logging

def parse_config(log_dir):
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    config = CompetitionConfig()
    return config
    
def get_update_model(log_dir):
    update_model_file = os.path.join(log_dir, "optimal_update_model.json")
    with open(update_model_file, "r") as f:
        model_json = json.load(f)
        
    params = model_json["params"]
    model_params = np.array(params)
    return model_params

def get_action(log_dir):
    action_file = os.path.join(log_dir, "action.json")
    with open(action_file, "r") as f:
        action_json = json.load(f)
    action = action_json["action"]
    return np.array(action)

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


def main(log_dirs):
    cfg = parse_config(log_dirs[0])
    cfg.simulation_time = 1000
    cfg.update_interval = 100
    cfg.warmup_time = 100
    cfg.task_dist_change_interval = -1
    cfg.past_traffic_interval = 100
    cfg.num_agents = 400
    cfg.h_update_late = False
    # cfg.reset_weights_path = os.path.join(log_dirs[2], "action.json")
    # cfg.iter_update_n_sim = 1
    # cfg.left_right_ratio = 0.1
    n_e, n_v = parse_map(cfg.map_path)
    
    
    from env_search.iterative_update.envs.online_env import CompetitionOnlineEnv
    from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
    from env_search.iterative_update.envs.env import CompetitionIterUpdateEnv
    
    from env_search.competition.update_model.update_model import CompetitionCNNUpdateModel
    
    env_old = CompetitionOnlineEnvNew(n_v, n_e, cfg, seed=1)
    
    old_obs, old_info = env_old.reset()    
    
    # raise NotImplementedError
    comp_map = Map(cfg.map_path)
    
    r_list = [0.1, 1.0, 10.0]
    # r_list = [0.1]
    # update_models = []
    actions = []
    for log_dir in log_dirs:
        action = get_action(log_dir)
        actions.append(action)
    
    done =False
    i=0
    
    
    while not done:
        obs = old_obs
        i+=1
        r_id = (i//4)%len(r_list)
        
        action = actions[r_id]
        # print(action.max(), action.min())
        # action = np.ones_like(action)
        
        # env_old.config.left_right_ratio = r_list[r_id]
        env_old.update_lr_weights(r_list[r_id])
        old_obs, rew, terminated, truncated, info = env_old.step(action)
        done = terminated or truncated
        # obs = off_obs
        
        # for j in range(5):
        # vis_arr(obs[4], name=f"usage_old_r{r_list[r_id]}_{i}")
        # vis_arr(off_obs[9], name=f"guidance_off_r{cfg.left_right_ratio}_{i}")
        
    print(info["result"]["throughput"])
    # env_old.generate_video()
    

def baseline_main(log_dir):
    cfg = parse_config(log_dir)
    # cfg.simulation_time = 5000
    # cfg.update_interval = 200
    # cfg.warmup_time = 100
    # cfg.task_dist_change_interval = -1
    # cfg.past_traffic_interval = 200
    # cfg.iter_update_n_sim = 1
    # cfg.left_right_ratio = 1.0
    n_e, n_v = parse_map(cfg.map_path)
    
    model_params = get_update_model(log_dir)
    from env_search.iterative_update.envs.online_env import CompetitionOnlineEnv
    from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
    from env_search.iterative_update.envs.env import CompetitionIterUpdateEnv
    
    from env_search.competition.update_model.update_model import CompetitionCNNUpdateModel
    
    # env_old = CompetitionOnlineEnv(n_v, n_e, cfg, seed=0)
    env_off = CompetitionIterUpdateEnv(n_v, n_e, cfg, seed=0)
    
    # old_obs, old_info = env_old.reset()
    off_obs, _ = env_off.reset()   
    
    # raise NotImplementedError
    comp_map = Map(cfg.map_path)
    
    r_list = [0.1, 1.0, 10.0]
    update_model = CompetitionCNNUpdateModel(comp_map, model_params, n_v, n_e)
    
    done =False
    i=0
    
    
    while not done:
        # obs = old_obs
        obs = off_obs
        
        i+=1
        
        r_id = (i//4)%3
        
        edge_usage_matrix = np.moveaxis(obs[:4], 0, 2)
        wait_usage_matrix = obs[4]
        curr_edge_weights_matrix = np.moveaxis(obs[5:9], 0, 2)
        curr_wait_costs_matrix = obs[9]

        # Get update value
        wait_cost_update_vals, edge_weight_update_vals = \
            update_model.get_update_values(
                wait_usage_matrix,
                edge_usage_matrix,
                curr_wait_costs_matrix,
                curr_edge_weights_matrix,
            )
        action = np.concatenate([wait_cost_update_vals, edge_weight_update_vals])
        print(action.max(), action.min())
        # old_obs, rew, terminated, truncated, info = env_old.step(action)
        off_obs, rew, terminated, truncated, info = env_off.step(action)
        done = terminated or truncated
        # obs = off_obs
        
        # for j in range(5):
        vis_arr(obs[4], name=f"usage_base_r{cfg.left_right_ratio}_{i}")
        # vis_arr(off_obs[9], name=f"guidance_off_r{cfg.left_right_ratio}_{i}")
        
    print(info["result"]["throughput"])
    print(action.max(), action.min())
    save_action_path = os.path.join(log_dir, "action.json")
    with open(save_action_path, "w") as f:
        json.dump({"action": action.tolist()}, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument('--logdir', type=str, required=True)
    # cfg = p.parse_args()
    
    log_dirs = ["logs/2024-07-16_19-42-46_overfit-ratio-r_Yzaw8d3u", "logs/2024-07-16_20-14-48_overfit-ratio-r_MaYs6qv4", "logs/2024-07-16_19-41-01_overfit-ratio-r_QFZwgMKH"]
    # log_dirs = log_dirs[:1]
    main(log_dirs)
    # baseline_main(log_dirs[2])
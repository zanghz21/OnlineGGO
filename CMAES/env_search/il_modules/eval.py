import argparse
import json
import os
from env_search.competition.update_model.utils import Map, comp_uncompress_edge_matrix, comp_uncompress_vertex_matrix
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule
from env_search.competition.update_model.update_model import CompetitionCNNUpdateModel, CNNMLPModel, LargerCNNUpdateModel
from env_search.utils.logging import get_current_time_str
from env_search.utils import min_max_normalize

import numpy as np
import gin
import logging
import torch

def parse_config(log_dir):
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    config = CompetitionConfig()
    return config

def parse_map(map_path):
    comp_map = Map(map_path)
    n_e = get_n_valid_edges(comp_map.graph, True, "competition")
    n_v = get_n_valid_vertices(comp_map.graph, "competition")
    return comp_map, n_e, n_v

def get_action(log_dir):
    action_file = os.path.join(log_dir, "action.json")
    with open(action_file, "r") as f:
        action_json = json.load(f)
    action = action_json["action"]
    return np.array(action)


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
    

def compute_save_action(comp_map: Map, n_v, n_e, expert_action, fill_value=-100):
    expert_action = min_max_normalize(expert_action, -10, 10)
    
    e_action = expert_action[n_v:]
    v_action = expert_action[:n_v]
    save_e_action = comp_uncompress_edge_matrix(comp_map, e_action, fill_value=fill_value)
    save_e_action = np.array(save_e_action).reshape(comp_map.height, comp_map.width, 4)
    save_e_action = np.moveaxis(save_e_action, 2, 0)
    
    save_v_action = comp_uncompress_vertex_matrix(comp_map, v_action, fill_value=fill_value)
    save_v_action = np.array(save_v_action).reshape(1, comp_map.height, comp_map.width)
    
    save_action = np.concatenate([save_e_action, save_v_action], axis=0)
    return save_action

    
def main(log_dirs, model: CompetitionCNNUpdateModel, data_dir):
    r_list = [0.1, 1.0, 10.0]
    r_id = np.random.randint(len(r_list))
    
    expert_actions = []
    for log_dir in log_dirs:
        e_action = get_action(log_dir)
        expert_actions.append(e_action)
    
    ### 1. cfg
    cfg = parse_config(log_dirs[0])
    cfg.simulation_time = 1000
    cfg.update_interval = 20
    cfg.warmup_time = 1
    cfg.task_dist_change_interval = -1
    cfg.past_traffic_interval = 100
    cfg.h_update_late = False
    cfg.has_task_obs = True
    # cfg.reset_weights_path = os.path.join(log_dirs[2], "action.json")
    # cfg.iter_update_n_sim = 1
    cfg.left_right_ratio = r_list[r_id]
    
    ### 2. env
    _, n_e, n_v = parse_map(cfg.map_path)
    from env_search.iterative_update.envs.online_env import CompetitionOnlineEnv
    env_old = CompetitionOnlineEnv(n_v, n_e, cfg, seed=0)
    
    ### 3. experiment
    for iter in range(10):
        obs, info = env_old.reset()
        
        done =False
        i=0
        
        while not done:
            i+=1
            
            if i%10 == 0:
                r_id = np.random.randint(len(r_list))
            
            obs = obs[-1:]
            wait_cost_update_vals, edge_weight_update_vals = model.get_update_values_from_obs(obs)
            action = np.concatenate([wait_cost_update_vals, edge_weight_update_vals])
            action = min_max_normalize(action, 0.1, 100)
            expert_action = expert_actions[r_id]
            expert_action = min_max_normalize(expert_action, 0.1, 100)
            error = np.abs(expert_action - action)
            # print(error)
            
            env_old.config.left_right_ratio = r_list[r_id]
            obs, rew, terminated, truncated, info = env_old.step(action)
            done = terminated or truncated
            
            
        print(info["result"]["throughput"])
    


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # logdir = "logs/2024-07-16_19-42-46_overfit-ratio-r_Yzaw8d3u"
    logdirs = ["logs/2024-07-16_19-42-46_overfit-ratio-r_Yzaw8d3u", "logs/2024-07-16_20-14-48_overfit-ratio-r_MaYs6qv4", "logs/2024-07-16_19-41-01_overfit-ratio-r_QFZwgMKH"]
    map_path = "maps/competition/human/pibt_warehouse_33x36_w_mode.json"
    il_ckpt_path = "il_results/240717_202925/ckpt/ckpt_final.pth"
    
    
    comp_map, n_e, n_v = parse_map(map_path)
    # student_model = CompetitionCNNUpdateModel(
    #     comp_map, 
    #     model_params=None, 
    #     n_valid_vertices=n_v,
    #     n_valid_edges=n_e, 
    #     nc=10, 
    # )
    il_ckpt_path = "il_results/240719_140527/ckpt/ckpt_final.pth"
    state_dict = torch.load(il_ckpt_path)
    student_model = CNNMLPModel(comp_map, nc=1)
    # student_model = LargerCNNUpdateModel(comp_map, None, n_v, n_e, 1, kernel_size=5)
    student_model.model.load_state_dict(state_dict["state_dict"])
    
    
    time_str = get_current_time_str()
    data_dir = os.path.join("data", time_str)
    main(logdirs, student_model, data_dir)
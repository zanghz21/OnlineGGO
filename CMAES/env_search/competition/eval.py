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

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])

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


def debug(cfg: CompetitionConfig, model_params, log_dir):
    cfg.simulation_time = 1000
    cfg.update_interval = 200
    cfg.warmup_time = 100
    cfg.task_dist_change_interval = -1
    cfg.past_traffic_interval = 1000
    cfg.iter_update_n_sim = 1
    cfg.left_right_ratio = 1.0
    n_e, n_v = parse_map(cfg.map_path)
    
    from env_search.iterative_update.envs.online_env import CompetitionOnlineEnv
    from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
    from env_search.iterative_update.envs.env import CompetitionIterUpdateEnv
    
    from env_search.competition.update_model.update_model import CompetitionCNNUpdateModel
    env_old = CompetitionOnlineEnv(n_v, n_e, cfg, seed=0)
    env_new = CompetitionOnlineEnvNew(n_v, n_e, cfg, seed=0)
    env_off = CompetitionIterUpdateEnv(n_v, n_e, cfg, seed=0)
    
    # old_obs, old_info = env_old.reset()
    # print("old:", old_info["result"]["final_pos"])
    # init_pos = old_info["result"]["final_pos"]
    
    # new_obs, info = env_new.reset()
    
    # raise NotImplementedError
    off_obs, off_info = env_off.reset()
    
    # with open("maps/competition/ours/pibt_warehouse-33x36_w_mode_cma-es_400_agents_four-way-move.json", "r") as f:
    #     weights_json = json.load(f)
    # weights = weights_json["weights"]
    # action = np.array(weights)
    
    
    # raise NotImplementedError
    comp_map = Map(cfg.map_path)
    update_model = CompetitionCNNUpdateModel(comp_map, model_params, n_v, n_e)
    rng = np.random.default_rng(seed=10)
    done =False
    i=0
    
    while not done:
        obs = off_obs
        # obs = new_obs
        i+=1
        wait_cost_update_vals, edge_weight_update_vals = update_model.get_update_values_from_obs(obs)
        action = np.concatenate([wait_cost_update_vals,
                                edge_weight_update_vals])
        print(action.max(), action.min())
        off_obs, rew, terminated, truncated, info = env_off.step(action)
        # print("off:", info["result"]["throughput"])
        # old_obs, rew, terminated, truncated, info = env_old.step(action)
        # print(env_old.num_task_finished)
        # new_obs, rew, terminated, truncated, info = env_new.step(action)
        # print(env_new.num_task_finished)
        # print(old_obs - off_obs)
        # raise NotImplementedError
        done = terminated or truncated
        # obs = off_obs
        
        # for j in range(5):
        # vis_arr(obs[4], name=f"usage_off_r{cfg.left_right_ratio}_{i}")
        # vis_arr(off_obs[9], name=f"guidance_off_r{cfg.left_right_ratio}_{i}")
        
    print(info["result"]["throughput"])
    

def base_offline_exp(cfg: CompetitionConfig, log_dir):
    cfg.h_update_late = False
    cfg.past_traffic_interval = 20
    cfg.simulation_time = 1000
    # cfg.task_dist_change_interval = 200
    cfg.update_interval = 1000
    cfg.warmup_time = 100
    
    optimal_weights = get_offline_weights(log_dir)
    n_e, n_v = parse_map(cfg.map_path)
    
    from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
    
    tp_list = []
    for seed in range(5):
        env = CompetitionOnlineEnvNew(n_v, n_e, cfg, seed=seed)
        obs, info = env.reset()
        done = False
        while not done:
            obs, rew, terminated, truncated, info = env.step(optimal_weights)
            done = terminated or truncated
        tp = info["result"]["throughput"]
        print(tp)
        tp_list.append(tp)
        
    print(np.mean(tp_list), np.std(tp_list))


def base_exp(cfg: CompetitionConfig, model_params, log_dir): 
    n_e, n_v = parse_map(cfg.map_path)
    eval_logdir = os.path.join(log_dir, "eval") # no use here
    module = CompetitionModule(cfg)
    tp_list = []   
    for seed in range(5):
        # res, _ = module.evaluate_iterative_update(model_params, eval_logdir, n_e, n_v, seed)
        res, _ = module.evaluate_online_update(model_params, eval_logdir, n_e, n_v, seed, env_type="new")
        tp = res['throughput']
        print(f"seed = {seed}, tp = {tp}")
        tp_list.append(tp)
    print(np.mean(tp_list), np.std(tp_list))
    
    
def transfer_exp(cfg: CompetitionConfig, model_params, log_dir):
    map_shapes = ["33x36", "45x47", "57x58", "69x69", "81x80", "93x91"]
    map_shapes = ["33x36"]
    num_agents_lists = [
        [200, 300, 400, 500, 600], 
        [450, 600, 750, 900, 1050], 
        [800, 1000, 1200, 1400, 1600], 
        [1000, 1300, 1600, 1900, 2200], 
        [1000, 1500, 2000, 2500, 3000], 
        [2000, 2500, 3000, 3500, 4000]
    ]
    eval_logdir = os.path.join(log_dir, "eval") # no use here
    os.makedirs(eval_logdir, exist_ok=True)
    
    time_str = get_current_time_str()
    
    logger = logging.getLogger("ggo")
    file_handler = logging.FileHandler(os.path.join(eval_logdir, f"ggo_{time_str}.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("start new experiments!")
    
    for i, map_shape in enumerate(map_shapes):
        map_path = f"maps/competition/expert_baseline/pibt_warehouse_{map_shape}_w_mode_flow_baseline.json" 
        cfg.map_path = map_path
        n_e, n_v = parse_map(map_path)
        for ag in num_agents_lists[i]:
            cfg.num_agents = ag
            module = CompetitionModule(cfg)
        
            tp_list = []
            for seed in range(5):
                res, _ = module.evaluate_online_update(model_params, eval_logdir, n_e, n_v, seed)
                tp = res['throughput']
                info = f"map={map_path}, ag={ag}, seed = {seed}, tp = {tp}"
                print(info)
                logger.info(info)
                tp_list.append(tp)
            logger.critical(f"map={map_path}, ag={ag}, tp_mean={np.mean(tp_list)}, tp_std={np.std(tp_list)}")
        

def main(log_dir):
    cfg, is_online = parse_config(log_dir)
    print("is_online:", is_online)
    if is_online:
        model_params = get_update_model(log_dir)
        base_exp(cfg, model_params, log_dir)
    else:
        base_offline_exp(cfg, log_dir)
        
    # debug(cfg, model_params, log_dir)
    # transfer_exp(cfg, model_params, log_dir)
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir)
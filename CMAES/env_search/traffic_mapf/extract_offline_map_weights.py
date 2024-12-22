import os
import json
import numpy as np
import gin
import argparse

from env_search.iterative_update.envs.trafficflow_env import TrafficFlowOfflineEnv
from env_search.traffic_mapf.module import TrafficMAPFModule
from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.update_model.update_model import CNNUpdateModel
from env_search.utils import min_max_normalize

def get_update_model(log_dir):
    update_model_file = os.path.join(log_dir, "optimal_update_model.json")
    with open(update_model_file, "r") as f:
        model_json = json.load(f)
        
    params = model_json["params"]
    model_params = np.array(params)
    return model_params

def main(log_dir):
    model_params = get_update_model(log_dir)
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    config = TrafficMAPFConfig()
    config.n_sim = 5
    
    env = TrafficFlowOfflineEnv(
        cfg=config, seed=0
    )
    
    update_mdl_kwargs = {}
    if config.iter_update_mdl_kwargs is not None:
        update_mdl_kwargs = config.iter_update_mdl_kwargs
        
    update_model = CNNUpdateModel(
        model_params, 
        **update_mdl_kwargs,
    )
    obs, info = env.reset()
    done = False
    while not done:
        action = update_model.get_update_values_from_obs(obs)
        obs, imp_throughput, done, _, info = env.step(action)
        curr_result = info["result"]
    print("final tp:", curr_result["throughput"])
    map_weights = min_max_normalize(env.raw_weights, 0.1, 100)
    save_path = os.path.join(log_dir, "optimal_weights.json")
    with open(save_path, "w") as f:
        json.dump({
            "weights": map_weights.flatten().tolist()
        }, f, indent=4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", type=str)
    cfg = p.parse_args()
    main(cfg.log_dir)

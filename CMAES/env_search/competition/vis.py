import argparse
import json
import os
import shutil
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule
from env_search.competition.competition_manager import CompetitionManager
from env_search.utils.logging import get_current_time_str
from env_search.utils.plot_utils import parse_save_file

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import numpy as np
import gin
import logging

logging.basicConfig(level=logging.INFO,
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


def vis_arr(arr_, mask=None, name="test", save_dir=None, v_max=None, has_grid=False):
    arr = arr_.copy()
    if save_dir is None:
        save_dir = "env_search/competition/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(arr.shape[1]//5, arr.shape[0]//5))  # Set size to 6x6 inches
    
    if has_grid:
        sns.heatmap(
            arr_,
            square=True,
            cmap="Reds",
            ax=ax,
            cbar=True,
            rasterized=False,
            annot_kws={"size": 30},
            linewidths=1,
            linecolor="black",
            xticklabels=False,
            yticklabels=False,
            vmin=0, 
            vmax=v_max
        )
    else:
        if mask is not None:
            arr = np.ma.masked_where(mask, arr)
        cmap = plt.cm.Reds
        # cmap.set_bad(color='black')
        if v_max is not None:
            im = ax.imshow(arr, cmap=cmap, interpolation="none", vmin=0, vmax=v_max)
        else:
            im = ax.imshow(arr, cmap=cmap, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
        # cbar = fig.colorbar(im, ax=ax)
        # cbar.ax.tick_params(labelsize=20)
    # ax.figure.tight_layout()
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
    # print("save_path =", os.path.join(save_dir, f"{name}.png"))
    plt.close()
    
def vis_ratio(arr_ratio_, mask=None, name="test", save_dir=None, max_v = 5):
    arr = np.clip(arr_ratio_, 1/max_v, max_v)

    if save_dir is None:
        save_dir = "env_search/competition/plots"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(arr.shape[1] // 5, arr.shape[0] // 5))

    # Create color array
    colors = np.ones((arr.shape[0], arr.shape[1], 3))  # Initialize to white
    red_mask = (arr > 1) & ~mask
    blue_mask = (arr <= 1) & ~mask

    colors[red_mask, 1] = 1 - (arr[red_mask] - 1) / max_v
    colors[red_mask, 2] = 1 - (arr[red_mask] - 1) / max_v
    colors[blue_mask, 0] = 1 - ((1 / arr[blue_mask]) - 1) / max_v
    colors[blue_mask, 1] = 1 - ((1 / arr[blue_mask]) - 1) / max_v

    ax.imshow(colors)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
    plt.close()


def vis_direction(gg_obs, name="test", save_dir=None):
    gg_ = gg_obs.copy()
    map_mask = gg_[-1] == 0  # wait action is zero <=> map
    gg_[gg_ == 0] = 200  # change all invalid actions to 200
    gg_direction = np.argmin(gg_, axis=0)

    # Create coordinate grids
    y, x = np.indices(map_mask.shape)
    x, y = x + 0.5, y + 0.5

    # Initialize arrays for directions and relative positions
    directions = np.array([
        [1, 0],  # Right
        [0, 1],  # Up
        [-1, 0], # Left
        [0, -1], # Down
        [0, 0]   # Wait
    ])

    offsets = np.array([
        [0, 0.5],  # Right
        [0.5, 0],  # Up
        [1, 0.5],  # Left
        [0.5, 1],  # Down
        [0.5, 0.5] # Wait
    ]) - 0.5

    # Create arrays for quiver plot
    ax = np.zeros_like(map_mask, dtype=float)
    ay = np.zeros_like(map_mask, dtype=float)
    x_adjusted = np.zeros_like(map_mask, dtype=float)
    y_adjusted = np.zeros_like(map_mask, dtype=float)
    
    for i in range(directions.shape[0]):
        mask = (gg_direction == i) & ~map_mask
        ax[mask], ay[mask] = directions[i]
        x_adjusted[mask], y_adjusted[mask] = x[mask] + offsets[i, 0], y[mask] + offsets[i, 1]

    # Separate black cells and direction cells
    bx_list = x[map_mask]
    by_list = y[map_mask]
    quiver_mask = ~map_mask
    x_list = x_adjusted[quiver_mask]
    y_list = y_adjusted[quiver_mask]
    ax_list = ax[quiver_mask]
    ay_list = ay[quiver_mask]

    if save_dir is None:
        save_dir = "env_search/competition/plots"
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(map_mask.shape[1] // 5, map_mask.shape[0] // 5))  # Set size to 6x6 inches
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.quiver(x_list, y_list, ax_list, ay_list, color="r")
    plt.scatter(bx_list, by_list, color='black', marker="s")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
    # print(os.path.join(save_dir, f"{name}.png"))
    plt.close()


def generate_left_right_ratio_arr_and_mask(gg_obs):
    right_gg, left_gg = gg_obs[0], gg_obs[2]
    left_right_ratio = right_gg/(left_gg + 0.01)
    mask = np.logical_or((gg_obs[0]==0), (gg_obs[2]==0))
    return left_right_ratio, mask

def base_online_exp(cfg: CompetitionConfig, model_params, log_dir, seed=0, vis=False, save_files=False, vis_only_final=False):
    n_e, n_v = parse_map(cfg.map_path)
    
    
    cfg.simulation_time = 1000
    if save_files:
        cfg.simulation_time = 10000
    if vis_only_final:
        cfg.simulation_time = 5000
    vis_past_interval = 20
    
    vis_logdir = os.path.join(log_dir, f"vis{vis_past_interval}")
    os.makedirs(vis_logdir, exist_ok=True)
    
    from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
    
    env = CompetitionOnlineEnvNew(
        n_valid_vertices=n_v,
        n_valid_edges=n_e, 
        config=cfg,
        seed=seed
    )
    comp_map = Map(cfg.map_path)
    update_mdl_kwargs = {}
    if cfg.iter_update_mdl_kwargs is not None:
        update_mdl_kwargs = cfg.iter_update_mdl_kwargs
    update_model = cfg.iter_update_model_type(
            comp_map, model_params, n_v, n_e,
            **update_mdl_kwargs,
        )

    done =False    
    obs, info = env.reset(options={"save_paths": save_files})
    if vis_only_final:
        cumulative_wait = np.zeros_like(env.comp_map.graph, dtype=int)
    while not done:
        # if vis:
        #     obs = np.concatenate([obs[:5], obs[-1:]], axis=0)
        wait_cost_update_vals, edge_weight_update_vals = update_model.get_update_values_from_obs(obs)
        action = np.concatenate([wait_cost_update_vals,
                                edge_weight_update_vals])
        obs, rew, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        if vis and not vis_only_final:
            wait_usage, edge_usage = env._gen_traffic_obs_new(past_traffic_interval=vis_past_interval, vis=True)
            traffic_obs = np.concatenate([edge_usage, wait_usage], axis=0, dtype=np.float32)
            for i in range(5):
                # v_max = 0.25 if i < 4 else 1
                vis_arr(traffic_obs[i], name=f"{env.i}_traffic{i}", save_dir=vis_logdir, v_max=vis_past_interval)
            gg_obs = env._gen_gg_obs()
            for i in range(5):
                vis_arr(gg_obs[i], name=f"{env.i}_gg{i}", save_dir=vis_logdir, v_max=1)
            left_right_ratio, mask = generate_left_right_ratio_arr_and_mask(gg_obs)
            vis_ratio(left_right_ratio, mask=mask, name=f"{env.i}_ratio", save_dir=vis_logdir)
            vis_direction(gg_obs, name=f"{env.i}_d", save_dir=vis_logdir)
            vis_arr(obs[-1], name=f"{env.i}_task", save_dir=vis_logdir)
            v_usage = env._gen_v_usage(past_traffic_interval=vis_past_interval)
            vis_arr(v_usage, name=f"{env.i}_v", save_dir=vis_logdir, v_max=vis_past_interval)
        elif vis_only_final:
            wait_usage, edge_usage = env._gen_traffic_obs_new(vis=True)
            cumulative_wait += wait_usage[0].astype(int)
    if vis:
        env.config.past_traffic_interval = cfg.simulation_time
        wait_usage, edge_usage = env._gen_traffic_obs_new(vis=True)
        traffic_obs = np.concatenate([edge_usage, wait_usage], axis=0, dtype=np.float32)
        for i in range(5):
            vis_arr(traffic_obs[i], name=f"all_traffic{i}", save_dir=vis_logdir)
        if vis_only_final:
            vis_arr(cumulative_wait, name="cumulative_wait", save_dir=vis_logdir, v_max=cfg.simulation_time)
    
    # print(info["result"]["throughput"])
    if save_files:
        res_file_path = os.path.join(log_dir, f"results_{seed}.json")
        shutil.copy("large_files_new/results.json", res_file_path)
        parse_save_file(file_path=res_file_path,
                        timestep_start=1, timestep_end=cfg.simulation_time+1, 
                        seed=seed, 
                        save_dir=vis_logdir)
    return info["result"]["throughput"]
    

def base_offline_exp(cfg: CompetitionConfig, log_dir, seed=0, vis=False, save_files=False, vis_only_final=False):
    cfg.h_update_late = False
    cfg.past_traffic_interval = 20
    cfg.simulation_time = 1000
    # cfg.task_dist_change_interval = 200
    cfg.update_interval = 20 if vis and not vis_only_final else cfg.simulation_time
    if save_files:
        cfg.simulation_time = 10000
        cfg.update_interval = 10000
    if vis_only_final:
        cfg.simulation_time = 5000
        cfg.update_interval = 5000
    vis_past_interval = 20
    
    cfg.warmup_time = 20
    
    cfg.has_traffic_obs = True
    cfg.has_gg_obs = True
    cfg.has_future_obs = False
    cfg.has_task_obs = True
    
    n_e, n_v = parse_map(cfg.map_path)
    if log_dir == "test":
        optimal_weights = np.ones(n_e+n_v)
        map_name = cfg.map_path.split('/')[-1].split('.')[0]        
        log_dir = os.path.join("logs", log_dir, map_name)
    else:
        optimal_weights = get_offline_weights(log_dir)
        
    vis_logdir = os.path.join(log_dir, f"vis{vis_past_interval}") # no use here
    os.makedirs(vis_logdir, exist_ok=True)
    
    from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
    
    tp_list = []
    env = CompetitionOnlineEnvNew(n_v, n_e, cfg, seed=seed)
    obs, info = env.reset(options={"save_paths": save_files})
    done = False
    while not done:
        # print("action", optimal_weights.min(), optimal_weights.argmin(), optimal_weights.max(), optimal_weights.argmax())
        obs, rew, terminated, truncated, info = env.step(optimal_weights)
        if vis and not vis_only_final:
            # env.config.past_traffic_interval = 1000
            wait_usage, edge_usage = env._gen_traffic_obs_new(past_traffic_interval=vis_past_interval, vis=True)
            # env.config.past_traffic_interval = 20
            traffic_obs = np.concatenate([edge_usage, wait_usage], axis=0, dtype=np.float32)
            for i in range(5):
                vis_arr(traffic_obs[i], name=f"{env.i}_traffic{i}", save_dir=vis_logdir, v_max=vis_past_interval)
            gg_obs = env._gen_gg_obs()
            for i in range(5):
                vis_arr(gg_obs[i], name=f"{env.i}_gg{i}", save_dir=vis_logdir, v_max=1)
            left_right_ratio, mask = generate_left_right_ratio_arr_and_mask(gg_obs)
            if env.i == 1:
                vis_ratio(left_right_ratio, mask=mask, name=f"{env.i}_ratio", save_dir=vis_logdir)
                vis_direction(gg_obs, name=f"{env.i}_d", save_dir=vis_logdir)
            v_usage = env._gen_v_usage(past_traffic_interval=vis_past_interval)
            vis_arr(v_usage, name=f"{env.i}_v", save_dir=vis_logdir, v_max=vis_past_interval)
            vis_arr(obs[-1], name=f"{env.i}_task", save_dir=vis_logdir)

        done = terminated or truncated

    if vis:
        env.config.past_traffic_interval = cfg.simulation_time
        wait_usage, edge_usage = env._gen_traffic_obs_new(vis=True)
        traffic_obs = np.concatenate([edge_usage, wait_usage], axis=0, dtype=np.float32)
        for i in range(5):
            vis_arr(traffic_obs[i], name=f"all_traffic{i}", save_dir=vis_logdir, v_max=cfg.simulation_time)
    
    if save_files:
        shutil.copy("large_files_new/results.json", os.path.join(log_dir, f"results_{seed}.json"))
        parse_save_file(file_path="large_files_new/results.json",
                    timestep_start=1, timestep_end=cfg.simulation_time+1, 
                    seed=seed, 
                    save_dir=vis_logdir)
    tp = info["result"]["throughput"]
    print(tp)
    return tp

def main(log_dir, seed, vis, save_files, vis_only_final, no_ckpt):
    cfg, is_online = parse_config(log_dir)
    print("is_online:", is_online)
    if is_online:
        model_params = get_update_model(log_dir)
        tp = base_online_exp(cfg, model_params, log_dir, seed, vis, save_files, vis_only_final)
    else:
        if no_ckpt:
            log_dir = "test"
        tp = base_offline_exp(cfg, log_dir, seed, vis, save_files, vis_only_final)
    return tp  

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--vis', action="store_true", default=False)
    p.add_argument('--vis_only_final', action="store_true", default=False)
    p.add_argument('--save_files', action="store_true", default=False)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_ckpt', action="store_true", default=False)
    cfg = p.parse_args()
    
    tp = main(log_dir=cfg.logdir, seed=cfg.seed, vis=cfg.vis, save_files=cfg.save_files, vis_only_final=cfg.vis_only_final, no_ckpt=cfg.no_ckpt)
    
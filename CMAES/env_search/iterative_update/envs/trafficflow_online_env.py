from env_search.competition.config import CompetitionConfig
from env_search.competition.update_model.utils import Map, comp_uncompress_vertex_matrix, comp_uncompress_edge_matrix
from env_search.utils import min_max_normalize, load_pibt_default_config, load_w_pibt_default_config, load_wppl_default_config, get_project_dir
from env_search.utils.logging import get_current_time_str, get_hash_file_name
from env_search.utils.task_generator import generate_task_and_agent
from env_search.iterative_update.envs.utils import visualize_simulation
import numpy as np
import os
from gymnasium import spaces
import json
import time
import subprocess
import gc

from simulators.trafficMAPF_on.period_on_sim import period_on_sim
from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.iterative_update.envs.env import REDUNDANT_COMPETITION_KEYS
import shutil


DIRECTION2ID = {
    "R":0, "D":3, "L":2, "U":1, "W":4
}

class TrafficFlowOnlineEnv:
    '''Env for [p-on]+GPIBT'''
    def __init__(
        self,
        config: TrafficMAPFConfig,
        seed: int
    ):
        
        self.config = config
        assert(self.config.update_gg_interval > 0)
        
        self.comp_map = Map(self.config.map_path)
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        # Use CNN observation
        h, w = self.comp_map.height, self.comp_map.width
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))
        self.lb, self.ub = (0.1, 100)
    
    def update_paths(self, agents_paths):
        for agent_moves, agent_new_paths in zip(self.move_hists, agents_paths):
            for s in agent_new_paths:
                if s == ",":
                    continue
                agent_moves.append(s)
        
        for i, agent_pos in enumerate(self.pos_hists):
            if len(agent_pos) == 0:
                agent_pos.append(self.starts[i])
            
            last_h, last_w = agent_pos[-1]
            
            for s in agents_paths[i]:
                if s == ",":
                    continue
                elif s == "R":
                    cur_pos = [last_h, last_w+1]
                elif s == "D":
                    cur_pos = [last_h+1, last_w]
                elif s == "L":
                    cur_pos = [last_h, last_w-1]
                elif s == "U":
                    cur_pos = [last_h-1, last_w]
                elif s == "W":
                    cur_pos = [last_h, last_w]
                else:
                    print(f"s = {s}")
                    raise NotImplementedError
                assert (cur_pos[0]>=0 and cur_pos[0]<self.comp_map.height \
                    and cur_pos[1]>=0 and cur_pos[1]<self.comp_map.width)
                agent_pos.append(cur_pos)
                last_h, last_w = agent_pos[-1]
        
    def _gen_future_obs(self, results):
        "exec_future, plan_future, exec_move, plan_move"
        # 5 dim
        h, w = self.comp_map.graph.shape
        exec_future_usage = np.zeros((5, h, w))
        for aid, (agent_path, agent_m) in enumerate(zip(results["exec_future"], results["exec_move"])):
            if aid in results["agents_finish_task"]:
                continue
            goal_id = results["final_tasks"][aid]
            for (x, y), m in zip(agent_path[1:], agent_m[1:]):
                if x*w+y == goal_id:
                    break
                d_id = DIRECTION2ID[m]
                exec_future_usage[d_id, x, y] += 1
        
        plan_future_usage = np.zeros((5, h, w))
        for aid, (agent_path, agent_m) in enumerate(zip(results["plan_future"], results["plan_move"])):
            if aid in results["agents_finish_task"]:
                continue
            goal_id = results["final_tasks"][aid]
            for (x, y), m in zip(agent_path, agent_m):
                if x*w+y == goal_id:
                    break
                d_id = DIRECTION2ID[m]
                plan_future_usage[d_id, x, y] += 1
                
        if exec_future_usage.sum()!=0:
            exec_future_usage = exec_future_usage/exec_future_usage.sum()
        if plan_future_usage.sum()!=0:
            plan_future_usage = plan_future_usage/plan_future_usage.sum()     
        
        return exec_future_usage, plan_future_usage
        
        
    def _gen_traffic_obs(self, is_init=False):
        h, w = self.comp_map.graph.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        
        if not is_init:
            time_range = min(self.config.past_traffic_interval, self.config.simu_time-self.left_timesteps)
        else:
            time_range = min(self.config.past_traffic_interval, self.config.warmup_time)
        
        # print("time range =", time_range)
        for t in range(time_range):
            for agent_i in range(self.config.num_agents):
                prev_x, prev_y = self.pos_hists[agent_i][-(time_range+1-t)]
                
                move = self.move_hists[agent_i][-(time_range-t)]
                id = DIRECTION2ID[move]
                if id < 4:
                    edge_usage[id, prev_x, prev_y] += 1
                else:
                    wait_usage[0, prev_x, prev_y] += 1
        
        # print("max", wait_usage.max(), wait_usage.argmax(), edge_usage.max(), edge_usage.argmax())
        if wait_usage.sum() != 0:
            wait_usage = wait_usage/wait_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage/edge_usage.sum() * 100
        return wait_usage, edge_usage

    def _gen_task_obs(self, result):
        h, w = self.comp_map.graph.shape
        task_usage = np.zeros((1, h, w))
        for aid, (x, y) in enumerate(result["curr_tasks"]):
            task_usage[0, x, y] += 1
        if task_usage.sum()!=0:
            task_usage = task_usage/task_usage.sum() * 10
        return task_usage
        
    def _gen_curr_pos_obs(self, result):
        h, w = self.comp_map.graph.shape
        pos_usage = np.zeros((2, h, w))
        for aid, (curr_id, goal_id) in enumerate(zip(result["curr_pos"], result["curr_tasks"])):
            x = curr_id // w
            y = curr_id % w
            
            gx = goal_id // w
            gy = goal_id % w
            
            pos_usage[0, x, y] = (gx-x)/ h
            pos_usage[1, x, y] = (gy-y)/w
        return pos_usage
        
    
    def _gen_obs(self, result, is_init=False):
        h, w = self.comp_map.height, self.comp_map.width
        obs = np.zeros((0, h, w))
        
        # past edge usage, including self-edges
        if self.config.has_traffic_obs:
            wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs(is_init)
            traffic_obs = np.concatenate([edge_usage_matrix, wait_usage_matrix], axis=0, dtype=np.float32)
            obs = np.concatenate([obs, traffic_obs], axis=0, dtype=np.float32)
            
        # guidance graph
        if self.config.has_gg_obs:
            wait_costs = min_max_normalize(self.curr_wait_costs, 0.1, 1)
            edge_weights = min_max_normalize(self.curr_edge_weights, 0.1, 1)
            wait_cost_matrix = np.array(
                comp_uncompress_vertex_matrix(self.comp_map, wait_costs))
            edge_weight_matrix = np.array(
                comp_uncompress_edge_matrix(self.comp_map, edge_weights))
            edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
            wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
            
            edge_weight_matrix = np.moveaxis(edge_weight_matrix, 2, 0)
            wait_cost_matrix = np.moveaxis(wait_cost_matrix, 2, 0)

            gg_obs = np.concatenate([edge_weight_matrix, wait_cost_matrix], axis=0, dtype=np.float32)
            obs = np.concatenate([obs, gg_obs], axis=0, dtype=np.float32)
            
        # if self.config.has_future_obs:
        #     exec_future_usage, plan_future_usage = self._gen_future_obs(result)
        #     obs = np.concatenate([obs, exec_future_usage+plan_future_usage], axis=0, dtype=np.float32)
        
        # current task loc
        if self.config.has_task_obs:
            task_obs = self._gen_task_obs(result)
            obs = np.concatenate([obs, task_obs], axis=0, dtype=np.float32)
        
        # if self.config.has_curr_pos_obs:
        #     curr_pos_obs = self._gen_curr_pos_obs(result)
        #     obs = np.concatenate([obs, curr_pos_obs], axis=0, dtype=np.float32)
        
        # current map obs
        if self.config.has_map_obs:
            obs = np.concatenate([obs, self.comp_map.graph.reshape(1, h, w)], dtype=np.float32)
        return obs

    
    def gen_sim_kwargs(self):
        kwargs = {
            "seed": int(self.rng.integers(100000)),
            "simu_time": self.config.simu_time, 
            "map_path": self.config.map_path, 
            "gen_tasks": self.config.gen_tasks,  
            "num_agents": self.config.num_agents, 
            "num_tasks": self.config.num_tasks, 
            "hidden_size": self.config.hidden_size, 
            "task_assignment_strategy": self.config.task_assignment_strategy, 
            "task_dist_change_interval": self.config.task_dist_change_interval, 
            "task_random_type": self.config.task_random_type, 
            "dist_sigma": self.config.dist_sigma, 
            "dist_K": self.config.dist_K, 
            "warmup_time": self.config.warmup_time, 
            "update_gg_interval": self.config.update_gg_interval
        }
        if not self.config.gen_tasks:
            assert self.config.all_json_path is not None
            kwargs["gen_tasks"] = False
            kwargs["all_json_path"] = self.config.all_json_path
        return kwargs
        
    def _run_sim(self, init_weight=False):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """        
        # Initial weights are assumed to be valid
        if init_weight:
            map_weights = self.curr_weights.flatten().tolist()
        else:
            map_weights = min_max_normalize(self.curr_weights, self.lb, self.ub).flatten().tolist()
        result_str = self.simulator.update_gg_and_step(map_weights)
        result = json.loads(result_str)

        self.left_timesteps -= self.config.update_gg_interval
        self.left_timesteps = max(0, self.left_timesteps)
        return result

    
    def step(self, action):
        '''
        action: [4, h, w]
        '''
        self.i += 1  # increment timestep
        
        _action = action * (1-self.comp_map.graph)
        self.curr_weights = np.moveaxis(_action, 0, 2)

        result = self._run_sim()
        
        assert self.starts is not None
        self.update_paths(result["actual_paths"])

        new_task_finished = result["num_task_finished"]
        reward = new_task_finished - self.num_task_finished
        self.num_task_finished = new_task_finished
        
        # terminated/truncate if no left time steps
        terminated = result["done"]
        truncated = terminated
        
        result["throughput"] = self.num_task_finished / self.config.simu_time

        # Info includes the results
        sub_result = {"throughput": result["throughput"]}
        info = {
            "result": sub_result,
            "weights": self.curr_weights
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    
    def reset(self, seed=None, options=None):
        self.i = 0
        self.num_task_finished = 0
        self.left_timesteps = self.config.simu_time
        
        self.starts = None
        
        self.pos_hists = [[] for _ in range(self.config.num_agents)] # record vertex history
        self.move_hists = [[] for _ in range(self.config.num_agents)] # record edge history

        # By default, we use uniform weights for warmup phase
        if self.config.reset_weights_path is None:
            self.curr_weights = np.ones((self.comp_map.height, self.comp_map.width, 4))
        else:
            with open(self.config.reset_weights_path, "r") as f:
                weights_json = json.load(f)
            weights = weights_json["weights"]
            self.curr_weights = np.array(weights)
            
        kwargs = self.gen_sim_kwargs()
        kwargs["map_weights"] = json.dumps(self.curr_weights.flatten().tolist())
        
        # initialize cpp simulator
        self.simulator = period_on_sim(**kwargs)
        result_str = self.simulator.warmup()
        result = json.loads(result_str)
        
        self.starts = result["start"] # agent start locations
        self.update_paths(result["actual_paths"])

        obs = self._gen_obs(result, is_init=True)
        info = {"result": {}}
        return obs, info
    
if __name__ == "__main__":
    import gin
    from env_search.utils import get_n_valid_edges, get_n_valid_vertices
    from env_search.competition.update_model.utils import Map
    cfg_file_path = "config/traffic_mapf/period_online/empty.gin"
    gin.parse_config_file(cfg_file_path, skip_unknown=True)
    cfg = TrafficMAPFConfig()
    cfg.num_agents = 2
    # cfg.warmup_time = 10
    # cfg.simu_time = 1000
    # cfg.update_gg_interval = 20
    # cfg.past_traffic_interval = 20
    # cfg.task_dist_change_interval = 200
    # cfg.has_traffic_obs = True
    # cfg.has_gg_obs = False
    # cfg.has_task_obs = False
    # cfg.task_assignment_strategy = "online_generate"
    # cfg.task_random_type = "Gaussian"
    # cfg.dist_sigma = 1.0
    
    
    env = TrafficFlowOnlineEnv(cfg, seed=0)
    
    def vis_arr(arr_, mask=None, name="test"):
        arr = arr_.copy()
        save_dir = "env_search/traffic_mapf/plots"
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
        
    np.set_printoptions(threshold=np.inf)
    obs, info = env.reset()
    print("after reset!")
    # for i in range(5):
    #     vis_arr(obs[i], name=f"step{env.i}_traffic{i}")
    done = False
    while not done:
        print(obs.shape)
        action = np.ones((4, 32, 32))
        obs, reward, terminated, truncated, info = env.step(action)
        # for i in range(5):
        #     vis_arr(obs[i], name=f"step{env.i}_traffic{i}")
        done = terminated or truncated
    
    print(info["result"]["throughput"])
            
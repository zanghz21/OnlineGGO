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
from simulators.wppl.py_sim import py_sim
from env_search.iterative_update.envs.env import REDUNDANT_COMPETITION_KEYS
import shutil


DIRECTION2ID = {
    "R":0, "D":3, "L":2, "U":1, "W":4
}

class CompetitionOnlineEnvNew:
    def __init__(
        self,
        n_valid_vertices,
        n_valid_edges,
        config: CompetitionConfig,
        seed: int
    ):
        self.n_valid_vertices=n_valid_vertices
        self.n_valid_edges=n_valid_edges
        
        self.config = config
        assert(self.config.update_interval > 0)
        
        self.comp_map = Map(self.config.map_path)
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        # Use CNN observation
        h, w = self.comp_map.height, self.comp_map.width
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None

    def generate_video(self):
        visualize_simulation(self.comp_map, self.pos_hists, "large_files_new/results.json")
        
    def update_paths_with_full_past(self, pos_hists, agents_paths):
        self.pos_hists = pos_hists
        self.move_hists = []
        for agent_path in agents_paths:
            self.move_hists.append(agent_path.replace(",", ""))
    
    def update_paths(self, agents_paths):
        '''
        update self.move_hists and self.pos_hists
        - agents_paths: List[str]
            - e.g. ["R,W,W,U,D", "L,D,D,W,R]
            - R: right, D: down, L: left, U: up, W: wait
        '''
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
        
        
    def _gen_traffic_obs_new(self, past_traffic_interval=None, is_init=False, vis=False):
        h, w = self.comp_map.graph.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        
        if past_traffic_interval is None:
            past_traffic_interval = self.config.past_traffic_interval
        if not is_init:
            time_range = min(past_traffic_interval, self.config.simulation_time-self.left_timesteps)
        else:
            time_range = min(past_traffic_interval, self.config.warmup_time)
        
        for t in range(time_range):
            for agent_i in range(self.config.num_agents):
                prev_x, prev_y = self.pos_hists[agent_i][-(time_range+1-t)]
                
                move = self.move_hists[agent_i][-(time_range-t)]
                id = DIRECTION2ID[move]
                if id < 4:
                    edge_usage[id, prev_x, prev_y] += 1
                else:
                    wait_usage[0, prev_x, prev_y] += 1
        
        # If not for visualization, we normalize the raw observation by the sum
        if not vis:
            if wait_usage.sum() != 0:
                wait_usage = wait_usage/wait_usage.sum() * 100
            if edge_usage.sum() != 0:
                edge_usage = edge_usage/edge_usage.sum() * 100
        return wait_usage, edge_usage

    def _gen_v_usage(self, past_traffic_interval=None, is_init=False):
        h, w = self.comp_map.graph.shape
        v_usage = np.zeros((h, w))
        
        if past_traffic_interval is None:
            past_traffic_interval = self.config.past_traffic_interval
        if not is_init:
            time_range = min(past_traffic_interval, self.config.simulation_time-self.left_timesteps)
        else:
            time_range = min(past_traffic_interval, self.config.warmup_time)
        
        for t in range(time_range):
            for agent_i in range(self.config.num_agents):
                curr_x, curr_y = self.pos_hists[agent_i][-(time_range-t)]
                v_usage[curr_x, curr_y] += 1
        return v_usage
        
    def _gen_task_obs(self, result):
        h, w = self.comp_map.graph.shape
        task_usage = np.zeros((1, h, w))
        for aid, goal_id in enumerate(result["final_tasks"]):
            x = goal_id // w
            y = goal_id % w
            task_usage[0, x, y] += 1
        
        # normalized by sum
        if task_usage.sum()!=0:
            task_usage = task_usage/task_usage.sum() * 10
        return task_usage
        
    def _gen_curr_pos_obs(self, result):
        '''
        - return relative postion between current pos and goal pos for all agent
        - note that at each timestep, each loc on the map has at most one agent
        '''
        h, w = self.comp_map.graph.shape
        pos_usage = np.zeros((2, h, w))
        for aid, (curr_id, goal_id) in enumerate(zip(result["final_pos"], result["final_tasks"])):
            x = curr_id // w
            y = curr_id % w
            
            gx = goal_id // w
            gy = goal_id % w
            
            pos_usage[0, x, y] = (gx-x)/ h
            pos_usage[1, x, y] = (gy-y)/w
        return pos_usage
        
    def _gen_gg_obs(self):
        h, w = self.comp_map.height, self.comp_map.width
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
        return gg_obs
    
    def _gen_obs(self, result, is_init=False):
        h, w = self.comp_map.height, self.comp_map.width
        obs = np.zeros((0, h, w))
        
        # graph edge usage (include self-edges)
        if self.config.has_traffic_obs:
            wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs_new(is_init)
            traffic_obs = np.concatenate([edge_usage_matrix, wait_usage_matrix], axis=0, dtype=np.float32)
            obs = np.concatenate([obs, traffic_obs], axis=0, dtype=np.float32)
            
        # current guidance graph weights
        if self.config.has_gg_obs:
            gg_obs = self._gen_gg_obs()
            obs = np.concatenate([obs, gg_obs], axis=0, dtype=np.float32)

        # graph edge usage in future steps (Not support all algos yet)
        if self.config.has_future_obs:
            exec_future_usage, plan_future_usage = self._gen_future_obs(result)
            obs = np.concatenate([obs, exec_future_usage+plan_future_usage], axis=0, dtype=np.float32)
            
        # current task
        if self.config.has_task_obs:
            task_obs = self._gen_task_obs(result)
            obs = np.concatenate([obs, task_obs], axis=0, dtype=np.float32)
            
        # current agents' location
        if self.config.has_curr_pos_obs:
            curr_pos_obs = self._gen_curr_pos_obs(result)
            obs = np.concatenate([obs, curr_pos_obs], axis=0, dtype=np.float32)
            
        # map
        if self.config.has_map_obs:
            obs = np.concatenate([obs, self.comp_map.graph.reshape(1, h, w)], axis=0, dtype=np.float32)
        return obs

            
    def get_base_kwargs(self, save_paths=False):
        '''generate base kwargs for cpp simulator'''
        
        kwargs = {
            "left_w_weight": self.config.left_right_ratio, 
            "right_w_weight": 1.0, 
            "map_json_path": self.config.map_path,
            "simulation_steps": self.config.simulation_time,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "plan_time_limit": self.config.plan_time_limit,
            "seed": int(self.rng.integers(100000)),
            "task_dist_change_interval": self.config.task_dist_change_interval, 
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": self.config.file_storage_path + "_new",
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal, 
            "warmup_steps": self.config.warmup_time, 
            "update_gg_interval": self.config.update_interval, 
            "h_update_late": self.config.h_update_late, 
            "dist_sigma": self.config.dist_sigma, 
            "dist_K": self.config.dist_K, 
            "save_paths": save_paths
        }
        
        # By default, self.config.gen_random = True, all tasks are generated in cpp simulator.
        # If set False, task is generated here.
        if not self.config.gen_random:
            file_dir = os.path.join(get_project_dir(), 'run_files', 'gen_task')
            os.makedirs(file_dir, exist_ok=True)
            sub_dir_name = get_hash_file_name()
            self.task_save_dir = os.path.join(file_dir, sub_dir_name)
            os.makedirs(self.task_save_dir, exist_ok=True)
            
            generate_task_and_agent(self.config.map_base_path, 
                total_task_num=100000, num_agents=self.config.num_agents, 
                save_dir=self.task_save_dir
            )
            
            kwargs["agents_path"] = os.path.join(self.task_save_dir, "test.agent")
            kwargs["tasks_path"] = os.path.join(self.task_save_dir, "test.task")
        
        # If task_dist_change_interval < 0, it is uniform task distribution by default
        if self.config.task_dist_change_interval > 0:
            kwargs["task_random_type"] = self.config.task_random_type
        
        if self.config.base_algo == "pibt":
            if self.config.has_future_obs:
                # use pibt to generate paths for [n_future] steps
                kwargs["config"] = load_w_pibt_default_config()
            else:
                kwargs["config"] = load_pibt_default_config()
        elif self.config.base_algo == "wppl":
            kwargs["config"] = load_wppl_default_config()
        else:
            print(f"base algo [{self.config.base_algo}] is not supported")
            raise NotImplementedError
        return kwargs
    
        
    def _run_sim(self,
                 init_weight=False):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """        
        # Initial weights are assumed to be valid
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                           self.ub).tolist()

        result_str = self.simulator.update_gg_and_step(edge_weights, wait_costs)
        result = json.loads(result_str)

        self.left_timesteps -= self.config.update_interval
        self.left_timesteps = max(0, self.left_timesteps)
        return result

    def update_lr_weights(self, ratio):
        # parse map left right location
        assert len(self.comp_map.home_loc_ids) > 0
        assert len(self.comp_map.end_points_ids) > 0
        
        home_prob = []
        for home_loc in self.comp_map.home_loc_ids:
            if home_loc[1] == 0:
                home_prob.append(ratio)
            else:
                home_prob.append(1.0)
        
        end_prob = [1] * len(self.comp_map.end_points_ids)
        total_prob = home_prob + end_prob
        self.simulator.update_tasks_base_distribution(total_prob)
        
    
    def step(self, action):
        '''action: List[float], (self-edge weights) + (other edge weights)'''
        self.i += 1  # increment timestep
        
        wait_cost_update_vals = action[:self.n_valid_vertices]
        edge_weight_update_vals = action[self.n_valid_vertices:]
        self.curr_wait_costs = wait_cost_update_vals # raw weights w/o normalization, possibly < 0
        self.curr_edge_weights = edge_weight_update_vals

        result = self._run_sim()
        
        # self.last_agent_pos = result["final_pos"]
        # self.last_tasks = result["final_tasks"]
        assert self.starts is not None
        self.update_paths(result["actual_paths"])

        new_task_finished = result["num_task_finished"]
        reward = new_task_finished - self.num_task_finished
        self.num_task_finished = new_task_finished
        
        # terminated/truncate if no left time steps
        terminated = result["done"]
        truncated = terminated
        if terminated or truncated:
            if not self.config.gen_random:
                # rm files of generated tasks
                if os.path.exists(self.task_save_dir):
                    shutil.rmtree(self.task_save_dir)
                else:
                    raise NotImplementedError
        
        result["throughput"] = self.num_task_finished / self.config.simulation_time

        # Info includes the results
        sub_result = {k: v for k, v in result.items() if k not in REDUNDANT_COMPETITION_KEYS}
        info = {
            "result": sub_result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        # t0 = time.time()
        obs = self._gen_obs(result)
        # t1 = time.time()
        # print("gen obs time =", t1-t0)
        return obs, reward, terminated, truncated, info

    
    def reset(self, seed=None, options=None):
        self.i = 0
        self.num_task_finished = 0
        self.left_timesteps = self.config.simulation_time
        self.last_agent_pos = None
        
        self.starts = None
        self.task_save_dir = None
        
        self.pos_hists = [[] for _ in range(self.config.num_agents)] # record vertex history
        self.move_hists = [[] for _ in range(self.config.num_agents)] # record edge history
        
        self.last_wait_usage = np.zeros(np.prod(self.comp_map.graph.shape))
        self.last_edge_usage = np.zeros(4*np.prod(self.comp_map.graph.shape))
        
        # If no particular specification, we use uniform weights in warmup phase
        if self.config.reset_weights_path is None:
            self.curr_edge_weights = np.ones(self.n_valid_edges)
            self.curr_wait_costs = np.ones(self.n_valid_vertices)
        else:
            with open(self.config.reset_weights_path, "r") as f:
                weights_json = json.load(f)
            weights = weights_json["weights"]
            self.curr_wait_costs = np.array(weights[:self.n_valid_vertices])
            self.curr_edge_weights = np.array(weights[self.n_valid_vertices:])
            
        # save detailed running meta data
        # It is used to generate the num-agents-reach-goals per timestep
        if options is not None:
            save_paths = options.get("save_paths", False)
        else:
            save_paths = False
        
        kwargs = self.get_base_kwargs(save_paths)
        
        # add initial guidance graph weights
        kwargs["weights"] = json.dumps(self.curr_edge_weights.tolist()) # weights except for self-edges
        kwargs["wait_costs"] = json.dumps(self.curr_wait_costs.tolist()) # self-edge weights
        
        # initialize cpp simulator
        self.simulator = py_sim(**kwargs)
        
        # run [n_warmup] steps to generate the initial obs
        result_str = self.simulator.warmup()
        result = json.loads(result_str)
        
        # update agents path info
        self.starts = result["starts"]
        self.update_paths(result["actual_paths"])

        # generate obs
        obs = self._gen_obs(result, is_init=True)
        info = {"result": {}}
        return obs, info
    
if __name__ == "__main__":
    import gin
    from env_search.utils import get_n_valid_edges, get_n_valid_vertices
    from env_search.competition.update_model.utils import Map
    cfg_file_path = "config/competition/test_env.gin"
    gin.parse_config_file(cfg_file_path)
    cfg = CompetitionConfig()
    cfg.has_future_obs = False
    cfg.num_agents = 800
    cfg.warmup_time = 20
    cfg.simulation_time = 1000
    cfg.update_interval = 20
    cfg.past_traffic_interval = 20
    cfg.task_dist_change_interval = -1
    cfg.has_traffic_obs = True
    cfg.has_gg_obs = False
    cfg.has_task_obs = True
    cfg.has_curr_pos_obs = False
    cfg.task_assignment_strategy = "online_generate"
    cfg.task_random_type = "Gaussian"
    cfg.dist_sigma = 1.0
    
    # cfg.gen_random = False
    cfg.map_base_path = "maps/competition/online_map/sortation_small.json"
    # cfg.map_path = "maps/competition/human/pibt_random_unweight_32x32.json"
    # cfg.map_path = "maps/competition/online_map/warehouse_small_narrow.json"
    
    comp_map = Map(cfg.map_path)
    domain = "competition"
    n_valid_vertices = get_n_valid_vertices(comp_map.graph, domain)
    n_valid_edges = get_n_valid_edges(comp_map.graph, bi_directed=True, domain=domain)
    
    env = CompetitionOnlineEnvNew(n_valid_vertices, n_valid_edges, cfg, seed=0)
    
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
        
    np.set_printoptions(threshold=np.inf)
    obs, info = env.reset()
    for i in range(5):
        vis_arr(obs[i], name=f"step{env.i}_traffic{i}")
    vis_arr(obs[-1], name=f"step{env.i}_task{i}")
    
    done = False
    while not done:
        print(obs.shape)
        action = np.ones(n_valid_vertices+n_valid_edges)
        obs, reward, terminated, truncated, info = env.step(action)
        for i in range(5):
            vis_arr(obs[i], name=f"step{env.i}_traffic{i}")
        vis_arr(obs[-1], name=f"step{env.i}_task{i}")
        done = terminated or truncated
    
    print(info["result"]["throughput"])
            
import numpy as np
from gymnasium import spaces
from env_search.warehouse.config import WarehouseConfig
from env_search.utils import (
    kiva_obj_types,
    min_max_normalize,
    kiva_uncompress_edge_weights,
    kiva_uncompress_wait_costs,
    load_pibt_default_config,
    load_wppl_default_config,
    get_project_dir,
    read_in_kiva_map, 
    kiva_env_str2number
)
import copy
import json
import os
from simulators.rhcr.WarehouseSimulator import WarehouseSimulator

class WarehouseOnlineEnv:

    def __init__(
        self,
        map_np,
        map_json,
        num_agents,
        eval_logdir,
        n_valid_vertices,
        n_valid_edges,
        config: WarehouseConfig,
        seed=0,
        init_weight_file=None,
    ):
        self.n_valid_vertices = n_valid_vertices
        self.n_valid_edges = n_valid_edges
        
        self.config = config
        self.map_np = map_np
        self.map_json = map_json
        self.num_agents = num_agents
        self.eval_logdir = eval_logdir
        self.block_idxs = [
            kiva_obj_types.index("@"),
        ]
        self.rng = np.random.default_rng(seed=seed)

        # Use CNN observation
        h, w = self.map_np.shape
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None


    def _gen_traffic_obs_ols(self, result):
        h, w = self.map_np.shape
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])
        # Normalize
        wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        
        traffic_obs = np.concatenate([edge_usage_matrix, wait_usage_matrix], axis=2)
        traffic_obs = np.moveaxis(traffic_obs, 2, 0)
        return traffic_obs
    
    def _gen_traffic_obs(self, is_init):
        h, w = self.map_np.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        
        if not is_init:
            time_range = min(self.config.past_traffic_interval, self.config.simulation_time-self.left_timesteps)
        else:
            time_range = min(self.config.past_traffic_interval, self.config.warmup_time)
        
        for t in range(time_range):
            for agent_i in range(self.num_agents):
                prev_loc = self.pos_hists[agent_i][-(time_range+1-t)]
                curr_loc = self.pos_hists[agent_i][-(time_range-t)]
                
                prev_r, prev_c = prev_loc // w, prev_loc % w
                if prev_loc == curr_loc:
                    wait_usage[0, prev_r, prev_c] += 1
                elif prev_loc + 1 == curr_loc: # R
                    edge_usage[0, prev_r, prev_c] += 1
                elif prev_loc + w == curr_loc: # D
                    edge_usage[1, prev_r, prev_c] += 1
                elif prev_loc - 1 == curr_loc: # L
                    edge_usage[2, prev_r, prev_c] += 1
                elif prev_loc - w == curr_loc: # U
                    edge_usage[3, prev_r, prev_c] += 1
                else:
                    print(prev_loc, curr_loc)
                    print(self.pos_hists[agent_i])
                    raise NotImplementedError
        
        # print("max", wait_usage.max(), wait_usage.argmax(), edge_usage.max(), edge_usage.argmax())
        if wait_usage.sum() != 0:
            wait_usage = wait_usage/wait_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage/edge_usage.sum() * 100
        # print("new, wait_usage:", wait_usage.max(), "edge_usage:", edge_usage.max())
        traffic_obs = np.concatenate([edge_usage, wait_usage], axis=0)
        return traffic_obs
    
    def _gen_future_obs(self, result):
        h, w = self.map_np.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        
        for agent_i in range(self.num_agents):
            for t in range(len(result["future_paths"][agent_i])-1):
                prev_loc = result["future_paths"][agent_i][t]
                curr_loc = result["future_paths"][agent_i][t+1]
                
                prev_r, prev_c = prev_loc // w, prev_loc % w
                if prev_loc == curr_loc:
                    wait_usage[0, prev_r, prev_c] += 1
                elif prev_loc + 1 == curr_loc: # R
                    edge_usage[0, prev_r, prev_c] += 1
                elif prev_loc + w == curr_loc: # D
                    edge_usage[1, prev_r, prev_c] += 1
                elif prev_loc - 1 == curr_loc: # L
                    edge_usage[2, prev_r, prev_c] += 1
                elif prev_loc - w == curr_loc: # U
                    edge_usage[3, prev_r, prev_c] += 1
                else:
                    print(prev_loc, curr_loc)
                    raise NotImplementedError
        
        # print("max", wait_usage.max(), wait_usage.argmax(), edge_usage.max(), edge_usage.argmax())
        if wait_usage.sum() != 0:
            wait_usage = wait_usage/wait_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage/edge_usage.sum() * 100
        # print("new, wait_usage:", wait_usage.max(), "edge_usage:", edge_usage.max())
        future_obs = np.concatenate([wait_usage, edge_usage], axis=0)
        return future_obs
    
    
    def _gen_gg_obs(self):
        edge_weight_matrix = np.array(
            kiva_uncompress_edge_weights(self.map_np,
                                         self.curr_edge_weights,
                                         self.block_idxs,
                                         fill_value=0))
        # While optimizing all wait costs, all entries of `wait_cost_matrix`
        # are different.
        if self.config.optimize_wait:
            wait_cost_matrix = np.array(
                kiva_uncompress_wait_costs(self.map_np,
                                           self.curr_wait_costs,
                                           self.block_idxs,
                                           fill_value=0))
        # Otherwise, `self.curr_wait_costs` is a single number and so all wait
        # costs are the same, but we need to transform it to a matrix.
        else:
            curr_wait_costs_compress = np.zeros(self.n_valid_vertices)
            curr_wait_costs_compress[:] = self.curr_wait_costs
            wait_cost_matrix = np.array(
                kiva_uncompress_wait_costs(self.map_np,
                                           curr_wait_costs_compress,
                                           self.block_idxs,
                                           fill_value=0))
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)
        
        h, w = self.map_np.shape
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
        
        gg_obs = np.concatenate([edge_weight_matrix, wait_cost_matrix], axis=2)
        gg_obs = np.moveaxis(gg_obs, 2, 0)
        return gg_obs
        
    def _gen_task_obs(self, result):
        h, w = self.map_np.shape
        task_usage = np.zeros((1, h, w))
        for aid, goal_id in enumerate(result["goal_locs"]):
            x = goal_id // w
            y = goal_id % w
            task_usage[0, x, y] += 1
        if task_usage.sum()!=0:
            task_usage = task_usage/task_usage.sum() * 10
        return task_usage
        
        
    def _gen_obs(self, result, is_init=False):
        h, w = self.map_np.shape
        obs = np.zeros((0, h, w))
        if self.config.has_traffic_obs:
            traffic_obs = self._gen_traffic_obs(is_init)
            obs = np.concatenate([obs, traffic_obs], axis=0)
        if self.config.has_gg_obs:
            gg_obs = self._gen_gg_obs()
            obs = np.concatenate([obs, gg_obs], axis=0)
        if self.config.has_future_obs:
            future_obs = self._gen_future_obs(result)
            obs = np.concatenate([obs, future_obs], axis=0)
        if self.config.has_task_obs:
            task_obs = self._gen_task_obs(result)
            obs = np.concatenate([obs, task_obs], axis=0)
        return obs
    
    def gen_base_kwargs(self,):
        sim_seed = self.rng.integers(10000)
        kwargs = {
            "seed": int(sim_seed), 
            "output": os.path.join(self.eval_logdir,
                              f"online-seed={sim_seed}"), 
            "scenario": self.config.scenario,
            "task": self.config.task,
            "agentNum": self.num_agents,
            "cutoffTime": self.config.cutoffTime,
            "OverallCutoffTime": self.config.overallCutoffTime, 
            "screen": self.config.screen,
            # "screen": 1, 
            "solver": self.config.solver,
            "id": self.config.id,
            "single_agent_solver": self.config.single_agent_solver,
            "lazyP": self.config.lazyP,
            "simulation_time": self.config.simulation_time,
            "simulation_window": self.config.simulation_window,
            "travel_time_window": self.config.travel_time_window,
            "potential_function": self.config.potential_function,
            "potential_threshold": self.config.potential_threshold,
            "rotation": self.config.rotation,
            "robust": self.config.robust,
            "CAT": self.config.CAT,
            "hold_endpoints": self.config.hold_endpoints,
            "dummy_paths": self.config.dummy_paths,
            "prioritize_start": self.config.prioritize_start,
            "suboptimal_bound": self.config.suboptimal_bound,
            "log": self.config.log,
            "test": self.config.test,
            "force_new_logdir": True,
            "save_result": self.config.save_result,
            "save_solver": self.config.save_solver,
            "save_heuristics_table": self.config.save_heuristics_table,
            "stop_at_traffic_jam": self.config.stop_at_traffic_jam,
            "left_w_weight": self.config.left_w_weight,
            "right_w_weight": self.config.right_w_weight,
            "warmup_time": self.config.warmup_time, 
            "update_gg_interval": self.config.update_gg_interval, 
            "task_dist_update_interval": self.config.task_dist_update_interval, 
            "task_dist_type": self.config.task_dist_type, 
            "dist_sigma": self.config.dist_sigma, 
            "dist_K": self.config.dist_K
        }
        return kwargs

    def _run_sim(self, init_weight=False):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """

        # Initial weights are assumed to be valid and optimize_waits = True
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
            new_weights = [*wait_costs, *edge_weights]
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            if self.config.optimize_wait:
                wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                               self.ub).tolist()
                new_weights = [*wait_costs, *edge_weights]
            else:
                all_weights = [self.curr_wait_costs, *edge_weights]
                all_weights = min_max_normalize(all_weights, self.lb, self.ub)
                new_weights = all_weights.tolist()

        # print(new_weights[:5])
        # raise NotImplementedError
        result_jsonstr = self.simulator.update_gg_and_step(self.config.optimize_wait, new_weights)
        result = json.loads(result_jsonstr)
        
        self.left_timesteps -= self.config.update_gg_interval
        self.left_timesteps = max(0, self.left_timesteps)
        return result

    def step(self, action):
        # self.i += 1  # increment timestep

        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs

        if self.config.optimize_wait:
            wait_cost_update_vals = action[:self.n_valid_vertices]
            edge_weight_update_vals = action[self.n_valid_vertices:]
        else:
            wait_cost_update_vals = action[0]
            edge_weight_update_vals = action[1:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        # Reward is difference between new throughput and current throughput
        result = self._run_sim()
        self.update_paths(result["all_paths"])
        new_tasks_finished = result["num_tasks_finished"]
        reward = new_tasks_finished - self.num_tasks_finished
        self.num_tasks_finished = new_tasks_finished
        
        return_result = {}
        return_result["throughput"] = self.num_tasks_finished / self.config.simulation_time

        done = result["done"]
        timeout = result["timeout"]
        congested = result["congested"]
        
        terminated = done | timeout | congested
        truncated = terminated
        
        # Info includes the results
        info = {
            "result": return_result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.num_tasks_finished = 0
        self.left_timesteps = self.config.simulation_time
        self.pos_hists = [[] for _ in range(self.num_agents)]
        # self.move_hists = [[] for _ in range(self.num_agents)]
        
        kwargs = self.gen_base_kwargs()
        curr_map_json = copy.deepcopy(self.map_json)
        curr_map_json["weight"] = False
        kwargs["map"] = json.dumps(curr_map_json)
        self.simulator = WarehouseSimulator(**kwargs)
        result_s = self.simulator.warmup()
        result = json.loads(result_s)
        self.update_paths(result["all_paths"])
        obs = self._gen_obs(result, is_init=True)
        info = {"result": {}}
        return obs, info
    
    
    def update_paths(self, all_paths):
        for aid, agent_path in enumerate(all_paths):
            if len(self.pos_hists[aid]) == 0:
                self.pos_hists[aid].extend(agent_path)
            else:
                self.pos_hists[aid].extend(agent_path[1:])
            # print(self.pos_hists[aid])

if __name__ == "__main__":
    import gin
    from env_search.utils import get_n_valid_edges, get_n_valid_vertices
    from env_search.warehouse.update_model.update_model import WarehouseCNNUpdateModel

    map_path = "maps/warehouse/human/kiva_large_w_mode.json"
    map_path = "maps/warehouse/sortation_small.json"
    cfg_file_path = "config/warehouse/online_update/33x36.gin"
    cfg_file_path = "config/warehouse/online_update/sort_200_dist_sigma.gin"
    gin.parse_config_file(cfg_file_path, skip_unknown=True)
    cfg = WarehouseConfig()
    cfg.has_future_obs = False
    cfg.warmup_time = 10
    cfg.simulation_time = 1000
    cfg.past_traffic_interval = 20
    cfg.has_traffic_obs = True
    cfg.has_gg_obs = False
    cfg.has_task_obs = True
    cfg.task_dist_update_interval = -1
    cfg.task_dist_type = "Gaussian"
    cfg.dist_sigma = 0.5
    print("cutoff:", cfg.overallCutoffTime)
    
    domain = "kiva"
    base_map_str, _ = read_in_kiva_map(map_path)
    base_map_np = kiva_env_str2number(base_map_str)
    n_valid_vertices = get_n_valid_vertices(base_map_np, domain)
    n_valid_edges = get_n_valid_edges(base_map_np, bi_directed=True, domain=domain)
    with open(map_path, "r") as f:
        map_json = json.load(f)
        
    def vis_arr(arr_, mask=None, name="test"):
        arr = arr_.copy()
        save_dir = "env_search/warehouse/plots"
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
    
    for i in range(1):
        
        env = WarehouseOnlineEnv(base_map_np, map_json, num_agents=200, eval_logdir='test', 
                             n_valid_vertices=n_valid_vertices, n_valid_edges=n_valid_edges, 
                             config=cfg, seed=0)
        cnt = 0
        obs, info = env.reset()
        # for i in range(4, 5):
        #     vis_arr(obs[i], name=f"step{cnt}_traffic{i}")
        # vis_arr(obs[-1], name=f"step{cnt}_task")
        
        done = False
        while not done:
            cnt+=1
            action = np.ones(1+n_valid_edges)
            obs, reward, terminated, truncated, info = env.step(action)
            # for i in range(4, 5):
            #     vis_arr(obs[i], name=f"step{cnt}_traffic{i}")
            # vis_arr(obs[-1], name=f"step{cnt}_task")
            done = terminated or truncated
        
        print("tp =", info["result"]["throughput"])
                
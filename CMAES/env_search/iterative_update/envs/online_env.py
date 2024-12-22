from env_search.competition.config import CompetitionConfig
from env_search.competition.update_model.utils import Map, comp_uncompress_vertex_matrix, comp_uncompress_edge_matrix
from env_search.utils import min_max_normalize, load_pibt_default_config, load_w_pibt_default_config, load_wppl_default_config, get_project_dir
from env_search.utils.logging import get_current_time_str, get_hash_file_name
from env_search.iterative_update.envs.utils import visualize_simulation
import numpy as np
import os
from gymnasium import spaces
import gymnasium
import json
import time
import subprocess
import gc

DIRECTION2ID = {
    "R":0, "D":3, "L":2, "U":1, "W":4
}

class CompetitionOnlineEnv(gymnasium.Env):
    def __init__(
        self,
        n_valid_vertices,
        n_valid_edges,
        config: CompetitionConfig,
        seed
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
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None
        
        self.rl_lb, self.rl_ub = (-10, 10)
        self.reward_scale = 10
        self.action_space = spaces.Box(low=self.rl_lb, high=self.rl_ub,
                            shape=(self.n_valid_edges + self.n_valid_vertices,))


    def generate_video(self):
        visualize_simulation(self.comp_map, self.pos_hists)
        
        
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
        
        
    def _gen_traffic_obs_new(self, is_init=False):
        h, w = self.comp_map.graph.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        
        if not is_init:
            time_range = min(self.config.past_traffic_interval, self.config.simulation_time-self.left_timesteps)
        else:
            time_range = min(self.config.past_traffic_interval, self.config.warmup_time)
            
        for t in range(time_range):
            for agent_i in range(self.config.num_agents):
                prev_x, prev_y = self.pos_hists[agent_i][-(time_range+1-t)]
                # cur_x, cur_y = self.pos_hists[agent_i][-(self.config.past_traffic_interval-t)]
                
                move = self.move_hists[agent_i][-(time_range-t)]
                id = DIRECTION2ID[move]
                if id < 4:
                    edge_usage[id, prev_x, prev_y] += 1
                else:
                    wait_usage[0, prev_x, prev_y] += 1
        
        if wait_usage.sum() != 0:
            wait_usage = wait_usage/wait_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage/edge_usage.sum() * 100
        # print("new, wait_usage:", wait_usage.max(), "edge_usage:", edge_usage.max())
        return wait_usage, edge_usage                       
                            
        
    def _gen_traffic_obs(self, result):
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])

        if self.config.use_cumulative_traffic:
            wait_usage_matrix += self.last_wait_usage
            edge_usage_matrix += self.last_edge_usage
        
        self.last_wait_usage = wait_usage_matrix
        self.last_edge_usage = edge_usage_matrix
        
        # Normalize
        # wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        # edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        
        if wait_usage_matrix.sum() != 0:
            wait_usage_matrix = wait_usage_matrix/wait_usage_matrix.sum() * 100
        if edge_usage_matrix.sum() != 0:
           edge_usage_matrix = edge_usage_matrix/edge_usage_matrix.sum() * 100
        
        h, w = self.comp_map.graph.shape
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        
        edge_usage_matrix = np.moveaxis(edge_usage_matrix, 2, 0)
        wait_usage_matrix = np.moveaxis(wait_usage_matrix, 2, 0)
        return wait_usage_matrix, edge_usage_matrix
        
    
    def _gen_task_obs(self, result):
        h, w = self.comp_map.graph.shape
        task_usage = np.zeros((1, h, w))
        for aid, goal_id in enumerate(result["final_tasks"]):
            x = goal_id // w
            y = goal_id % w
            task_usage[0, x, y] += 1
        if task_usage.sum()!=0:
            task_usage = task_usage/task_usage.sum() * 10
        return task_usage
    
    def _gen_curr_pos_obs(self, result):
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
    
    def _gen_obs(self, result, is_init=False):
        wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs_new(is_init)
        # wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs(result)
        
        wait_cost_matrix = np.array(
            comp_uncompress_vertex_matrix(self.comp_map, self.curr_wait_costs))
        edge_weight_matrix = np.array(
            comp_uncompress_edge_matrix(self.comp_map, self.curr_edge_weights))
        
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

        h, w = self.comp_map.height, self.comp_map.width
        
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
        
        edge_weight_matrix = np.moveaxis(edge_weight_matrix, 2, 0)
        wait_cost_matrix = np.moveaxis(wait_cost_matrix, 2, 0)
        
        obs = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=0,
            dtype=np.float32,
        )
        if self.config.has_future_obs:
            exec_future_usage, plan_future_usage = self._gen_future_obs(result)
            obs = np.concatenate([obs, exec_future_usage+plan_future_usage], axis=0, dtype=np.float32)
        if self.config.has_task_obs:
            task_obs = self._gen_task_obs(result)
            obs = np.concatenate([obs, task_obs], axis=0, dtype=np.float32)
        if self.config.has_curr_pos_obs:
            curr_pos_obs = self._gen_curr_pos_obs(result)
            obs = np.concatenate([obs, curr_pos_obs], axis=0, dtype=np.float32)
        # print("in step, obs.shape =", obs.shape)
        return obs

    def _run_sim(self,
                 init_weight=False,
                 manually_clean_memory=True,
                 save_in_disk=True):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """
        # cmd = f"./lifelong_comp --inputFile {self.input_file} --simulationTime {self.simulation_time} --planTimeLimit 1 --fileStoragePath large_files/"
        # print("_run_sim")
        
        # Initial weights are assumed to be valid
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                           self.ub).tolist()
        if not init_weight:
            simulation_steps = min(self.left_timesteps, self.config.update_interval)
        else:
            simulation_steps = min(self.left_timesteps, self.config.warmup_time)

        kwargs = {
            "left_w_weight": self.config.left_right_ratio, 
            "right_w_weight": 1.0, 
            "map_json_path": self.config.map_path,
            "simulation_steps": simulation_steps,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "weights": json.dumps(edge_weights),
            "wait_costs": json.dumps(wait_costs),
            "plan_time_limit": self.config.plan_time_limit,
            # "seed": int(self.rng.integers(100000)),
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": self.config.file_storage_path + "ENV",
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal, 
            "h_update_late": self.config.h_update_late
        }
        if self.config.task_dist_change_interval > 0:
            kwargs["task_random_type"] = self.config.task_random_type
        if self.config.base_algo == "pibt":
            if self.config.has_future_obs:
                kwargs["config"] = load_w_pibt_default_config()
            else:
                kwargs["config"] = load_pibt_default_config()
        elif self.config.base_algo == "wppl":
            kwargs["config"] = load_wppl_default_config()
        else:
            print(f"base algo [{self.config.base_algo}] is not supported")
            raise NotImplementedError
        
        if self.last_agent_pos is not None:
            kwargs["init_agent"] = True
            kwargs["init_agent_pos"] = str(self.last_agent_pos)
        
        if self.last_tasks is not None:
            kwargs["init_task"] = True
            kwargs["init_task_ids"] = str(self.last_tasks)
            
        if self.left_timesteps <= self.next_update_dist_time:
            assert self.config.task_dist_change_interval > 0
            # generate left and right weights
            p = self.rng.random()
            r0 = self.rng.random() * (1 - self.config.left_right_ratio_bound) + self.config.left_right_ratio_bound
            r = r0 if p < 0.5 else 1/r0
            kwargs["left_w_weight"] = r
            kwargs["right_w_weight"] = 1
            while self.next_update_dist_time >= self.left_timesteps:
                self.next_update_dist_time -= self.config.task_dist_change_interval

        if not manually_clean_memory:
            # kwargs["seed"] = int(self.rng.integers(100000))
            kwargs["seed"] = self.i
            print("seed", self.i)
            # print(kwargs)
            from simulators.wppl import py_driver as wppl_py_driver
            result_jsonstr = wppl_py_driver.run(**kwargs)
            result = json.loads(result_jsonstr)
        else:
            kwargs["seed"] = int(self.rng.integers(100000))
            if save_in_disk:
                file_dir = os.path.join(get_project_dir(), 'run_files')
                os.makedirs(file_dir, exist_ok=True)
                file_name = get_hash_file_name()
                file_path = os.path.join(file_dir, file_name)
                with open(file_path, 'w') as f:
                    json.dump(kwargs, f)
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                delimiter2 = "----DELIMITER2----DELIMITER2----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
from simulators.wppl import py_driver as wppl_py_driver
import json
import time

file_path='{file_path}'
with open(file_path, 'r') as f:
    kwargs_ = json.load(f)

t0 = time.time()
result_jsonstr = wppl_py_driver.run(**kwargs_)
t1 = time.time()
print("{delimiter2}")
print(t1-t0)
print("{delimiter2}")
result = json.loads(result_jsonstr)

np.set_printoptions(threshold=np.inf)

print("{delimiter1}")
print(result)
print("{delimiter1}")

                """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
                if os.path.exists(file_path):
                    os.remove(file_path)
                else:
                    raise NotImplementedError

            else:
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
from simulators.wppl import py_driver as wppl_py_driver
import json

kwargs_ = {kwargs}
kwargs_["seed"] = int({self.rng.integers(100000)})
result_jsonstr = wppl_py_driver.run(**kwargs_)
result = json.loads(result_jsonstr)

np.set_printoptions(threshold=np.inf)
print("{delimiter1}")
print(result)
print("{delimiter1}")
                    """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
            outputs = output.split(delimiter1)
            if len(outputs) <= 2:
                print(output)
                raise NotImplementedError
            else:
                result_str = outputs[1].replace('\n', '').replace(
                    'array', 'np.array')
                result = eval(result_str)

            gc.collect()
        self.left_timesteps -= simulation_steps
        # print("old:", result[0]["starts"])
        return result

    def step(self, action):
        self.i += 1  # increment timestep

        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs
        wait_cost_update_vals = action[:self.n_valid_vertices]
        edge_weight_update_vals = action[self.n_valid_vertices:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        result = self._run_sim()
        # print("raw:", result["num_task_finished"])
        self.num_task_finished += result["num_task_finished"]
        self.last_agent_pos = result["final_pos"]
        self.last_tasks = result["final_tasks"]
        # print("start", result["starts"])
        # print("pos", self.last_agent_pos)
        # print("task", self.last_tasks)
        # print("path", result["actual_paths"])
        assert self.starts is not None
        self.update_paths(result["actual_paths"])

        # Reward is final step update throughput
        reward = result["num_task_finished"]/self.config.simulation_time
        reward *= self.reward_scale
        
        # terminated/truncate if no left time steps
        terminated = (self.left_timesteps <= 0)
        truncated = terminated
        
        # if terminated:
        #     # print("raw tp =", result["throughput"])
        #     reward = result["num_task_finished"]
        #     # print("rew =", reward)

        result["throughput"] = self.num_task_finished/self.config.simulation_time
        # Info includes the results
        info = {
            "result": result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        if terminated:
            info["episode"] ={"r": result["throughput"], "l": self.i}
        
        return self._gen_obs(result), reward, terminated, truncated, info

    
    def reset(self, seed=None, options=None):
        self.i = 0
        self.num_task_finished = 0
        self.left_timesteps = self.config.simulation_time + self.config.warmup_time
        self.last_agent_pos = None
        self.last_tasks = None
        self.pos_hists = [[] for _ in range(self.config.num_agents)]
        self.move_hists = [[] for _ in range(self.config.num_agents)]
        
        if self.config.task_dist_change_interval > 0:
            self.next_update_dist_time = self.config.simulation_time
        else:
            self.next_update_dist_time = -1
        
        self.starts = None
        
        self.last_wait_usage = np.zeros(np.prod(self.comp_map.graph.shape))
        self.last_edge_usage = np.zeros(4*np.prod(self.comp_map.graph.shape))
        
        if self.config.reset_weights_path is None:
            self.curr_edge_weights = np.ones(self.n_valid_edges)
            self.curr_wait_costs = np.ones(self.n_valid_vertices)
        else:
            with open(self.config.reset_weights_path, "r") as f:
                weights_json = json.load(f)
            weights = weights_json["weights"]
            self.curr_wait_costs = np.array(weights[:self.n_valid_vertices])
            self.curr_edge_weights = np.array(weights[self.n_valid_vertices:])
                
        result = self._run_sim(init_weight=True)
        self.last_agent_pos = result["final_pos"]
        self.last_tasks = result["final_tasks"]
        self.starts = result["starts"]
        # print("start", self.starts)
        # print("pos", self.last_agent_pos)
        # print("task", self.last_tasks)
        # print("path", result["actual_paths"])
        self.update_paths(result["actual_paths"])
        
        
        obs = self._gen_obs(result, is_init=True)
        info = {"result": result}
        # raise NotImplementedError
        return obs, info


if __name__ == "__main__":
    import gin
    from env_search.utils import get_n_valid_edges, get_n_valid_vertices
    from env_search.competition.update_model.utils import Map
    cfg_file_path = "config/competition/test_env.gin"
    gin.parse_config_file(cfg_file_path)
    cfg = CompetitionConfig()
    cfg.has_future_obs = False
    cfg.has_curr_pos_obs = True
    cfg.warmup_time = 20
    cfg.simulation_time = 1000
    cfg.past_traffic_interval = 100
    cfg.task_dist_change_interval = -1
    cfg.update_interval = 10
    cfg.num_agents = 400
    comp_map = Map(cfg.map_path)
    domain = "competition"
    n_valid_vertices = get_n_valid_vertices(comp_map.graph, domain)
    n_valid_edges = get_n_valid_edges(comp_map.graph, bi_directed=True, domain=domain)
    
    env = CompetitionOnlineEnv(n_valid_vertices, n_valid_edges, cfg, seed=2)
    
    np.set_printoptions(threshold=np.inf)
    obs, info = env.reset()

    done = False
    while not done:
        print(env.i, env.num_task_finished)
        action = np.ones(n_valid_vertices+n_valid_edges)
        _, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        
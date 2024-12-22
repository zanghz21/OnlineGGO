import os
import gymnasium as gym
import json
import numpy as np
import copy
# import py_driver  # type: ignore # ignore pylance warning
# from simulators.rhcr import warehouse_sim  # type: ignore # ignore pylance warning

# from abc import ABC
from gymnasium import spaces

from env_search.warehouse.config import WarehouseConfig
from env_search.competition.config import CompetitionConfig
from env_search.competition.update_model.utils import (
    Map,
    comp_uncompress_edge_matrix,
    comp_uncompress_vertex_matrix,
)
from env_search.utils import (
    kiva_obj_types,
    min_max_normalize,
    kiva_uncompress_edge_weights,
    kiva_uncompress_wait_costs,
    load_pibt_default_config,
    load_wppl_default_config,
    get_project_dir,
)
from env_search.utils.task_generator import generate_task_and_agent

import gc

from env_search.utils.logging import get_current_time_str, get_hash_file_name
from env_search.traffic_mapf.config import TrafficMAPFConfig
import time
import subprocess
import shutil

def generate_hash_file_path():
    file_dir = os.path.join(get_project_dir(), 'run_files')
    os.makedirs(file_dir, exist_ok=True)
    file_name = get_hash_file_name()
    file_path = os.path.join(file_dir, file_name)
    return file_path


class TrafficFlowOfflineEnv(gym.Env):
    """Env for off+GPIBT"""

    def __init__(
        self,
        cfg: TrafficMAPFConfig, 
        seed,  
        init_weight_file=None,
    ):
        super().__init__()
        self.i = 0  # timestep
        self.config = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.comp_map = Map(self.config.map_path)        
        
        self.max_iter = self.config.iter_update_n_iters
        
        self.init_weight_file = init_weight_file
        self.action_space = spaces.Box(low=-100.0, high=100.0,
                                       shape=(self.comp_map.height, self.comp_map.width, 4))

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.comp_map.height, self.comp_map.width, 9))

    def gen_sim_kwargs(self):
        kwargs = {
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
            "dist_K": self.config.dist_K
        }
        if not self.config.gen_tasks:
            assert self.config.all_json_path is not None
            kwargs["gen_tasks"] = False
            kwargs["all_json_path"] = self.config.all_json_path
        return kwargs
        
    def _run_sim(self, init_weight=False, save_in_disk=True):
        kwargs = self.gen_sim_kwargs()
        kwargs["seed"] = int(self.rng.integers(100000))
        if init_weight:
            map_weights = self.raw_weights
        else:
            map_weights = min_max_normalize(self.raw_weights, 0.1, 100)
        kwargs["map_weights"] = json.dumps(map_weights.flatten().tolist())
        
        delimiter = "[=====delimiter======]"
        if not self.config.use_lns:
            simulator_path = "simulators.trafficMAPF_off"
        else:
            simulator_path = "simulators.trafficMAPF_off_lns"
        
        # Trick to avoid mem leak issue of the cpp simulator
        if save_in_disk:
            file_path = generate_hash_file_path()
            with open(file_path, 'w') as f:
                json.dump(kwargs, f)
            run_results = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
from {simulator_path} import py_driver
import json
import time

file_path='{file_path}'
with open(file_path, 'r') as f:
    kwargs_ = json.load(f)

results = []
for _ in range({self.config.n_sim}):
    one_sim_result_jsonstr = py_driver.run(**kwargs_)
    result_json = json.loads(one_sim_result_jsonstr)
    results.append(result_json)

print("{delimiter}")
print(results)
print("{delimiter}")

                """], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                raise NotImplementedError
        else:
            run_results = subprocess.run(['python', '-c', f"""\
import numpy as np
from {simulator_path} import py_driver
import json

kwargs_ = {kwargs}
results = []
for _ in range({self.config.n_sim}):
    one_sim_result_jsonstr = py_driver.run(**kwargs_)
    result_json = json.loads(one_sim_result_jsonstr)
    results.append(result_json)
np.set_printoptions(threshold=np.inf)
print("{delimiter}")
print(results)
print("{delimiter}")
            """], stdout=subprocess.PIPE)
        
        output = run_results.stdout.decode('utf-8')
        stderr_msg = run_results.stderr.decode('utf-8')
        # process output
        outputs = output.split(delimiter)
        if len(outputs) <= 2:
            print("==== run stdout ====")
            print(output)
            print("==== run stderr ====")
            print(stderr_msg)
            print("==== map_weights as follow ====")
            print(map_weights)
            with open("debug_net.log", 'w') as f:
                json.dump(map_weights, f)
            raise NotImplementedError
        else:
            results_str = outputs[1].replace('\n', '').replace(
                'array', 'np.array')
            # print(collected_results_str)
            results = eval(results_str)

        collect_results = {}
        for k in results[0].keys():
            collect_results[k] = np.mean([r[k] for r in results], axis=0) 
        gc.collect()
        return collect_results
    
    def _gen_obs(self, result):
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])

        if wait_usage_matrix.sum()!=0:
            wait_usage_matrix = wait_usage_matrix/wait_usage_matrix.sum() * 100
        if edge_usage_matrix.sum()!=0:
            edge_usage_matrix = edge_usage_matrix/edge_usage_matrix.sum() * 100
        
        edge_weight_matrix = min_max_normalize(self.raw_weights, 0.1, 1)

        h, w = self.comp_map.height, self.comp_map.width
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        obs = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix
            ],
            axis=2,
            dtype=np.float32,
        )
        obs = np.moveaxis(obs, 2, 0)
        return obs
    
    def step(self, action):
        '''
            action: [4, h, w]
        '''
        self.i += 1  # increment timestep
        
        _action = action * (1-self.comp_map.graph)
        self.raw_weights = np.moveaxis(_action, 0, 2)
        

        # Reward is difference between new throughput and current throughput
        result = self._run_sim()
        new_throughput = result["throughput"]
        reward = new_throughput - self.curr_throughput
        self.curr_throughput = new_throughput

        # terminated/truncate only if max iter is passed
        terminated = self.i > self.max_iter
        truncated = terminated

        # Info includes the results
        info = {
            "result": result
        }

        return self._gen_obs(result), reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        # return observation, info
        if self.init_weight_file is None or self.init_weight_file == "":
            self.raw_weights = np.ones((self.comp_map.height, self.comp_map.width, 4))
        else:
            raise NotImplementedError

        # Get baseline throughput
        self.i = 1  # We will run 1 simulation in reset
        init_result = self._run_sim(init_weight=True)
        self.init_throughput = init_result["throughput"]
        self.curr_throughput = init_result["throughput"]
        info = {
            "result": init_result,
        }
        return self._gen_obs(init_result), info

    def render(self):
        return NotImplementedError()

    def close(self):
        return NotImplementedError()


if __name__ == "__main__":
    import gin
    gin.parse_config_file("config/traffic_mapf/periodical_on/empty.gin", skip_unknown=True)
    cfg = TrafficMAPFConfig()
    env = TrafficFlowOfflineEnv(
        cfg, seed=0
    )
    
    done = False
    env.reset()
    while not done:
        action = np.random.random((4, 32, 32))
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(info["result"]["throughput"])
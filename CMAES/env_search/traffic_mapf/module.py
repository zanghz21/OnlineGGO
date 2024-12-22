import numpy as np
import json
import gc
import warnings
import subprocess
import os
import time

from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.result import TrafficMAPFResult
from env_search.utils import MIN_SCORE, get_project_dir
from env_search.utils.logging import get_current_time_str, get_hash_file_name
from env_search.iterative_update.envs.trafficflow_env import TrafficFlowOfflineEnv
from env_search.iterative_update.envs.trafficflow_online_env import TrafficFlowOnlineEnv
from env_search.traffic_mapf.update_model.update_model import CNNUpdateModel

def generate_hash_file_path():
    file_dir = os.path.join(get_project_dir(), 'run_files')
    os.makedirs(file_dir, exist_ok=True)
    file_name = get_hash_file_name()
    file_path = os.path.join(file_dir, file_name)
    return file_path
    
class TrafficMAPFModule:
    def __init__(self, config: TrafficMAPFConfig):
        self.config = config
        self.rng = np.random.default_rng(seed=self.config.seed)
        self.check_cfg()
    
    def check_cfg(self):
        if self.config.rotate_input:
            assert (self.config.output_size == 1)
    
    def gen_sim_kwargs(self, nn_weights_list):
        # nn_weights_list[-1] -= 5
        kwargs = {
            "simu_time": self.config.simu_time, 
            "map_path": self.config.map_path, 
            "win_r": self.config.win_r, 
            "rotate_input": self.config.rotate_input, 
            "has_map": self.config.has_map, 
            "has_path": self.config.has_path, 
            "has_previous": self.config.has_previous,
            "use_all_flow": self.config.use_all_flow, 
            "output_size": self.config.output_size, 
            "net_type": self.config.net_type, 
            "net_input_type": self.config.net_input_type,
            "past_traffic_interval": self.config.past_traffic_interval,  
            "network_params": json.dumps(nn_weights_list), 
            "gen_tasks": self.config.gen_tasks,  
            "num_agents": self.config.num_agents, 
            "num_tasks": self.config.num_tasks, 
            "hidden_size": self.config.hidden_size, 
            "task_assignment_strategy": self.config.task_assignment_strategy, 
            "use_cached_nn": self.config.use_cached_nn, 
            "default_obst_flow": self.config.default_obst_flow, 
            "learn_obst_flow": self.config.learn_obst_flow, 
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
        
        
    def evaluate(self, nn_weights: np.ndarray, seed: int, save_in_disk=True):
        nn_weights_list=nn_weights.tolist()
        kwargs = self.gen_sim_kwargs(nn_weights_list)
        kwargs["seed"] = int(seed)
        
        delimiter = "[=====delimiter======]"
        simulator_path = "simulators.trafficMAPF_lns" if self.config.use_lns else "simulators.trafficMAPF"
        
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
            """], stdout=subprocess.PIPE).stdout.decode('utf-8')
        
        output = run_results.stdout.decode('utf-8')
        stderr_msg = run_results.stderr.decode('utf-8')
        # process output
        outputs = output.split(delimiter)
        if len(outputs) <= 2:
            print("==== run stdout ====")
            print(output)
            print("==== run stderr ====")
            print(stderr_msg)
            print("==== nn weights as follow ====")
            print(nn_weights_list)
            with open("debug_net.log", 'w') as f:
                json.dump(nn_weights_list, f)
            raise NotImplementedError
        else:
            results_str = outputs[1].replace('\n', '').replace(
                'array', 'np.array')
            # print(collected_results_str)
            results = eval(results_str)

        collect_results = {}
        for k in results[0].keys():
            collect_results[k] = np.mean([r[k] for r in results]) 
        gc.collect()
        return collect_results
    
    def evaluate_offline(self, model_params, seed):
        '''off+GPIBT, PIU, similar logic to off+PIBT'''
        env = TrafficFlowOfflineEnv(
            cfg=self.config, seed=seed
        )
        
        update_mdl_kwargs = {}
        if self.config.iter_update_mdl_kwargs is not None:
            update_mdl_kwargs = self.config.iter_update_mdl_kwargs
            
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
        return curr_result
    
    def evaluate_period_online(self, model_params, seed):
        '''[p-on]+GPIBT'''
        env = TrafficFlowOnlineEnv(
            config=self.config, seed=seed
        )
        
        update_mdl_kwargs = {}
        if self.config.iter_update_mdl_kwargs is not None:
            update_mdl_kwargs = self.config.iter_update_mdl_kwargs
            
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
        return curr_result
    
    def evaluate_offline_in_period_on(self, off_action, seed):
        env = TrafficFlowOnlineEnv(
            config=self.config, seed=seed
        )
        off_action = off_action.reshape(env.comp_map.height, env.comp_map.width, 4)
        action = np.moveaxis(off_action, 2, 0)
        obs, info = env.reset()
        done = False
        while not done:
            obs, imp_throughput, done, _, info = env.step(action)
            curr_result = info["result"]
        return curr_result
        
    
    def process_eval_result(self, curr_result_json):
        throughput = curr_result_json.get("throughput")
        obj = throughput
        throughput_std = curr_result_json.get("throughput_std")

        return TrafficMAPFResult.from_raw(
            obj=obj, throughput=throughput, throughput_std=throughput_std
        )
    
    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)

if __name__ == "__main__":
    py_driver.run()
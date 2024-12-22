import json
import numpy as np
import simulators.offline.py_driver as py_driver
# import simulators.offline_lns.py_driver as py_driver
import time
import os

save_dir = './baseline_results'
os.makedirs(save_dir, exist_ok=True)

# map_path = "guided-pibt/benchmark-lifelong/maps/ggo_maps/33x36.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/sortation_small_kiva.map"
map_path = "guided-pibt/benchmark-lifelong/maps/warehouse_small_narrow_kiva.map"

weights_path = ""
with open(weights_path, "r") as f:
    weights_json = json.load(f)
map_weights = weights_json["weights"]

map_weights = [1] * len(map_weights)

kwargs = {
    "gen_tasks": True,  
    "map_path": map_path, 
    "num_agents": 600, 
    "num_tasks": 100000,
    "seed": 24,  
    "task_assignment_strategy": "online_generate", 
    "task_dist_change_interval": -1, 
    "task_random_type": "Gaussian", 
    "dist_sigma": 1, 
    "hidden_size": 20, 
    "simu_time": 1000, 
    "default_obst_flow": 0.0,  
    "save_path": "./results.json", 
    "map_weights": json.dumps(map_weights)
}
    
t = time.time()
result_json_s = py_driver.run(**kwargs)
print("sim_time = ", time.time()-t)
result_json = json.loads(result_json_s)
print("real time =", result_json["cpu_runtime"])
print(result_json["throughput"])
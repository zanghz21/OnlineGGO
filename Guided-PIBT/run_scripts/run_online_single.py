import json
import numpy as np
import simulators.net.py_driver as py_driver
import time
import os

print("pid =", os.getpid())

model_path = ""
with open(model_path, "r") as f:
    model_json = json.load(f)
model_params = model_json["params"]

# map_path = "guided-pibt/benchmark-lifelong/maps/ggo_maps/33x36.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/room-32-32-4.map"
map_path = "guided-pibt/benchmark-lifelong/maps/pibt_random_unweight_32x32.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/sortation_small_kiva.map"

kwargs = {
    "gen_tasks": True,  
    "map_path": map_path, 
    "num_agents": 400, 
    "num_tasks": 100000,
    "seed": 0,  
    "task_assignment_strategy": "online_generate", 
    "has_map": False, 
    "has_path": False,
    "has_previous": False, 
    "output_size": 4,
    "rotate_input": False, 
    "win_r": 2,
    "task_dist_change_interval": -1, 
    "task_random_type": "Gaussian", 
    "dist_sigma": 0.5, 
    "hidden_size": 20, 
    "simu_time": 1000, 
    "use_all_flow": True,
    "use_cached_nn": True,
    "net_type": "quad",
    "default_obst_flow": 0.0, 
    "learn_obst_flow": False, 
    "network_params": json.dumps(model_params), 
}
    
t = time.time()
result_json_s = py_driver.run(**kwargs)
print("sim_time = ", time.time()-t)
print(json.loads(result_json_s)["throughput"])
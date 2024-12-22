# import sys
# sys.path.append('build')
# sys.path.append('scripts')
import json
import numpy as np
import simulators.offline.period_on_sim as period_on_sim
import time
import os

print("pid =", os.getpid())


save_dir = './baseline_results'
os.makedirs(save_dir, exist_ok=True)

# map_path = "guided-pibt/benchmark-lifelong/maps/ggo_maps/33x36.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/room-32-32-4.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/pibt_random_unweight_32x32.map"
map_path = "guided-pibt/benchmark-lifelong/maps/sortation_small_kiva.map"
map_path = "guided-pibt/benchmark-lifelong/maps/empty-32-32.map"
kwargs = {
    "gen_tasks": True,  
    "map_path": map_path, 
    "num_agents": 800, 
    "num_tasks": 100000,
    "seed": 0,  
    "task_assignment_strategy": "online_generate", 
    "win_r": 1,
    "task_dist_change_interval": -1, 
    "task_random_type": "GaussianMixed", 
    "dist_sigma": 1, 
    "hidden_size": 20, 
    "simu_time": 1000, 
    # "map_weights": json.dumps([1]* (32*32*4)), 
    "warmup_time": 20, 
    "update_gg_interval": 20
}

simulator = period_on_sim.period_on_sim(**kwargs)
result_s = simulator.warmup()
result_json = json.loads(result_s)
print(result_json.keys())
print(result_json["num_task_finished"])
print(result_json["start"][0])

raise NotImplementedError
done = False
while not done:
    result_s = simulator.update_gg_and_step([1]* (32*32*4))
    result_json = json.loads(result_s)
    done = result_json["done"]
    print(result_json["num_task_finished"])

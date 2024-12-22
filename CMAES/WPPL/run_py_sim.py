import sys
sys.path.append('build')
sys.path.append('scripts')
from map import Map
import json
import numpy as np

with_wait_costs=True

# map=Map(map_path)
# map.print_graph(map.graph)

# map_json_path = "../maps/competition/online_map/task_dist/test.json"
# map_json_path = "../maps/competition/human/pibt_warehouse_33x36_w_mode.json"
# map_json_path = "../maps/competition/online_map/task_dist/warehouse_33x36_w.json"
map_json_path = "../maps/competition/online_map/sortation_small.json"
# map_json_path = "../maps/competition/human/pibt_room-64-64-8.json"
# map_json_path = "../maps/competition/human/pibt_random_unweight_32x32.json"
with open(map_json_path, "r") as f:
    map_json = json.load(f)
    map_json_str = json.dumps(map_json)

import py_driver # type: ignore # ignore pylance warning
import py_sim

import json
config_path="configs/test.json"

weights_path = "../logs/2024-07-16_19-42-46_overfit-ratio-r_Yzaw8d3u/action.json"
with open(weights_path, "r") as f:
    weights_json = json.load(f)
full_weights = weights_json["action"]


weights_path = "../logs/2024-07-16_19-41-01_overfit-ratio-r_QFZwgMKH/action.json"
with open(weights_path, "r") as f:
    weights_json = json.load(f)
full_weights0 = weights_json["action"]

weights_path = "../logs/2024-08-01_02-15-55_sortation-small-800_d7AahPUz/optimal_weights.json"
with open(weights_path, "r") as f:
    weights_json = json.load(f)
full_weights = weights_json["weights"]


def min_max_normalize(arr, lb, ub):
    arr = np.asarray(arr)
    min_ele = np.min(arr)
    max_ele = np.max(arr)
    if max_ele - min_ele < 1e-3:
        # Clip and then return
        return np.clip(arr, lb, ub)
    arr_norm = lb + (arr - min_ele) * (ub - lb) / (max_ele - min_ele)
    if np.any(np.isnan(arr_norm)):
        print(arr)
    return arr_norm

full_weights = min_max_normalize(full_weights, 0.1, 100).tolist()
full_weights0 = min_max_normalize(full_weights0, 0.1, 100).tolist()
print(min(full_weights0), min(full_weights))
with open(config_path) as f:
    config=json.load(f)
    config_str=json.dumps(config)

simulator=py_sim.py_sim(
    # For map, it uses map_path by default. If not provided, it'll use map_json
    # which contains json string of the map
    # map_path=map_path,
    # map_json_str = map_json_str,
    map_json_path = map_json_path,
    simulation_steps=1000,
    update_gg_interval=1000, 
    # for the problem instance we use:
    # if random then we need specify the number of agents and total tasks, also random seed,
    gen_random=True,
    # gen_random=False, 
    # agents_path="large_files/agent.agent",
    # tasks_path="large_files/task.task", 
    num_agents=800,
    # network_params=str([1, ]*2693), 
    num_tasks=100000,
    warmup_steps=10, 
    seed=0,
    # save_paths=True,
    # weight of the left/right workstation, only applicable for maps with workstations
    left_w_weight=0.1,
    right_w_weight=1,
    # else we need specify agents and tasks path to load data.
    # agents_path="example_problems/random.domain/agents/random_600.agents",
    # tasks_path="example_problems/random.domain/tasks/random-32-32-20-600.tasks",
    # weights are the edge weights, wait_costs are the vertex wait costs
    # if not specified here, then the program will use the one specified in the config file.
    # weights=compressed_weights_json_str,
    # wait_costs=compressed_wait_costs_json_str, 
    h_update_late=False,
    # task_dist_change_interval = 200,
    # task_random_type="Gaussian", 
    # dist_sigma = 1.0, 
    # weights=str([1]*2540), 
    # wait_costs=str([1]*819), 
    # weights=str(full_weights[948:]), 
    # wait_costs=str(full_weights[:948]), 
    # if we don't load config here, the program will load the default config file.
    config=config_str,    
    save_paths=True,
    # the following are some things we don't need to change in the weight optimization case.
    plan_time_limit=1, # in seconds, time limit for planning at each step, no need to change
    preprocess_time_limit=1800, # in seconds, time limit for preprocessing, no need to change
    file_storage_path="large_files/", # where to store the precomputed large files, no need to change
    task_assignment_strategy="online_generate", # how to assign tasks to agents, no need to change
    num_tasks_reveal=1, # how many new tasks are revealed, no need to change
)


warmup_r = simulator.warmup()
warmup_json = json.loads(warmup_r)
# print(warmup_json["final_tasks"])
# print(warmup_json["final_pos"])
# print(json.loads(warmup_r).keys())
# print("curr pos:", simulator.get_curr_pos())
# raise NotImplementedError
import time

cnt = 0
# new_distribution = np.random.randn(502).tolist()
# simulator.update_tasks_base_distribution(new_distribution)
nt = []
while True:
    cnt += 1
    start = time.time()

    
    # dist = simulator.get_tasks_distribution()
    # print(dist)
    # print(dist[:22])
    # update_r_str = simulator.update_gg_and_step([cnt]*2540, [cnt]*819)
    wait_costs = full_weights[:1564]
    edge_weights = full_weights[1564:]
    update_r_str = simulator.update_gg_and_step(edge_weights, wait_costs)
    # update_r_str = simulator.update_gg_and_step([1]*5064, [1]*1564)
    # update_r_str = simulator.update_gg_and_step([1]*11108, [1]*3232)
    end = time.time()
    # raise NotImplementedError
    print(end-start)
    update_r = json.loads(update_r_str)
    
    agents_finish_task = update_r["agents_finish_task"]
    # print(agents_finish_task)
    # final_pos = update_r["final_pos"]
    # print(final_pos)
    # final_task = update_r["final_tasks"]
    # curr_pos = simulator.get_curr_pos()
    # print("final task:", final_task)
    # print("curr pos:", curr_pos)
    # if cnt >= 100:
    #     break
    print(update_r["num_task_finished"])
    nt.append(update_r["num_task_finished"])
    if update_r["done"]:
        print("done!")
        break

# for i in range(len(nt)):
#     if i == 0:
#         print(nt)
#     else:
#         print(nt[i]-nt[i-1])
            
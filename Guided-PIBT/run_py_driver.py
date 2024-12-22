# import sys
# sys.path.append('build')
# sys.path.append('scripts')
import json
import numpy as np
import simulators.sum_ovc.py_driver as py_driver
import simulators.net.py_driver as py_driver
import time
import os

# import torch

# network_params = (np.random.rand(2041)).tolist()
# network_params = (np.random.rand(3811)).tolist()
# network_params = (np.ones(2104)).tolist()
# network_params = (np.ones(7804)).tolist()
# network_params = (np.ones(5254)).tolist()
# network_params = (np.random.rand(3904)).tolist()
# network_params = (np.ones(400)).tolist()
# network_params = (np.random.rand(400)).tolist()
# network_params = (np.ones(144)).tolist()
# network_params = (np.ones(192)).tolist()
# network_params = (np.ones(48)).tolist()
# network_params = (np.ones(16)).tolist()
network_params = (np.zeros(32)).tolist()
# network_params = (np.ones(560)).tolist()
# network_params = (np.ones(1060)).tolist()
# network_params=[11.062725067138672, -1.2850298881530762, -5.3751630783081055, -1.1263861656188965, 1.003326654434204, 6.403059005737305, 4.454437255859375, 1.4244210720062256, 1.229013442993164, 10.126388549804688, 1.6852002143859863, 7.424952983856201, 6.695119857788086, 7.990030765533447, 10.631780624389648, 9.203983306884766, 2.0874123573303223, 0.7973337173461914, 5.329954624176025, -8.996137619018555, 9.735237121582031, 8.354061126708984, -5.231357574462891, -3.9552555084228516, 8.280035972595215, 9.4442138671875, -9.462157249450684, 10.378240585327148, 17.135334014892578, -6.4590864181518555, 7.781689643859863, 9.199886322021484, 5.434293270111084, 4.339292049407959, 1.1298346519470215, 7.33743953704834, -1.485783576965332, 3.2927517890930176, 5.147486209869385, 1.7584948539733887, 8.357352256774902, 5.630908966064453, 6.855302333831787, 3.2735862731933594, 7.480297565460205, 9.747757911682129, -0.7206935882568359, 5.83193302154541, 1.1334388256072998, 1.9918122291564941, 12.054191589355469, 0.9323306083679199, 19.63946533203125, 5.532678604125977, 4.302480697631836, -1.838810920715332, 1.7460682392120361, 0.572967529296875, 11.216442108154297, -1.3531737327575684, -5.2931108474731445, -1.2342743873596191, -0.9438929557800293, 10.854823112487793, 6.813413619995117, 6.987953186035156, 11.434244155883789, 12.216104507446289, 7.605258941650391, 5.142612457275391, 4.989601135253906, 1.5059561729431152, 8.517690658569336, 1.8204538822174072, -2.4782886505126953, 8.158496856689453, 13.061989784240723, 2.1564741134643555, 2.8951611518859863, 5.980406761169434, 4.869112014770508, 9.872040748596191, 4.8215718269348145, 7.855231285095215, 4.538567543029785, 6.115790843963623, 7.609106063842773, 8.427629470825195, 0.8612995147705078, 6.339592933654785, 2.3355154991149902, 8.888065338134766, -3.647202491760254, 4.098965167999268, 2.805684804916382, 7.605686187744141, 4.275572776794434, 19.36056900024414, -0.07097768783569336, 10.034370422363281]

# linear baseline
# network_params = np.zeros(500)
# network_params = np.zeros((4, 100))
# network_params[:, 48:52]=500
# network_params = network_params.flatten().tolist()

# quad baseline
v_params = np.zeros((4, 100))
# v_params[:, 48:52]=50
# v_params[0, 44] =100
# v_params[1, 29] =100
# v_params[2, 54] =100
# v_params[3, 71] =100
v_params[0, 52:56] =50
v_params[1, 68:72] =50
v_params[2, 44:48] =50
v_params[3, 28:32] =50
# v_params[0, 54]=150
# v_params[1, 71]=150
# v_params[2, 44]=150
# v_params[3, 29]=150
e_params = np.zeros((4, 40))
# e_params[0, 20] = 100
# e_params[1, 14] = 100
# e_params[2, 22] = 100
# e_params[3, 23] = 100
e_params[0, 22]=100
e_params[1, 23]=100
e_params[2, 20]=100
e_params[3, 14]=100
# network_params = np.concatenate([v_params.flatten(), e_params.flatten()])
# network_params /= 100
# network_params = np.ones(125)
# network_params = np.ones(140)
# network_params = np.ones(165)
# network_params = np.ones(560)
# network_params = (v_params*100).flatten().tolist()
# network_params = network_params.flatten().tolist()

# minimum baseline
network_params = [0.5, 0.5, 0.5, 0.5, 1, 1,
                  0.5, 0.5, 0.5, 0.5, 1, 1,
                  0.5, 0.5, 0.5, 0.5, 1, 1, 
                  0.5, 0.5, 0.5, 0.5, 1, 1]

save_dir = './baseline_results'
os.makedirs(save_dir, exist_ok=True)
# for ag in [2000, 6000, 10000, 14000, 18000]:
#     for i in range(1, 6):
#         map_path = f"guided-pibt/benchmark-lifelong/sortation_medium_{i}_{ag}.json"
    # for ag in [1200, 1800, 2400, 3000, 3600]:
    #     map_path = f"guided-pibt/benchmark-lifelong/sortation_60x100_wide_{ag}.json"
all_json_path = "guided-pibt/benchmark-lifelong/sortation_small_0_800.json"
# all_json_path = "guided-pibt/benchmark-lifelong/sortation_medium_1_10000.json"
# map_path = "guided-pibt/benchmark-lifelong/room-64-64-8_0_1000.json"
    # map_path = "guided-pibt/benchmark-lifelong/warehouse_large_13_8000.json"
# kwargs = {
#     "all_json_path": all_json_path, 
#     "has_map": False, 
#     "has_path": False,
#     "has_previous": False, 
#     "output_size": 4,  
#     # "hidden_size": 20, 
#     "simu_time": 1000, 
#     "use_all_flow": True, 
#     "use_cached_nn": False, 
#     "win_r": 0, 
#     "net_type": "linear",
#     "network_params": json.dumps(network_params)
# }
map_path = "guided-pibt/benchmark-lifelong/maps/ggo_maps/33x36.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/ost003d_downsample_repaired.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/ost003d.map"
# map_path = "guided-pibt/benchmark-lifelong/maps/lak303d_downsample.map"
kwargs = {
    "gen_tasks": True,  
    "map_path": map_path, 
    "num_agents": 400, 
    "num_tasks": 100000,
    "seed": 0,  
    # "task_dist_change_interval": 200, 
    # "task_random_type": "GaussianMixed", 
    "dist_K": 3, 
    "task_assignment_strategy": "roundrobin", 
    "has_map": False, 
    "has_path": False,
    "has_previous": False, 
    "output_size": 4,
    "rotate_input": False, 
    "win_r": 2,
    "hidden_size": 20, 
    "simu_time": 10000, 
    "use_all_flow": True, 
    # "net_type": "quad", 
    "net_type": "minimum",
    "net_input_type": "minimum", 
    "network_params": json.dumps(network_params)
}
t = time.time()
result_json_s = py_driver.run(**kwargs)
result_json = json.loads(result_json_s)
print("sim_time = ", time.time()-t)
print(result_json["throughput"])
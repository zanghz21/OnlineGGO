import json
import numpy as np
import os
import matplotlib.pyplot as plt
from map_scripts.maps import TrafficMapfMap
from tqdm import tqdm
from datetime import datetime

def get_time_str():
    now = datetime.now()
    time_str = now.strftime('%y%m%d_%H%M%S')
    return time_str


def vis_arr(arr_, mask=None, name="test", save_dir=None):
    arr = arr_.copy()
    base_dir = "guide-path-vis/"
    save_dir = base_dir if save_dir is None else os.path.join(base_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    if mask is not None:
        arr = np.ma.masked_where(mask, arr)
    cmap = plt.cm.Reds
    cmap.set_bad(color='black')
    plt.imshow(arr, cmap=cmap, interpolation="none")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()
    
    
def gen_png(agent_id, timestep, agent_guide_path, map_np: np.ndarray, time_str):
    map_path = np.zeros_like(map_np)
    for i, (x, y) in enumerate(agent_guide_path):
        map_path[x, y] = 1+i/100
    vis_arr(map_path, mask=(map_np == 1), name = f"t{timestep}_a{agent_id}", save_dir=time_str)

def main():
    base_dir = "guide-paths-hm"
    map_path = "guided-pibt/benchmark-lifelong/maps/ggo_maps/33x36.map"
    map_path = "guided-pibt/benchmark-lifelong/maps/sortation_small_kiva.map"
    # map_path = "guided-pibt/benchmark-lifelong/maps/warehouse_small_kiva.map"
    
    map_np = TrafficMapfMap(map_path).map_np
    
    time_str = get_time_str()
    
    for t in range(700, 701):
        file = os.path.join(base_dir, f"t{t}.json")
        with open(file, "r") as f:
            agents_gp = json.load(f)
        for agent_id in tqdm(range(400)):
            gen_png(agent_id, t, agents_gp[agent_id], map_np, time_str)
            
if __name__ == "__main__":
    main()

        
    
    
    
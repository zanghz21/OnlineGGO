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
    plt.imshow(arr, cmap=cmap, interpolation="none", vmax=100000)
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.tick_params(labelsize=16) 
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"{name}.pdf"))
    plt.close()
    

def add_tile_usage(tile_usage, agents_path):
    for agent_path in agents_path:
        for i, (x, y) in enumerate(agent_path):
            tile_usage[x, y] += 1

 
def main():
    base_dir = "guide-paths/guide-paths-hm"
    map_path = "guided-pibt/benchmark-lifelong/maps/ggo_maps/33x36.map"
    map_path = "guided-pibt/benchmark-lifelong/maps/sortation_small_kiva.map"
    # map_path = "guided-pibt/benchmark-lifelong/maps/warehouse_small_kiva.map"
    
    map_np = TrafficMapfMap(map_path).map_np
    tile_usage = np.zeros_like(map_np)
    
    time_str = get_time_str()
    
    for t in tqdm(range(1000)):
        file = os.path.join(base_dir, f"t{t}.json")
        with open(file, "r") as f:
            agents_gp = json.load(f)
        add_tile_usage(tile_usage, agents_gp)    
    vis_arr(tile_usage, mask=(map_np == 1), name=f"{time_str}", save_dir="gp_tile_usage")
            
if __name__ == "__main__":
    main()

        
    
    
    
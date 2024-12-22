import json
from env_search.traffic_mapf.utils import Map, get_map_name
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os



def get_json_name(json_path):
    return json_path.split('/')[-1][:-5]
    
def plot_fig(data, mask, save_dir, save_name):
    h, w = mask.shape
    plt.figure(figsize=(int(w/10), int(h/10)))
    mask_data = np.ma.array(data, mask=mask)
    cmap = plt.cm.Reds
    cmap.set_bad(color='black')
    plt.imshow(mask_data, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"{save_name}.png"))
    plt.close()


def main(json_path, map_path=None, save_dir=None):
    with open(json_path, 'r') as f:
        result_json = json.load(f)
    tile_usage = result_json["tile_usage"]
    vertex_wait_mat = result_json["vertex_wait_matrix"]

    if map_path is None:
        map_path = result_json["map_path"] + '.map'
        print(map_path)
        
    mapf_map = Map(map_path)

    tile_usage = np.array(tile_usage).reshape(*mapf_map.graph.shape)
    vertex_wait_mat = np.array(vertex_wait_mat).reshape(*mapf_map.graph.shape)

    map_mask = mapf_map.graph.astype(bool)

    if save_dir is None:
        save_dir = os.path.dirname(json_path)
    plot_fig(tile_usage, map_mask, save_dir, f"{get_json_name(json_path)}_tile_usage")
    plot_fig(vertex_wait_mat, map_mask, save_dir, f"{get_json_name(json_path)}_vertex_wait")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str)
    parser.add_argument("--json_path", type=str)
    args = parser.parse_args()
    
    main(args.json_path, args.map_path)


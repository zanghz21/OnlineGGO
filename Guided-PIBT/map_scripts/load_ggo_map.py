import json
import os


def parse_single_map(map_path, save_dir):
    with open(map_path, "r") as f:
        map_json = json.load(f)
        
    map_name = map_json["name"]
    new_map_path = os.path.join(save_dir, f"{map_name}.map")
    with open(new_map_path, "w") as f:
        f.write("type octile\n")
        f.write(f"height {map_json['n_row']}\n")
        f.write(f"width {map_json['n_col']}\n")
        f.write("map\n")
        for map_line in map_json["layout"]:
            f.write(map_line.upper()+"\n")

def parse_map_dir():
    map_dir = "ggo_maps"
    save_dir = map_dir
    for root, dirs, files in os.walk(map_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            map_path = os.path.join(root, file)
            parse_single_map(map_path, save_dir)
            
if __name__ == "__main__":
    map_path = "../CMAES/maps/competition/human/pibt_random_unweight_32x32.json"
    save_dir = "guided-pibt/benchmark-lifelong/maps"
    parse_single_map(map_path, save_dir)
            
        
        
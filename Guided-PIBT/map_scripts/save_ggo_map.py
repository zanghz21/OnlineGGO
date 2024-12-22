import os
import json

map_path = ""

with open(map_path, 'r') as f:
    lines = f.readlines()
    
h = int(lines[1].split(" ")[1])
w = int(lines[2].split(" ")[1])
print(h, w)


save_dir = ""
save_name = ""
save_path = os.path.join(save_dir, save_name+".json")

map_json = {}
map_json["name"] = save_name
map_json["n_row"] = h
map_json["n_col"] = w
layout = []
for line in lines[4:]:
    layout.append(line.replace('\n', '').replace('S', 'e').replace('E', 'w'))

map_json["layout"] = layout

with open(save_path, 'w') as f:
    json.dump(map_json, f, indent=4)
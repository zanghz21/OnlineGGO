import numpy as np


class TrafficMapfMap:
    def __init__(self, fp) -> None:
        self.fp = fp
        self.read_map()
    
    def read_map(self):
        with open(self.fp, 'r') as f:
            lines = f.readlines()
        h = int(lines[1].split(" ")[1])
        w = int(lines[2].split(" ")[1])
        map_np = np.zeros((h, w))
        e_cnt, s_cnt = 0, 0
        for i, line in enumerate(lines[4:]):
            for j, s in enumerate(line):
                if s == "@" or s == "T":
                    map_np[i, j] = 1
                if s == "E":
                    map_np[i, j] = 2
                    e_cnt += 1
                if s == "S":
                    map_np[i, j] = 3
                    s_cnt += 1
        self.map_np = map_np
        self.e_cnt = e_cnt
        self.s_cnt = s_cnt


def map_np2str(map_np: np.ndarray):
    h, w = map_np.shape
    lines = []
    for i in range(h):
        line = ''
        for j in range(w):
            if map_np[i, j] == 0:
                line += '.'
            elif map_np[i, j] == 1:
                line += '@'
            elif map_np[i, j] == 2:
                line += 'E'
            elif map_np[i, j] == 3:
                line += 'S'
            else:
                print(f"{map_np[i,j]} is not supported now")
                raise NotImplementedError
        lines.append(line)
    return lines
    
def save_mapfile_from_np(map_np: np.ndarray, tgt_map_path):
    h, w = map_np.shape
    map_str = map_np2str(map_np)
    with open(tgt_map_path, 'w') as f:
        f.write("type octile\n")
        f.write(f"height {h}\n")
        f.write(f"width {w}\n")
        f.write("map\n")
        for map_line in map_str:
            f.write(map_line.upper()+"\n")
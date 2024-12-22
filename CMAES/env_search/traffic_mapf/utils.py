import os
import numpy as np

def get_map_name(map_path: str):
    map_name_list = map_path.split('/')[-1].split('.')[:-1]
    map_name = ''.join(map_name_list)
    return map_name

    
class Map:
    """Data structure used for edge weight format convertion."""

    def __init__(self, fp):
        self.fp = fp
        self.fn = get_map_name(fp)
        self.height = None
        self.width = None
        self.graph = None
        self.full_graph = None

        self.load(self.fp)

    def load(self, fp: str):
        self.fp = fp
        if fp.endswith(".map"):
            with open(fp, "r") as f:
                # skip the type line
                f.readline()
                self.height = int(f.readline().split()[-1])
                self.width = int(f.readline().split()[-1])
                self.graph = np.zeros((self.height, self.width), dtype=int)
                self.full_graph =  np.zeros((self.height, self.width), dtype=int)
                # skip the map line
                f.readline()
                for row in range(self.height):
                    line = f.readline().strip()
                    assert len(line) == self.width
                    for col, loc in enumerate(line):
                        # obstacle
                        if loc == "@" or loc == "T":
                            self.graph[row, col] = 1
                            self.full_graph[row, col] = 1
                        if loc == "E":
                            self.full_graph[row, col] = 2
                        if loc == "S":
                            self.full_graph[row, col] = 3
                        if loc == "W":
                            self.full_graph[row, col] = 4
                            
        elif fp.endswith(".grid"):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def print_graph(self, graph: np.ndarray):
        map = ""
        height, width = graph.shape
        for i in range(height):
            for j in range(width):
                map += str(graph[i, j])
            map += "\n"
        print(map)

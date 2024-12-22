import numpy as np


map_path = 'guided-pibt/benchmark-lifelong/maps/sortation_small_narrow.map'
with open(map_path, 'r') as f:
    lines = f.readlines()

h = int(lines[1].split(" ")[1])
w = int(lines[2].split(" ")[1])
print(h, w)

map_np = np.zeros((h, w))
e_cnt, s_cnt = 0, 0
for i, line in enumerate(lines[4:]):
    for j, s in enumerate(line):
        if s == "@":
            map_np[i, j] = 1
        if s == "E":
            map_np[i, j] = 2
            e_cnt += 1
        if s == "S":
            map_np[i, j] = 3
            s_cnt += 1

def manhattan_d(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.abs(x1-x2) + np.abs(y1-y2)


# print(map_np[:5, :5])
e_pos = np.where(map_np == 2)
s_pos = np.where(map_np == 3)

ds = []
for t in range(10000):
    e_id = np.random.randint(e_cnt)
    s_id = np.random.randint(s_cnt)
    
    e = e_pos[0][e_id], e_pos[1][e_id]
    s = s_pos[0][s_id], s_pos[1][s_id]
    d = manhattan_d(e,s)
    ds.append(d)
    
print("avg dist =", np.mean(ds))
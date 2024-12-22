import numpy as np
import json

def save_fig(arr, mask=None, name=None):
    import matplotlib.pyplot as plt
    if name is None:
        name = "test"
    if mask is None:
        plt.imshow(arr, cmap='Reds', interpolation='nearest')
        plt.colorbar()
        plt.savefig(f"{name}.png")
        plt.close()

def get_gaussian(h, w, center_h, center_w, sigma=1):
    x0 = np.linspace(-5, 5, w)
    y0 = np.linspace(-5, 5, h)
    x, y = np.meshgrid(x0, y0)
    z = np.exp(-((x-x0[center_w])**2 / (2*sigma**2) + (y-y0[center_h])**2/(2*sigma**2)))
    return z

def sample_2d_index(cdf, shape):
    rand = np.random.rand() * cdf[-1]
    index_1d = np.searchsorted(cdf, rand)
    return np.unravel_index(index_1d, shape)
    
total_task_num = 100000
num_agents = 800
map_path = 'maps/competition/online_map/sortation_small.json'

with open(map_path, 'r') as f:
    map_json = json.load(f)

h = map_json["n_row"]
w = map_json["n_col"]

map_e = np.zeros((h, w))
map_o = np.zeros((h, w))
map_w = np.zeros((h, w))
for (r, map_line) in enumerate(map_json["layout"]):
    for (c, s) in enumerate(map_line):
        if s == "e":
            map_e[r, c] = 1
        if s == "@":
            map_o[r, c] = 1
        if s == "w":
            map_w[r, c] = 1

def gen_task(task_list):
    center_h = np.random.randint(h)
    center_w = np.random.randint(w)
    # raise NotImplementedError
    dist_full = get_gaussian(h, w, center_h, center_w)
    dist_e = dist_full * map_e
    dist_e_flat = dist_e.flatten()
    e_cdf = np.cumsum(dist_e_flat)

    e_ids = [sample_2d_index(e_cdf, (h, w)) for _ in range(num_agents)]
    w_full_ids = np.argwhere(map_w==1)
    w_ids = w_full_ids[np.random.choice(w_full_ids.shape[0], size=num_agents, replace=True)]
    w_ids = [tuple(w_id) for w_id in w_ids]

    for e_id in e_ids:
        e_pos = e_id[0]*w + e_id[1]
        task_list.append(e_pos)
    for w_id in w_ids:
        w_pos = w_id[0]*w + w_id[1]
        task_list.append(w_pos)
        
    # map_tasks = np.zeros((h, w)) 
    # for e_id in e_ids:
    #     map_tasks[e_id] += 1
    #     assert(map_e[e_id] == 1)
    # for w_id in w_ids:
    #     map_tasks[w_id] += 1
    #     assert(map_w[tuple(w_id)] == 1)

    # save_fig(map_tasks, name="task")

def gen_agent():
    empty_full_ids = np.argwhere(map_o == 0.)
    empty_ids = empty_full_ids[np.random.choice(empty_full_ids.shape[0], size=num_agents, replace=False)]
    empty_ids = [emp_id[0]*w + emp_id[1] for emp_id in empty_ids]
    return empty_ids

    
def main():
    task_list = []
    while len(task_list) < total_task_num:
        gen_task(task_list)
    
    task_file = "maps/competition/online_map/task_dist/test.task"
    with open(task_file, "w") as f:
        f.write(f"{total_task_num}\n")
        f.writelines(
            [f"{task_pos}\n" for task_pos in task_list[:total_task_num]]
                     )
        
    agent_list = gen_agent()
    agent_file = "maps/competition/online_map/task_dist/test.agent"
    with open(agent_file, "w") as f:
        f.write(f"{num_agents}\n")
        f.writelines(
            [f"{a_pos}\n" for a_pos in agent_list]
                     )


if __name__ == "__main__":
    main()
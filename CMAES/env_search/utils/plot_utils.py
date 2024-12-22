import json
import matplotlib.pyplot as plt
import numpy as np
import os


def moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')    


def parse_save_file(file_path, timestep_start, timestep_end, seed=0, save_dir=None):
    with open(file_path, "r") as f:
        results_json = json.load(f)
    events = results_json["events"]
    t_arr = np.arange(timestep_start, timestep_end)
    tasks_arr = np.zeros(timestep_end-timestep_start)
    for agent_events in events:
        for event in agent_events:
            task_id, timestep, event_msg = event
            if event_msg == "finished":
                if timestep < timestep_start:
                    print(f"warning: exists timestep < timestep_start, skip! {timestep}<{timestep_start}")
                    continue
                if timestep >= timestep_end:
                    print(f"warning: exists timestep >= timestep_end, skip! {timestep}>={timestep_end}")
                    continue
                tasks_arr[timestep-timestep_start] += 1
                
    if save_dir is None:
        save_dir = "env_search/plots"
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(t_arr, tasks_arr, color='RoyalBlue', alpha=0.2)
    tasks_arr_smooth = moving_average(tasks_arr)
    plt.plot(t_arr[:len(tasks_arr_smooth)], tasks_arr_smooth, color='RoyalBlue')
    plt.xlabel("timestep")
    plt.ylabel("num tasks finished")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"num_tasks_finished_{seed}.png"))
    plt.close()
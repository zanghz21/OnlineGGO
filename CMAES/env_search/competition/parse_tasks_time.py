import os
import matplotlib.pyplot as plt
import json
import numpy as np

def moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

def parse_save_file(file_path, timestep_start, timestep_end):
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
    return tasks_arr


def avg_plot_on_and_off(on_arr, off_arr, t_arr):
    avg_on_arr = on_arr.mean(axis=0)
    avg_off_arr = off_arr.mean(axis=0)
    plt.figure()
    smooth_off = moving_average(avg_off_arr, window_size=3)
    plt.plot(t_arr, avg_off_arr, color="MediumOrchid", alpha=0.3)
    plt.plot(t_arr[:len(smooth_off)], smooth_off, color="MediumOrchid", label="off+PIBT(vanilla)")
    
    smooth_on = moving_average(avg_on_arr, window_size=3)
    plt.plot(t_arr, avg_on_arr, color="RoyalBlue", alpha=0.4)
    plt.plot(t_arr[:len(smooth_on)], smooth_on, color="RoyalBlue", label="on+PIBT(vanilla)")
    
    plt.xlabel("timestep", loc="center", fontsize=20)
    plt.ylabel("num of reached goals", loc="center", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("on_and_off.png")

def add_plot(data_arr, t_arr, color="MediumOrchid", label="test", window_size=3):
    smooth_arr = moving_average(data_arr, window_size=window_size)
    # plt.plot(t_arr, data_arr, color=color, alpha=0.3)
    plt.plot(t_arr[:len(smooth_arr)], smooth_arr, color=color, label=label)
    

def end_fig(save_path):
    plt.xlabel("timestep", loc="center", fontsize=24)
    plt.ylabel("num of reached goals", loc="center", fontsize=24)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    
    
def main_on_and_off():
    on_logdir = "/media/project0/hongzhi/TrafficFlowMAPF/CMAES/slurm_logs/2024-08-07_00-10-04_sortation-small-800-online-dists-traf-task_2SYntkbs"
    off_logdir = "/media/project0/hongzhi/TrafficFlowMAPF/CMAES/slurm_logs/2024-08-06_21-56-13_sortation-small-800-dists_PXHzdMrw"
    
    on_logdir = "/media/project0/hongzhi/TrafficFlowMAPF/CMAES/logs/eevee/2024-08-13_10-04-12_32x32-random-400-online-traf-task_uTsG7SQd/"
    off_logdir = "/media/project0/hongzhi/TrafficFlowMAPF/CMAES/logs/2024-07-23_13-36-24_competition-highway-32x32-cma-es-random-map-400-agents-cnn-iter-update_vFgFjt9H"
    seeds = [i for i in range(1,2)]
    
    start, end = 1, 1001
    on_tasks_arr = []
    for seed in seeds:
        file_path = os.path.join(on_logdir, f"results_{seed}.json")
        tasks_arr = parse_save_file(file_path, timestep_start=start, timestep_end=end)
        on_tasks_arr.append(tasks_arr.copy())
        
    on_tasks_arr = np.array(on_tasks_arr)
    
    off_tasks_arr = []
    for seed in seeds:
        file_path = os.path.join(off_logdir, f"results_{seed}.json")
        tasks_arr = parse_save_file(file_path, timestep_start=start, timestep_end=end)
        off_tasks_arr.append(tasks_arr.copy())
    
    off_tasks_arr = np.array(off_tasks_arr)
    
    t_arr = np.arange(start, end)
    avg_plot_on_and_off(on_tasks_arr, off_tasks_arr, t_arr)
    

def main_multi_seed():
    base_logdir = ""
    plt_title = "on+PIBT" # ["on+PIBT", "off+PIBT", "PIBT"]
    
    seeds = [i for i in range(3)]
    colors = ["Gold", "Coral", "MediumOrchid"]
    
    start, end = 1, 10001
    w_size = (end - start)//500
    t_arr = np.arange(start, end)
    
    
    plt.figure()
    for i, seed in enumerate(seeds):
        raw_logfile = os.path.join(base_logdir, f"results_{seed}.json")
        tasks_arr = parse_save_file(raw_logfile, timestep_start=start, timestep_end=end)
        add_plot(tasks_arr, t_arr, color=colors[i], label=f"{seed}", window_size=w_size)

    plt.title(plt_title, fontsize=24)
    save_path = os.path.join(base_logdir, "seed.png")
    end_fig(save_path)
    

if __name__ == "__main__":
    main_multi_seed()

    
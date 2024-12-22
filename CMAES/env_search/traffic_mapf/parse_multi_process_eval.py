import json
import argparse
from env_search.traffic_mapf.multi_process_eval import EXP_AGENTS
import os
import csv
import argparse

def parse_eval_results(base_dir, time_str, n_evals):
    for map_type in EXP_AGENTS:
        if map_type in base_dir:
            if "warehouse" in map_type:
                if "large" in base_dir:
                    map_type = "warehouse_large"
                elif "narrow" in base_dir:
                    map_type = "warehouse_small_narrow"
                elif "60x100" in base_dir:
                    map_type = "warehouse_60x100"
                else:
                    map_type = "warehouse_small"
            break

    agent_ls = EXP_AGENTS[map_type]
    print(agent_ls)
    
    base_dir = base_dir
    
    csv_base_dir = os.path.join(base_dir, "logs")
    
    for ag in agent_ls:
        csv_dir = os.path.join(csv_base_dir, f"ag{ag}")
        os.makedirs(csv_dir, exist_ok=True)
        csv_file = os.path.join(csv_dir, f"{time_str}.csv")
        with open(csv_file, mode="w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['seed', 'tp', 'sim_time'])
            # writer.writerow(['seed', 'tp'])
        
        for seed in range(n_evals):
            file_path = os.path.join(base_dir, f"ag{ag}", f"{seed}", f"{time_str}.json")
            if not os.path.exists(file_path):
                print(f"file [{file_path}] not found! skip")
                continue
            with open(file_path, 'r') as f:
                results = json.load(f)
                
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                if "sim_time" in results.keys():
                    writer.writerow([seed, results["throughput"], results["sim_time"]])
                else:
                    writer.writerow([seed, results["throughput"]])
                

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str)
    p.add_argument("--time_str", type=str)
    p.add_argument("--n_evals", type=int)
    cfg = p.parse_args()
    
    
    parse_eval_results(cfg.base_dir, cfg.time_str, cfg.n_evals)
    
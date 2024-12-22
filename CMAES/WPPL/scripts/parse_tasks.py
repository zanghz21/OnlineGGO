import json

results_file = "../large_files/results.json"

with open(results_file, "r") as f:
    results_json = json.load(f)

raw_tasks = results_json["tasks"]
tasks = [task[1]*36+task[2] for task in raw_tasks]

raw_starts = results_json["start"]
starts = [start[0]*36+start[1] for start in raw_starts]

task_file ="../large_files/task.task"
with open(task_file, 'w') as f:
    f.writelines([str(len(tasks))+"\n"])
    f.writelines([str(task)+"\n" for task in tasks])

agent_file ="../large_files/agent.agent"
with open(agent_file, 'w') as f:
    f.writelines([str(len(starts))+"\n"])
    f.writelines([str(start)+"\n" for start in starts])
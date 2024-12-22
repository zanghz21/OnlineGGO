import numpy as np
from env_search.utils.worker_state import get_learn2follow_module


def run_learn2follow(
    nn_weights: np.ndarray, 
    seed
):
    learn2follow_module = get_learn2follow_module()
    result = learn2follow_module.evaluate(nn_weights, seed)
    return result
    

def process_learn2follow_results(curr_result_jsons):
    # print("in run.py")
    learn2follow_module = get_learn2follow_module()
    
    if isinstance(curr_result_jsons, list):
        keys = curr_result_jsons[0].keys()
        curr_result_json = {key: [] for key in keys}
        for result_json in curr_result_jsons:
            for key in keys:
                curr_result_json[key].append(result_json[key])
        
        for key in keys:
            curr_result_json[key] = np.mean(curr_result_json[key])
    else:
        raise NotImplementedError
    results = learn2follow_module.process_eval_result(curr_result_json)
    return results
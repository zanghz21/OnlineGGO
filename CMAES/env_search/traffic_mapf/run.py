import numpy as np

from env_search.utils.worker_state import get_traffic_mapf_module

def run_traffic_mapf(
    nn_weights: np.ndarray, seed
):
    '''on+GPIBT'''
    traffic_mapf_module = get_traffic_mapf_module()
    result = traffic_mapf_module.evaluate(
        nn_weights, seed
    )
    return result

def run_traffic_mapf_offline(
    nn_weights: np.ndarray, seed
):
    '''off+GPIBT'''
    traffic_mapf_module = get_traffic_mapf_module()
    result = traffic_mapf_module.evaluate_offline(
        nn_weights, seed
    )
    return result

def run_traffic_mapf_period_online(
    nn_weights: np.ndarray, seed
):
    '''[p-on]+GPIBT'''
    traffic_mapf_module = get_traffic_mapf_module()
    result = traffic_mapf_module.evaluate_period_online(
        nn_weights, seed
    )
    return result

def process_traffic_mapf_results(curr_result_jsons):
    traffic_mapf_module = get_traffic_mapf_module()
    
    if isinstance(curr_result_jsons, list):
        keys = curr_result_jsons[0].keys()
        curr_result_json = {key: [] for key in keys}
        new_curr_result_json = {key: [] for key in keys}
        for result_json in curr_result_jsons:
            for key in keys:
                curr_result_json[key].append(result_json[key])
        
        for key in keys:
            new_curr_result_json[key] = np.mean(curr_result_json[key])
            new_curr_result_json[f"{key}_std"] = np.std(curr_result_json[key])
    else:
        raise NotImplementedError
    results = traffic_mapf_module.process_eval_result(new_curr_result_json)
    return results
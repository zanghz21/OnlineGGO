from env_search.learn2follow.config import Learn2FollowConfig
from env_search.learn2follow.result import Learn2FollowResult

import numpy as np
from env_search.utils import MIN_SCORE, get_project_dir
import warnings

import os
import sys
sys.path.insert(0, os.path.join(get_project_dir(), 'learn2follow'))
from learn2follow.example import run_follower, create_custom_env


class Learn2FollowModule:
    def __init__(self, config: Learn2FollowConfig) -> None:
        self.cfg = config
        
    def evaluate(self, nn_weights: np.ndarray, seed):
        self.cfg.seed = seed
        base_env = create_custom_env(self.cfg)
        model_path = "learn2follow/model/follower"
        results = run_follower(base_env, nn_weights.tolist(), model_path)
        return results
    
    def process_eval_result(self, curr_result_json):
        # print("in module")
        throughput = curr_result_json.get("avg_throughput")
        obj = throughput
        return Learn2FollowResult.from_raw(
            obj=obj, throughput=throughput
        )
    
    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)
        
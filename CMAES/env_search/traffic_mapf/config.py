import gin
from dataclasses import dataclass
from typing import Collection, Optional, Tuple, List, Callable, Dict


@gin.configurable
@dataclass
class TrafficMAPFConfig:
    map_path: str = None
    all_json_path: str = None
    simu_time: int = 1000
    n_sim: int = 1
    
    # network cfg
    rotate_input: bool = False
    win_r: int = 2
    has_map: bool = False
    has_path: bool = False
    has_previous: bool = False
    use_all_flow: bool = True
    output_size: int = 4
    hidden_size: int = 50
    net_type: str = "quad"
    net_input_type: str = "flow"
    past_traffic_interval: int = -1
    use_cached_nn: bool = True
    default_obst_flow: float = 0.0
    learn_obst_flow: bool = False
    
    # sim cfg
    use_lns: bool = False
    gen_tasks: bool = True
    num_agents: bool = gin.REQUIRED
    num_tasks: int = 100000
    seed: int = 0
    task_assignment_strategy: str = "roundrobin"
    
    # offline sim cfg
    iter_update_n_iters: int = 1
    iter_update_mdl_kwargs: Dict = None
    
    # online(periodical) sim cfg
    update_gg_interval: int = 20
    warmup_time: int = 20
    past_traffic_interval: int = 20
    reset_weights_path: str = None
    has_traffic_obs: bool = True
    has_gg_obs: bool = False
    has_task_obs: bool = True
    has_map_obs: bool = False        
    
    # varied dist
    task_dist_change_interval: int = -1
    task_random_type: str = "Gaussian"
    dist_sigma: float = 0.5
    dist_K: int = 3
        
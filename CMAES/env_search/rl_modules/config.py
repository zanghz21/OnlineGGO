import dataclasses

@dataclasses.dataclass
class RLConfig:
    total_timesteps: int = 200000
    seed: int = 0
    
    
    # policy
    lr: float = 1e-4
    entropy_coef: float = 0.0
    clip_range: float = 0.1
    gamma: float = 0.95
    ppo_epoch: int = 4
    n_steps: int = 4
    
    # log
    log_interval: int = 10
    tb_base_dir: str = "tb_dir"
    tb_dir: str = None
    
    # eval
    eval_episodes: int = 5
    eval_freq: int = 200
    

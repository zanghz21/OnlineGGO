import gin
from dataclasses import dataclass

@gin.configurable
@dataclass
class Learn2FollowConfig:
    animation = False
    num_agents = 128
    seed: int = 0
    map_name = "wfi_warehouse"
    max_episode_steps = 256
    show_map_names = False
    algorithm = 'Follower'
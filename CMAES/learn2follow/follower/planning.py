from pogema import GridConfig

# noinspection PyUnresolvedReferences
import cppimport.import_hook
# noinspection PyUnresolvedReferences
from follower_cpp.planner import planner

from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from follower.gg_network import GGCNNNetwork, GGNetConfig
from typing import List

class PlannerConfig(BaseModel):
    use_static_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True
    net_params: List[float] = None


class Planner:
    def __init__(self, cfg: PlannerConfig, net: GGCNNNetwork=None):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg
        self.gg_net = net

    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None

    def update(self, obs):
        num_agents = len(obs)
        obs_radius = len(obs[0]['obstacles']) // 2
        if self.planner is None:
            self.planner = [planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost) for _ in range(num_agents)]
            for i, p in enumerate(self.planner):
                p.set_abs_start(self.starts[i])
            if self.cfg.use_static_cost:
                pen_calc = planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost)
                penalties = pen_calc.precompute_penalty_matrix(obs_radius)
                for p in self.planner:
                    p.set_penalties(penalties)

        for k in range(num_agents):
            if obs[k]['xy'] == obs[k]['target_xy']:
                continue
            obs[k]['agents'][obs_radius][obs_radius] = 0
            self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'])
            num_occupied_matrix = self.planner[k].get_num_occupied_matrix()
            dyn_costs = self.get_dyn_costs(num_occupied_matrix)
            self.planner[k].set_dyn_costs(dyn_costs)
            obs[k]['agents'][obs_radius][obs_radius] = 1
            self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])


    def get_dyn_costs(self, num_occupied_matrix):
        if self.gg_net is None:
            dyn_costs = np.array(num_occupied_matrix) + 1
            dyn_costs = dyn_costs.tolist()
        else: 
            dyn_costs = self.gg_net.get_dyn_costs(num_occupied_matrix)
        return dyn_costs
        
    def get_path(self):
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path())
        return results


class ResettablePlanner:
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg
        self._net_cfg = GGNetConfig()
        self._agent = None
        self._network = None
        if self._cfg.net_params is not None:
            # print("calling init network")
            self.initialize_network()
            # raise NotImplementedError

    def initialize_network(self):
        self._network = GGCNNNetwork(self._net_cfg)
        self._network.set_params(self._cfg.net_params)
    
    def update(self, observations):
        return self._agent.update(observations)

    def get_path(self):
        return self._agent.get_path()

    def reset_states(self, ):
        self._agent = Planner(self._cfg, self._network)

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Learn2FollowResult:
    agg_obj: float = None
    agg_result_obj: float = None
    throughput: float = None
    failed: bool = False
    agg_measures: np.ndarray = np.zeros(2) # never be used in CMA-ES
    
    @staticmethod
    def from_raw(obj, throughput) -> None:
        return Learn2FollowResult(
            agg_obj = obj, 
            agg_result_obj = obj, 
            throughput = throughput, 
            failed = False,  #TODO: change accordingly
        )
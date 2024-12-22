from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class TrafficMAPFResult:
    agg_obj: float = None
    agg_result_obj: float = None
    
    std_obj: float = None
    
    throughput: float = None
    failed: bool = False
    agg_measures: np.ndarray = np.zeros(2) # never be used in CMA-ES
    
    @staticmethod
    def from_raw(obj, throughput, throughput_std) -> None:
        return TrafficMAPFResult(
            agg_obj = obj, 
            agg_result_obj = obj, 
            std_obj = throughput_std, 
            throughput = throughput, 
            failed = False,  #TODO: change accordingly
        )
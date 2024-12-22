import dataclasses

@dataclasses.dataclass
class ILArgs:
    seed: int = 0
    data_dir: str = "data/240717_201031"
    batch_size: int = 256
    
    device: str = "cuda:0"
    lr: float = 1e-3
    weight_decay: float = 5e-4
    total_iters: int = 100000
    
    init_ckpt_path: str = None
    loss_type: str = "mse"
    
    save_ckpt_interval: int = -1
    save_fig_interval: int = 200
    
    save_suffix: str = ''
    save_dir: str = "il_results"
    
import torch
import torch.nn as nn

class GGNetConfig:
    def __init__(self) -> None:
        self.n_in = 1
        self.n_hid = 32
        self.k = 3
        
        
class GGCNNNetwork(nn.Module):
    def __init__(self, cfg: GGNetConfig):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Conv2d(self.cfg.n_in, self.cfg.n_hid, self.cfg.k, 1, self.cfg.k//2), 
            nn.ReLU(), 
            nn.Conv2d(self.cfg.n_hid, self.cfg.n_hid, 1, 1, 0), 
            nn.ReLU(), 
            nn.Conv2d(self.cfg.n_hid, 1, 1, 1, 0), 
            nn.ReLU() # ensure the output is no smaller than 0
        )
        self.n_params = sum(p.numel() for p in self.model.parameters())
        # print(f"num of params: {self.n_params}")
        
    def set_params(self, new_params):
        assert (len(new_params) == self.n_params)
        
        new_params = torch.tensor(new_params, dtype=float)
        index = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.data = new_params[index:index+numel].view_as(param).data
            index += numel
    
    def get_dyn_costs(self, num_occupations):
        x = torch.tensor(num_occupations, dtype=float).unsqueeze(0).unsqueeze(0)
        o = self.model(x)
        output = o.squeeze()
        output = output.detach().cpu().tolist()
        return output

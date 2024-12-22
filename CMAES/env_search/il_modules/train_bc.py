import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from env_search.il_modules.config import ILArgs
from env_search.il_modules.data_provider import DataProvider
from env_search.competition.update_model.update_model import CompetitionCNNUpdateModel, CNNMLPModel, LargerCNNUpdateModel
from env_search.utils.logging import get_current_time_str
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices

def weighted_mse(predict, label, weights):
    bsz = weights.shape[0]
    square_diff = (predict - label)**2
    weighted_square_diff = weights.view(-1, 1) * square_diff.reshape(bsz, -1)
    # print("square diff ", square_diff.max(), square_diff.min())
    # print("weighted square diff ", weighted_square_diff.max(), weighted_square_diff.min())
    # print("weights:", weights.max(), weights.min())
    return weighted_square_diff.mean()
    
    
class BCTrainer:
    def __init__(self, student_model, il_args: ILArgs) -> None:
        self.cfg = il_args
        self.time_str = get_current_time_str()
        self.model = student_model.model.to(self.cfg.device)
        
        self.data_provider = DataProvider(self.cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.cfg.lr, 
                                          weight_decay=self.cfg.weight_decay)
        # self.lr_scheduler = torch.optim.lr_scheduler.LRScheduler()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.total_iters
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, total_iters=self.cfg.total_iters)
        

    def train(self):        
        if self.cfg.init_ckpt_path is not None:
            self.model.load_state_dict(torch.load(self.cfg.init_ckpt_path)['state_dict'])
        for train_iter in tqdm(range(1, self.cfg.total_iters + 1)):
            iter_start_time = time.time()
            batch_data = self.data_provider.get_batch_data(self.cfg.device)
            batch_obs, batch_actions = batch_data['observations'], batch_data["actions"]
            self.optimizer.zero_grad()
            model_actions = self.model(batch_obs)
            # model_actions = torch.clip(model_actions, min=0, max=10)
            if self.cfg.loss_type == 'mse':
                ignore_mask = (batch_actions == -100.0)
                mask_model_actions = model_actions.reshape(*batch_actions.shape) * (~ignore_mask)
                mask_batch_actions = batch_actions * (~ignore_mask)
                loss = nn.functional.mse_loss(mask_model_actions, mask_batch_actions, reduction='mean')
            elif self.cfg.loss_type == 'nll':
                raise NotImplementedError
                # can less than 0 in continuous case
                loss = - self.model.actor.action_dist.log_prob(batch_actions).mean()
            else:
                raise NotImplementedError
            
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            if self.cfg.save_ckpt_interval > 0 and train_iter % self.cfg.save_ckpt_interval == 0:
                ckpt = {"loss": loss.item(), "state_dict": self.model.state_dict()}
                save_dir = os.path.join(self.cfg.save_dir, self.time_str, 'ckpt')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'ckpt_{train_iter}' + self.cfg.save_suffix +'.pth')
                torch.save(ckpt, save_path)
            
            if self.cfg.save_fig_interval > 0 and train_iter % self.cfg.save_fig_interval == 0:
                plt.figure("loss curve")
                plt.yscale('log')
                plt.plot(train_iter, loss.item(), 'bo', markersize=0.5)
                save_dir =  os.path.join(self.cfg.save_dir, self.time_str, 'plot')
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'train_loss' + self.cfg.save_suffix +'.png'))
                
            # iter_time = time.time()-iter_start_time
            # print(f"iter_time: {iter_time}, loss: {loss.item()}")
            
            if train_iter % 1000 == 0:
                print(f"", end="", flush=True)
            
            # if train_iter > 10:
            #     raise NotImplementedError    
        if self.cfg.save_ckpt_interval > 0:
            plt.close()
        
        save_dir = os.path.join(self.cfg.save_dir, self.time_str, 'ckpt')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'ckpt_final' + self.cfg.save_suffix +'.pth')
        ckpt = {"loss": loss.item(), "state_dict": self.model.state_dict()}
        torch.save(ckpt, save_path)
    
    def eval(self):
        pass


def parse_map(map_path):
    comp_map = Map(map_path)
    n_e = get_n_valid_edges(comp_map.graph, True, "competition")
    n_v = get_n_valid_vertices(comp_map.graph, "competition")
    return comp_map, n_e, n_v

if __name__ == '__main__':
    map_path = "maps/competition/human/pibt_warehouse_33x36_w_mode.json"
    il_args = ILArgs()
    comp_map, n_e, n_v = parse_map(map_path)
    
    nc = 6
    student_model = CompetitionCNNUpdateModel(
        comp_map, 
        model_params=None, 
        n_valid_vertices=n_v,
        n_valid_edges=n_e, 
        nc=nc,
        kernel_size=5 
    )
    print(student_model.num_params)
    raise NotImplementedError
    student_model = CNNMLPModel(
        comp_map, nc
    )
    # student_model = LargerCNNUpdateModel(
    #     comp_map, 
    #     model_params=None, 
    #     n_valid_vertices=n_v,
    #     n_valid_edges=n_e, 
    #     nc=nc, 
    #     kernel_size=5
    # )
    bc_trainer = BCTrainer(student_model, il_args)
    bc_trainer.train()
    
    
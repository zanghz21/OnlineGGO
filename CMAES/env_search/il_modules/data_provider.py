import numpy as np
import torch
from env_search.il_modules.config import ILArgs
from env_search.utils import min_max_normalize
import os
from scipy.ndimage import convolve, gaussian_filter


class DataProvider:
    def __init__(self, data_config: ILArgs) -> None:
        self.cfg = data_config
        self.np_random = np.random.RandomState(self.cfg.seed)
        self.load_all_data()
        
    
    def load_all_data(self):        
        filenames = []
        for root, dirs, files in os.walk(self.cfg.data_dir):
            for file in files:
                if file.endswith('.npy'):
                    filenames.append(os.path.join(root, file))

        self.observations = []
        self.actions = []
        
        for filename in filenames:
            data = np.load(filename, allow_pickle=True)
            
            for d in data:
                obs = d['obs']
                action = d['action']
                
                blur = False
                if blur:
                    # kernel = np.ones((3, 3))
                    # padded_array = np.pad(obs, pad_width=1, mode='constant', constant_values=0)
                    # # Apply the 3x3 kernel to the padded array
                    # obs = convolve(padded_array, kernel, mode='constant', cval=0.0)
                    obs = gaussian_filter(obs, sigma=0.2)
                if len(obs.shape) == 2:
                    obs = obs.reshape(1, *obs.shape)
                
                self.observations.append(obs)
                # self.actions.append(min_max_normalize(action, -10, 10))
                self.actions.append(action)
                  
        self.data_size = len(self.actions)
        print("data size =", self.data_size)
        self.observations = np.stack(self.observations, axis=0)
        self.actions = np.stack(self.actions, axis=0)
        
    def get_batch_data(self, device = 'cpu'):
        batch_idx = self.np_random.choice(self.data_size, self.cfg.batch_size)
        return self.get_batch_data_given_id(batch_idx, device)
    
    def get_batch_data_given_id(self, batch_idx, device = 'cpu'):
        batch_obs = torch.tensor(self.observations[batch_idx], dtype=torch.float32).to(device=device)
        batch_action = torch.tensor(self.actions[batch_idx], dtype=torch.float32).to(device=device)
        batch_data = {
            "observations": batch_obs, 
            "actions": batch_action
        }
        return batch_data
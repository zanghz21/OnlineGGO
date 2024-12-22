from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import Space, spaces
import torch
import torch.nn as nn

def get_equal_size_cnn(n_input_channels, n_hid_chan, kernel_size, has_flatten, has_bcn, n_layers=3):
    if has_bcn:
        nn_list = [nn.Conv2d(n_input_channels, n_hid_chan, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(n_hid_chan), 
                nn.Conv2d(n_hid_chan, n_hid_chan, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(n_hid_chan), 
                nn.Conv2d(n_hid_chan, 5, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(5)]
    else:
        nn_list = [nn.Conv2d(n_input_channels, n_hid_chan, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
                nn.LeakyReLU()]
        mid_layer = [nn.Conv2d(n_hid_chan, n_hid_chan, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
                nn.LeakyReLU()]  # Pattern to repeat
        for l in range(n_layers - 2):
            for m in mid_layer:
                nn_list.append(m)
        nn_list += [nn.Conv2d(n_hid_chan, 5, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
                nn.LeakyReLU()]
    if has_flatten:
        nn_list.append(nn.Flatten())
    cnn = nn.Sequential(*nn_list)
    return cnn


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    '''
     structure: CNN + 1 linear layer
     CNN structure is copied from PIU
    '''
    def __init__(self, observation_space: Space, features_dim: int = 128, 
                 n_hid_chan = 32, 
                 kernel_size = 3, 
                 n_layers = 3) -> None:
        super().__init__(observation_space, features_dim)
        # need update
        n_input_channels = observation_space.shape[0]
        assert kernel_size % 2 == 1
        self.cnn = get_equal_size_cnn(n_input_channels, n_hid_chan, kernel_size, has_flatten=True, has_bcn=True, n_layers=n_layers)
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def forward(self, observations: torch.Tensor):
        return self.linear(self.cnn(observations))
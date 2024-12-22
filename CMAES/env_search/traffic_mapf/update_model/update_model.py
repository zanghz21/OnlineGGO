import gin
import numpy as np
import torch
import torch.nn as nn
from env_search.utils import n_params


@gin.configurable
class CNNUpdateModel:
    """Convolutional Neural Network (CNN) update model use a CNN to get the
    update values.
    """

    def __init__(
        self,
        model_params,
        nc: int = 9,
        kernel_size: int = 3,
        n_hid_chan: int = 32,
    ):

        self.nc = nc
        self.kernel_size = kernel_size
        self.n_hid_chan = n_hid_chan

        # We want the input and output to have the same W and H for each conv2d
        # layer, so we add padding.
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.padding = padding

        self.model = self._build_model(
            nc,
            kernel_size,
            padding,
            n_hid_chan,
        )

        # Set params
        self.num_params = n_params(self.model)
        if model_params is not None:
            self.set_params(model_params)


    def get_update_values_from_obs(
        self, obs: np.ndarray
    ):
        obs = torch.from_numpy(obs).to(torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model.forward(obs)
            output = output.squeeze().cpu().numpy()
        return output

    def _build_model(
        self,
        nc,
        kernel_size,
        padding,
        n_hid_chan,
    ):
        model = nn.Sequential()

        # Three layers of conv2d
        self.n_in_chan = nc
        model.add_module(
            f"initial:conv:in_chan-{n_hid_chan}",
            nn.Conv2d(
                self.n_in_chan,
                n_hid_chan,
                kernel_size,
                1,
                padding,
                bias=True,
            ),
        )
        model.add_module(f"initial:relu", nn.ReLU(inplace=True))
        model.add_module(f"initial:BatchNorm", nn.BatchNorm2d(self.n_hid_chan))

        model.add_module(
            f"internal1:conv:{n_hid_chan}-{n_hid_chan}",
            nn.Conv2d(n_hid_chan, n_hid_chan, 1, 1, 0, bias=True),
        )
        model.add_module(f"internal1:relu", nn.ReLU(inplace=True))
        model.add_module(f"internal1:BatchNorm", nn.BatchNorm2d(self.n_hid_chan))

        model.add_module(
            f"internal2:conv:{n_hid_chan}-4",
            nn.Conv2d(n_hid_chan, 4, 1, 1, 0, bias=True),
        )
        model.add_module(f"internal2:relu", nn.ReLU(inplace=True))
        model.add_module(f"internal2:BatchNorm", nn.BatchNorm2d(4))
        # model.add_module("internal2:sigmoid", nn.Sigmoid())

        return model

    def set_params(self, weights):
        """Set the params of the model

        Args:
            weights (np.ndarray): weights to set, 1D numpy array
        """
        with torch.no_grad():
            if weights.shape != (self.num_params,):
                print(f"num params should be {self.num_params}")
                raise NotImplementedError

            state_dict = self.model.state_dict()

            s_idx = 0
            for param_name in state_dict:
                if "BatchNorm.running_mean" in param_name or \
                   "BatchNorm.running_var" in param_name or \
                   "BatchNorm.num_batches_tracked" in param_name:
                    continue
                param_shape = state_dict[param_name].shape
                param_dtype = state_dict[param_name].dtype
                param_device = state_dict[param_name].device
                curr_n_param = np.prod(param_shape)
                to_set = torch.tensor(
                    weights[s_idx:s_idx + curr_n_param],
                    dtype=param_dtype,
                    requires_grad=True,  # May used by dqd
                    device=param_device,
                )
                to_set = torch.reshape(to_set, param_shape)
                assert to_set.shape == param_shape
                s_idx += curr_n_param
                state_dict[param_name] = to_set

            # Load new params
            self.model.load_state_dict(state_dict)

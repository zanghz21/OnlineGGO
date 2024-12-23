import gin
import numpy as np
import torch

from abc import ABC
from torch import nn
from env_search.utils import n_params, min_max_normalize
from env_search.competition.update_model.utils import (
    comp_compress_edge_matrix,
    comp_compress_vertex_matrix,
)


class CompetitionBaseUpdateModel(ABC):
    """Base class for update models

    Args:
        comp_map (Map): Helper object to convert representation of edge/waits
        model_params (float or array-like): parameter(s) of the model
    """

    def __init__(self, comp_map, model_params, n_valid_vertices, n_valid_edges):
        self.comp_map = comp_map
        self.model_params = model_params
        self.n_valid_vertices = n_valid_vertices
        self.n_valid_edges = n_valid_edges

    def get_update_values(
        self,
        wait_usage_matrix,
        edge_usage_matrix,
        wait_cost_matrix,
        edge_weight_matrix,
    ):
        """Return the update value in compressed list format.

        Args:
            wait_usage_matrix (array-like): shape[h, w] usage of wait action in
                each vertex
            edge_usage_matrix (array-like): shape [h, w, 4], usage matrix of
                edge
            wait_cost_matrix (array-like): shape [h, w], matrix of wait cost
            edge_weight_matrix (array-like): shape [h, w, 4], edge weight matrix

        Returns:
            wait_cost_update_val (list): update vals of wait costs
            edge_weight_update_val (list): update vals of edge weights
        """


@gin.configurable
class CompetitionCNNUpdateModel(CompetitionBaseUpdateModel):
    """Convolutional Neural Network (CNN) update model use a CNN to get the
    update values.
    """

    def __init__(
        self,
        comp_map,
        model_params,
        n_valid_vertices,
        n_valid_edges,
        nc: int = 10,
        kernel_size: int = 3,
        n_hid_chan: int = 32,
    ):
        super().__init__(
            comp_map,
            model_params,
            n_valid_vertices,
            n_valid_edges,
        )

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

    def get_update_values(
        self,
        wait_usage_matrix,
        edge_usage_matrix,
        wait_cost_matrix,
        edge_weight_matrix,
    ):

        h, w = self.comp_map.height, self.comp_map.width
        edge_usage_matrix = edge_usage_matrix.reshape(1, h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(1, h, w, 1)
        edge_weight_matrix = edge_weight_matrix.reshape(1, h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(1, h, w, 1)

        input = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=3,
        )
        input = np.moveaxis(input, 3, 1)
        input = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            output = self.model.forward(input)
            output = output.squeeze().cpu().numpy()

        edge_weight_update_vals = np.moveaxis(output[:4], 0, 2)
        wait_cost_update_vals = output[-1]
        edge_weight_update_vals = comp_compress_edge_matrix(
            self.comp_map,
            edge_weight_update_vals.flatten(),
        )
        wait_cost_update_vals = comp_compress_vertex_matrix(
            self.comp_map,
            wait_cost_update_vals.flatten(),
        )

        return np.array(wait_cost_update_vals), np.array(
            edge_weight_update_vals)
    
    def get_update_values_from_obs(
        self, obs: np.ndarray
    ):
        obs = torch.from_numpy(obs).to(torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model.forward(obs)
            output = output.squeeze().cpu().numpy()

        edge_weight_update_vals = np.moveaxis(output[:4], 0, 2)
        wait_cost_update_vals = output[-1]
        edge_weight_update_vals = comp_compress_edge_matrix(
            self.comp_map,
            edge_weight_update_vals.flatten(),
        )
        wait_cost_update_vals = comp_compress_vertex_matrix(
            self.comp_map,
            wait_cost_update_vals.flatten(),
        )

        return np.array(wait_cost_update_vals), np.array(
            edge_weight_update_vals)

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
            f"internal2:conv:{n_hid_chan}-5",
            nn.Conv2d(n_hid_chan, 5, 1, 1, 0, bias=True),
        )
        model.add_module(f"internal2:relu", nn.ReLU(inplace=True))
        model.add_module(f"internal2:BatchNorm", nn.BatchNorm2d(5))
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


class CNNMLPModel:
    def __init__(self, comp_map, nc, k=3, hid_c = 16) -> None:
        self.n_in_chan = nc
        self.n_hid_chan = hid_c
        self.kernel_size = k
        self.comp_map = comp_map
        self.action_size = self.comp_map.height * self.comp_map.width * 5
        self._build_model()
    
    def _build_model(self):
        padding = self.kernel_size // 2
        self.model = nn.Sequential(
            nn.Conv2d(self.n_in_chan, self.n_hid_chan, self.kernel_size, 1, padding, bias=True), 
            nn.ReLU(), 
            nn.Conv2d(self.n_hid_chan, self.n_hid_chan, 1, 1, 0, bias=True), 
            nn.ReLU(), 
            nn.Conv2d(self.n_hid_chan, 5, 1, 1, 0, bias=True), 
            nn.Flatten(), 
            nn.Linear(self.action_size, self.action_size)
        )
    
    def get_update_values_from_obs(
        self, obs: np.ndarray
    ):
        obs = torch.from_numpy(obs).to(torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model.forward(obs)
            output = output.squeeze().cpu().numpy()
            output = output.reshape(5, self.comp_map.height, self.comp_map.width)

        edge_weight_update_vals = np.moveaxis(output[:4], 0, 2)
        wait_cost_update_vals = output[-1]
        edge_weight_update_vals = comp_compress_edge_matrix(
            self.comp_map,
            edge_weight_update_vals.flatten(),
        )
        wait_cost_update_vals = comp_compress_vertex_matrix(
            self.comp_map,
            wait_cost_update_vals.flatten(),
        )

        return np.array(wait_cost_update_vals), np.array(
            edge_weight_update_vals)
        

class LargerCNNUpdateModel(CompetitionBaseUpdateModel):
    """Convolutional Neural Network (CNN) update model use a CNN to get the
    update values.
    """

    def __init__(
        self,
        comp_map,
        model_params,
        n_valid_vertices,
        n_valid_edges,
        nc: int = 10,
        kernel_size: int = 3,
        n_hid_chan: int = 32,
    ):
        super().__init__(
            comp_map,
            model_params,
            n_valid_vertices,
            n_valid_edges,
        )

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

    def get_update_values(
        self,
        wait_usage_matrix,
        edge_usage_matrix,
        wait_cost_matrix,
        edge_weight_matrix,
    ):

        h, w = self.comp_map.height, self.comp_map.width
        edge_usage_matrix = edge_usage_matrix.reshape(1, h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(1, h, w, 1)
        edge_weight_matrix = edge_weight_matrix.reshape(1, h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(1, h, w, 1)

        input = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=3,
        )
        input = np.moveaxis(input, 3, 1)
        input = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            output = self.model.forward(input)
            output = output.squeeze().cpu().numpy()

        edge_weight_update_vals = np.moveaxis(output[:4], 0, 2)
        wait_cost_update_vals = output[-1]
        edge_weight_update_vals = comp_compress_edge_matrix(
            self.comp_map,
            edge_weight_update_vals.flatten(),
        )
        wait_cost_update_vals = comp_compress_vertex_matrix(
            self.comp_map,
            wait_cost_update_vals.flatten(),
        )

        return np.array(wait_cost_update_vals), np.array(
            edge_weight_update_vals)
    
    def get_update_values_from_obs(
        self, obs: np.ndarray
    ):
        obs = torch.from_numpy(obs).to(torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model.forward(obs)
            output = output.squeeze().cpu().numpy()

        edge_weight_update_vals = np.moveaxis(output[:4], 0, 2)
        wait_cost_update_vals = output[-1]
        edge_weight_update_vals = comp_compress_edge_matrix(
            self.comp_map,
            edge_weight_update_vals.flatten(),
        )
        wait_cost_update_vals = comp_compress_vertex_matrix(
            self.comp_map,
            wait_cost_update_vals.flatten(),
        )

        return np.array(wait_cost_update_vals), np.array(
            edge_weight_update_vals)

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
            nn.Conv2d(n_hid_chan, n_hid_chan, 3, 1, 1, bias=True),
        )
        model.add_module(f"internal1:relu", nn.ReLU(inplace=True))
        model.add_module(f"internal1:BatchNorm", nn.BatchNorm2d(self.n_hid_chan))

        model.add_module(
            f"internal2:conv:{n_hid_chan}-5",
            nn.Conv2d(n_hid_chan, 5, 1, 1, 0, bias=True),
        )
        model.add_module(f"internal2:relu", nn.ReLU(inplace=True))
        model.add_module(f"internal2:BatchNorm", nn.BatchNorm2d(5))
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
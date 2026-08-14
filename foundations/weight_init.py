import torch
import torch.nn as nn
import math
from typing import List
# import numpy as np


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        # NumPy:
        # mean = 0
        # std = np.sqrt(2 / (fan_in + fan_out))
        # return np.random.normal(mean, std, size=(fan_out, fan_in))

        # PyTorch:
        torch.manual_seed(0)
        mean = 0
        std = math.sqrt(2 / (fan_in + fan_out))
        matrix = torch.randn(fan_out, fan_in) * std
        return torch.round(matrix, decimals=4).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        
        # PyTorch:
        torch.manual_seed(0)
        mean = 0
        std = math.sqrt(2 / fan_in)
        matrix = torch.randn(fan_out, fan_in) * std
        return torch.round(matrix, decimals=4).tolist()
        

    # def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
    #     # Forward random input through num_layers with the given init_type.
    #     # Use torch.manual_seed(0) once at the start.
    #     # Return the std of activations after each layer, rounded to 2 decimals.
        
    #     torch.manual_seed(0)
    #     x = torch.randn(1, input_dim)       # random-value input vector, (1 x input_dim) vector
    #     weights = []
    #     weight_initializer = self.xavier_init if init_type == "xavier" else self.kaiming_init
    #     weights.append(torch.tensor(weight_initializer(fan_in=input_dim, fan_out=hidden_dim)))        # W1
    #     for i in range(1, num_layers):
    #         weights.append(torch.tensor(weight_initializer(fan_in=hidden_dim, fan_out=hidden_dim)))   # W2 - Wn
    #     stds = []
    #     for w in weights:
    #         z = x @ w.T
    #         x = torch.relu(z)
    #         stds.append(round(x.std().item(), 2))
    #     return stds

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> list[float]:
        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds











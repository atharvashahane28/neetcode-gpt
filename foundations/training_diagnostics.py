import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = round(torch.mean(x).item(), 4)
                    std = round(torch.std(x).item(), 4)
                    dead = round(((x <= 0).all(dim=0).sum() / x.shape[1]).item(), 4)
                    print(dead)
                    entry = {
                        "mean": mean,
                        "std": std,
                        "dead_fraction": dead
                    }
                    res.append(entry)
        return res


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        y_pred = model(x)                   # forward pass
        loss = nn.MSELoss()(y_pred, y)      # calculate loss
        loss.backward()                     # backward pass
        res = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                gradient = layer.weight.grad
                mean = round(torch.mean(gradient).item(), 4)
                std = round(torch.std(gradient).item(), 4)
                norm = round(torch.norm(gradient).item(), 4)
                res.append({"mean": mean, "std": std, "norm": norm})
        return res


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for act, grad in zip(activation_stats, gradient_stats):
            if act["dead_fraction"] > 0.5:
                return "dead_neurons"
            elif grad["norm"] > 1000:
                return "exploding_gradients"
            elif grad["norm"] < 1e-5:
                return "vanishing_gradients"
            elif act["std"] < 0.1:
                return "vanishing_gradients"
            elif act["std"] > 10:
                return "exploding_gradients"
        return "healthy"

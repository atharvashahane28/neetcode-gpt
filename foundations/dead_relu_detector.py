import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_neurons = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    dead = round(((x == 0).all(dim=0).sum() / x.shape[1]).item(), 4)
                    dead_neurons.append(dead)
        return dead_neurons


    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        strictly_increasing = True
        prev = float("-inf")
        for dead in dead_fractions:
            if dead > 0.5:
                return "use_leaky_relu"
            if dead <= prev:
                strictly_increasing = False
            prev = dead
        if dead_fractions[0] > 0.3:
            return "reinitialize"
        elif strictly_increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"
        else:
            return "healthy"

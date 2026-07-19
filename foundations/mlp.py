import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        
        def relu(z):
            # Returns ReLU(z)
            zeros = np.zeros(z.shape[0])
            return np.maximum(zeros, z)

        num_layers = len(weights)
        z = x
        for i in range(num_layers):
            nxt = relu(z @ weights[i] + biases[i])      # calculate current logits/neurons/activations
            z = nxt
        return np.round(z, 5)
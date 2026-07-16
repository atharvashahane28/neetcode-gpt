import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)

        # Define sigmoid and relu functions that work with single numbers
        def sigmoid(x: np.float64) -> np.float64:
            return 1 / (1 + np.e ** (-1 * x))

        def relu(x: np.float64) -> np.float64:
            return max(np.float64(0), x)

        # linear transformation operation
        res = np.float64(np.dot(x, w) + b)      # res is a single number, since this is only over 1 neuron
        # pass through non-linear activation function
        if activation == "sigmoid":
            res = sigmoid(res)
        elif activation == "relu":
            res = relu(res)
        else:
            return 0
        return np.round(res, 5)     # round and return
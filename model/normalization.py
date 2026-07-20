import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mean = x.mean()             # or np.mean(x) or torch.mean(x) if x was a tensor
        var = x.std() ** 2          # same as above but std instead of mean
        eps = 1e-5
        res = (x - mean) / (math.sqrt(var + eps)) * gamma + beta
        return np.round(res, 5)

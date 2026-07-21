import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            eps = 1e-5
            res = (x - mean) / np.sqrt(var + eps) * gamma + beta
            res = np.round(res, 4)
            # Ignore squiggly red error lines, it's some compiler fault
            running_mean = running_mean * (1 - momentum) + momentum * mean
            running_mean = np.round(running_mean, 4)
            running_var = running_var * (1 - momentum) + momentum * var
            running_var = np.round(running_var, 4)
            return res.tolist(), running_mean.tolist(), running_var.tolist()
        else:
            res = (x - running_mean) / np.sqrt(running_var + eps)
            return np.round(res, 4).tolist(), running_mean.tolist(), running_var.tolist()

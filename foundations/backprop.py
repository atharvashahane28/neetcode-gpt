import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return 1 / (1 + np.e ** (-1 * x))

        def forward(x: NDArray[np.float64], w: NDArray[np.float64], b: np.float64) -> np.float64:
            return sigmoid(np.dot(x, w) + b)
        
        y_pred = forward(x, w, np.float64(b))       # find y_pred/y^hat using forward function
        loss = 0.5 * ((y_pred - y_true) ** 2)       # calculate loss
        # dL/dw = dL/d(y_pred) * d(y_pred)/dw
        # dL/db = dL/d(y_pred) * d(y_pred)/db
        dL_dw = (y_pred - y_true) * (y_pred) * (1 - y_pred) * x     # find dL/dw
        dL_db = (y_pred - y_true) * (y_pred) * (1 - y_pred)         # find dL/db
        return np.round(dL_dw, 5), np.round(dL_db, 5)               # round and return
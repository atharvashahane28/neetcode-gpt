from _collections_abc import dict_keys
import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        def relu(z):
            # Returns ReLU(z), where z is a vector of logits/neurons
            zeros = np.zeros(z.shape[0])
            res = np.maximum(z, zeros)
            return res

        def forward(x, W1, W2):
            # Carries out 1 forward pass
            # Number of neurons in:
            # Input vector = m
            # Hidden layer = h
            # Output layer = n
            # x = input vector (1 x m)
            # W1 = weight matrix between input -> hidden (h x m)
            # W2 = weight matrix between hidden -> output (n x h)
            z1 = np.squeeze(np.matmul(x, W1.T) + b1)    # z1: (1 x h) vector
            a1 = relu(z1)                               # a1: (1 x h) vector
            z2 = np.squeeze(np.matmul(a1, W2.T) + b2)   # z2: (1 x n) vector, this is the output (y^ / y_pred)
            return z1, a1, z2

        def calculate_loss(y_true, y_pred):
            # Return MSE between the 2 vectors
            return np.mean((y_true - y_pred) ** 2)

        def backward(x, W1, z1, a1, W2, z2, y_true):
            # Part 1: Calculate gradient descent for all parameters
            n = y_true.shape[0]
            # Note: matrix multiplication can be done with np.matmul OR `@` operator
            # 1. dL/dz2: (1 x n) vector, same shape as z2
            dz2 = 2 * (z2 - y_true) / n         
            # 2. dL/dW2: (n x h) matrix, same shape as W2
            dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)
            # Note: np.matmul(dz2.T, a1) doesn't work since they're both vectors, so NumPy just finds the dot product instead of the matrix multiplication outer product
            # 3. dL/db2: (1 x n) vector, same shape as b2
            db2 = dz2
            # 4a: dL/da1: (1 x h) vector, same shape as a1
            da1 = np.matmul(dz2, W2)
            # Note: ⊙ means element-wise multiplication, [c = a * b] => [c_i = a_i * b_i]
            mask = z1 > 0       # create the ReLU mask, this is a z1-shaped numpy array/vector
            # 4b: dL/dz1: (1 x h) vector, same size as z1
            dz1 = da1 * mask
            # 5. dL/dW1: (h x m) matrix, same size as W1
            dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
            # 6. dL/db1: (1 x m) vector, same size as z1
            db1 = dz1
            
            # Part 2: Update each parameter once (isn't actually done in this problem)
            # Do nothing here for now

            return dW1, db1, dW2, db2
        
        # Convert all arrays into NumPy arrays
        x = np.array(x, dtype=np.float64)
        W1 = np.array(W1, dtype=np.float64)
        W2 = np.array(W2, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)

        z1, a1, z2 = forward(x, W1, W2)                                 # execute 1 forward pass
        loss = calculate_loss(y_true, z2)                               # calculate loss
        dW1, db1, dW2, db2 = backward(x, W1, z1, a1, W2, z2, y_true)    # execute 1 backward pass
        return {
            "dW1": np.round(dW1, 4),
            "dW2": np.round(dW2, 4),
            "db1": np.round(db1, 4),
            "db2": np.round(db2, 4),
            "loss": np.round(loss, 4)
        }


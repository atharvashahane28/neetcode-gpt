import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        def forward(X, W, b, y):
            # W: (n_features x 1) matrix, Numpy array
            z = X @ W + b
            z = np.squeeze(z)       # This is important!!! It converts dimensions from (3, 1) to (3,)
            # single-neuron linear regression, so no non-linear activation function
            # z is y_pred, shape: (n_samples x 1)
            mse_loss = np.mean((z - y) ** 2)
            return z, mse_loss
        
        def calculate_gradients(X, y, z, mse_loss):
            n = X.shape[0]
            dW = 2 / n * X.T @ (z - y)
            db = 2 / n * np.sum(z - y)
            return dW, db
        
        def update_weights(W, b, dW, db, lr):
            W = W - lr * dW
            b = b - lr * db
            return W, b
        
        def loop(X, y, W, b, lr):
            z, mse_loss = forward(X, W, b, y)
            dW, db = calculate_gradients(X, y, z, mse_loss)
            W, b = update_weights(W, b, dW, db, lr)
            return W, b
        
        # Initialize weights and biases
        n_samples, n_features = X.shape
        W = np.zeros((n_features, ))    # Should be a (n_features x 1) matrix, but I created a n_features-sized vector instead to avoid extra dimention which causes issues later
        b = np.float64(0)
        for _ in range(epochs):
            W, b = loop(X, y, W, b, lr)
        W, b = np.round(W, 5), np.round(b, 5)
        return W, b


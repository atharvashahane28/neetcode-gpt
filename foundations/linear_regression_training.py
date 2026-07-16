import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64]) -> float:
        # note that N is just len(X)
        return -2 * np.matmul(ground_truth - model_prediction, X) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights = initial_weights
        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X=X, weights=weights)
            # for weight_idx in range(len(weights)):
            #     gradient = self.get_derivative(
            #         model_prediction=predictions,
            #         ground_truth=Y,
            #         N=X.shape[0],
            #         X=X,
            #         desired_weight=weight_idx
            #     )
            gradient = self.get_derivative(
                model_prediction=predictions,
                ground_truth=Y,
                N=X.shape[0],
                X=X
            )
            weights -= self.learning_rate * gradient
        return np.round(weights, 5)
            

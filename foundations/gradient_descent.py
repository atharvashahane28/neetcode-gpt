class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        func = lambda x : x ** 2
        deriv = lambda x : 2 * x
        x_old, x_new = init, init
        for _ in range(iterations):
            x_new = x_old - learning_rate * deriv(x_old)
            x_old = x_new
        return round(x_new, 5)
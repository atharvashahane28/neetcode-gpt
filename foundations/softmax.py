import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = np.max(z)           # find maximum logit
        z = z - max_z               # subtract maximum logit from all logits
        exp_z = np.e ** z           # find e ^ (each logit)
        exp_z_sum = np.sum(exp_z)   # add all values from exp_z (denominator in softmax formula)
        softmax = np.round(exp_z / exp_z_sum, 4)    # calculate softmax and round to 4 dec places
        return softmax
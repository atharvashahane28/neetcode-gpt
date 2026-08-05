import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pos = np.arange(seq_len).reshape(-1, 1)     # size: (seq_len, 1)
        index = np.arange(0, d_model, 2)            # size: (1, d_model // 2)
        angles = pos / (10000 ** (index / d_model)) # size: (seq_len, d_model // 2) after broadcasting
        encoding = np.zeros((seq_len, d_model))     # size: (seq_len, d_model)
        encoding[:, 0::2] = np.sin(angles)  # works because left and right operands are of the same shape
        encoding[:, 1::2] = np.cos(angles)  # ''
        return np.round(encoding, 5)
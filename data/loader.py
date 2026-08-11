import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        
        torch.manual_seed(0)
        start_indexes = torch.randint(low=0, high=data.shape[0] - context_length, size=(batch_size, ))
        X, Y = [], []
        for idx in start_indexes:
            X += [data[idx : idx + context_length]]
            Y += [data[idx + 1 : idx + 1 + context_length]]
        X = torch.stack(X)
        Y = torch.stack(Y)
        return [X, Y]

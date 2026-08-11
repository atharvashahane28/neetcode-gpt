import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # Sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss = 0
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            start_indexes = torch.randint(low=0, high=data.shape[0] - context_length, size=(batch_size, ))
            X, Y = [], []
            for start in start_indexes:
                X.append(data[start : start + context_length])
                Y.append(data[start + 1 : start + context_length + 1])
            X, Y = torch.stack(X), torch.stack(Y)
            logits = model(X)
            batch_size, context_length, vocab_size = logits.shape
            logits = logits.reshape(batch_size * context_length, vocab_size)
            Y = Y.reshape(batch_size * context_length)
            loss = F.cross_entropy(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.item()
        return round(loss, 4)
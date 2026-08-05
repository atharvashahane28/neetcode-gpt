import math
import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_transformation = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_transformation = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_transformation = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        # NOTE: WATCH THE 3BLUE1BROWN VIDEO BEFORE DOING THIS!!

        # Step 1: Project input through K, Q, and V linear transformation layers
        K = self.key_transformation(embedded)
        Q = self.query_transformation(embedded)
        V = self.value_transformation(embedded)
        batch_size, context_length, attention_dim = K.shape
        # K, Q, V has the same shape: (batch_size, context_length, attention_dim)

        # Step 2: Find initial attention scores, i.e., dot-product weights of query and key vectors.
        # .transpose(-1, -2) swaps the last 2 dimensions. 
        # You can't just do K.T because K is a 3D tensor (batch_size, context_length, attention_dim)
        # So, Q is (batch_size, context_length, attention_dim) and K-transpose is (batch_size, attention_dim, context_length)
        # Then, Q @ K-transpose will perform mat-mul over the last 2 dims while iterating over batch_size, their mat-mul will give (batch_size, context_length, context_length)
        attentions = Q @ K.transpose(-1, -2) / math.sqrt(attention_dim)
        
        # Step 3: Zero out all "future" values
        # The zeroing out part is different from the 3Blue1Brown video.
        # Here the upper triangle is 0, so y-axis tokens attend to x-axis tokens
        # create matrix of shape (context_length, context_length) where all upper triangular values are 0
        mask = torch.tril(torch.ones(context_length, context_length))
        # For all values in attentions where the corresponding value in mask is 0, replace with float("-inf")
        scores = attentions.masked_fill(mask == 0, float("-inf"))
        
        # Step 4: Softmax with dim=2
        scores = self.softmax(scores)   # dim=2 already applied during initialization

        # Step 5: Final mat-mul with V
        scores = scores @ V
        
        # Step 6: Round and return
        return torch.round(scores, decimals=4)




import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        batch_size, context_length, x_dim = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)

        Q = self.q_proj(x).reshape(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(batch_size, context_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(batch_size, context_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        model_dim = Q.shape[-1]
        K = torch.repeat_interleave(K, repeats=self.num_heads // self.num_kv_heads, dim=1)
        V = torch.repeat_interleave(V, repeats=self.num_heads // self.num_kv_heads, dim=1)
        attentions = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones((context_length, context_length), device=x.device))
        attentions = attentions.masked_fill(mask == 0, float("-inf"))
        attentions = nn.functional.softmax(attentions, dim=-1)
        additions = attentions @ V
        additions = additions.transpose(1, 2).reshape(batch_size, context_length, -1)
        output = self.output_proj(additions)
        return torch.round(output, decimals=4)

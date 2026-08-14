import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None and self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat((self.cache_k, new_k), dim=1)
            self.cache_v = torch.cat((self.cache_v, new_v), dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Return (rounded output, kv_cache)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if not kv_cache:
            kv_cache = KVCache()
        full_k, full_v = kv_cache.update(new_k=k, new_v=v)
        # At this point,
        # q: (batch_size, new_seq_len, model_dim),
        # full_k: (batch_size, total_seq_len, model_dim),
        # full_v: (batch_size, total_seq_len, model_dim)
        # To calculate the attention/weight matrix, we need to dot-prod every q with every full_k
        # NOTE: Don't forget the `/ math.sqrt(full_k.shape[-1])` term! This is part of the attentions formula present in Neetcode but not in 3Blue1Brown
        # attentions: (batch_size, new_seq_len, total_seq_len)
        # Then, to find the vectors to be added to x, we need to mat-mul the attention matrix with full_v
        # Mistake in the question - this should have been masked but isn't
        attentions = nn.functional.softmax(q @ full_k.transpose(-1, -2) / math.sqrt(full_k.shape[-1]), dim=2)
        additions = attentions @ full_v
        return torch.round(additions, decimals=4), kv_cache


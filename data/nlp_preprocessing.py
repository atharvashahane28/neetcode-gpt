import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        words = []
        for sentence in positive + negative:
            words += sentence.split()
        words = list(set(words))
        words = sorted(words)
        ids = [i + 1 for i in range(len(words))]
        indexMap = {}
        for word, wordId in zip(words, ids):
            indexMap[word] = wordId
        res = []
        for sentence in positive + negative:
            curr = []
            for word in sentence.split():
                curr.append(indexMap[word])
            res.append(torch.tensor(curr))
        res = nn.utils.rnn.pad_sequence(res, batch_first=True)
        return res

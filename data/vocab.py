from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        characters = sorted(list(set(text)))
        stoi = {}
        itos = {}
        i = 0
        for ch in characters:
            stoi[ch] = i
            itos[i] = ch
            i += 1
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        characters = list(text)
        res = [stoi[ch] for ch in characters]
        return res

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        res = ""
        for i in ids:
            res += itos[i]
        return res

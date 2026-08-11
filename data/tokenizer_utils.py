from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.

        res = []
        for num in numbers:
            res.append(self.greedy_tokenizer(text=str(num), vocab=vocab))
        return res


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.

        return len(self.greedy_tokenizer(text=text, vocab=vocab))


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.

        return round(len(self.greedy_tokenizer(text=text, vocab=vocab)) / len(text.split()), 4)


    def greedy_tokenizer(self, text: str, vocab: Dict[str, int]) -> List[str]:
        # Takes in a string of text and converts it into a list of tokens

        vocab_prefixes = set()
        for token in vocab.keys():
            # Split the vocab keys into all possible prefixes
            # For example - ["225", "71"] -> set{"2", "22", "25", "7", "71"}
            # This makes searching for these prefixes O(1) time
            prev = ""
            for idx in range(len(token)):
                vocab_prefixes.add(prev + token[idx])
                prev += token[idx]
        tokens = []
        prev = ""
        for idx in range(len(text)):
            if prev + text[idx] in vocab_prefixes:
                # keep adding to the substring greedily
                    prev += text[idx]
            else:
                if prev in vocab:
                    # add prev if it is a valid vocabulary key
                    tokens.append(prev)
                else:
                    # if not, add all individual characters within prev
                    prevCharacters = list(prev)
                    tokens.extend(prevCharacters)
                prev = text[idx]
        tokens.append(prev)     # reset prev to ""
        return tokens








from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        characters = [ch for ch in corpus if ch != " "]
        freqMap = {}
        merges = []
        for _ in range(num_merges):
            maxFreq = 0
            maxPair = ("", "")
            for i in range(len(characters) - 1):
                left, right = characters[i], characters[i + 1]
                currFreq = freqMap.get((left, right), 0) + 1
                if currFreq > maxFreq or currFreq == maxFreq and left + right < maxPair[0] + maxPair[1]:
                    # greater frequency or equal frequency and lexicographically smaller pair
                    maxPair = (left, right)
                    maxFreq = currFreq
                freqMap[(left, right)] = currFreq
            nextCharacters = []
            i = 0
            while i < len(characters) - 1:
                if characters[i] == maxPair[0] and characters[i + 1] == maxPair[1]:
                    # Merge the characters
                    nextCharacters.append(characters[i] + characters[i + 1])
                    i += 2
                else:
                    # Just add the left character
                    nextCharacters.append(characters[i])
                    i += 1
            characters = nextCharacters
            merges.append([maxPair[0], maxPair[1]])
        return merges
                    



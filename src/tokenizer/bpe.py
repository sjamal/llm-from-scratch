import re
from typing import Dict, List, Tuple

class BPETokenizer:
    """Byte Pair Encoding (BPE) Subword Tokenizer built from scratch.
    
    Learns frequent character pairs from a training corpus and merges them
    iteratively to construct a subword vocabulary.
    """

    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: List[str] = []

    def _get_stats(self, words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Count frequencies of adjacent symbol pairs."""
        pairs: Dict[Tuple[str, str], int] = {}
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Merge all occurrences of a specific pair in a single word tuple."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train(self, text: str) -> None:
        """Train the BPE tokenizer on a raw text corpus until vocab_size is reached."""
        # Clean text and split into words with an end-of-word indicator ('</w>')
        raw_words = re.findall(r"\w+|\S", text.lower())
        word_freqs: Dict[Tuple[str, ...], int] = {}
        
        for w in raw_words:
            char_tuple = tuple(list(w) + ["</w>"])
            word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1

        # Determine how many merge steps to run
        num_merges = self.vocab_size - len(set("".join(text.lower())))
        
        for _ in range(max(0, num_merges)):
            stats = self._get_stats(word_freqs)
            if not stats:
                break
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break

            # Store the merge rule
            merged_symbol = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_symbol
            
            # Update word dictionary with merged pairs
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_word(word, best_pair)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

        # Reconstruct final vocabulary list
        vocab_set = set()
        for word in word_freqs.keys():
            for token in word:
                vocab_set.add(token)
        self.vocab = sorted(list(vocab_set))

    def encode(self, text: str) -> List[str]:
        """Encode input text into BPE subword tokens using learned merge rules."""
        raw_words = re.findall(r"\w+|\S", text.lower())
        encoded_tokens = []

        for w in raw_words:
            word = tuple(list(w) + ["</w>"])
            for pair, merged in self.merges.items():
                word = self._merge_word(word, pair)
            encoded_tokens.extend(word)

        return encoded_tokens

    def decode(self, tokens: List[str]) -> str:
        """Decode a list of BPE subword tokens back into readable text."""
        raw_text = "".join(tokens).replace("</w>", " ")
        return raw_text.strip()


if __name__ == "__main__":
    corpus = "hug hugger hugging hugs hug hugger"
    tokenizer = BPETokenizer(vocab_size=12)
    tokenizer.train(corpus)
    
    sample_text = "hugging hugs"
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Corpus: '{corpus}'")
    print(f"Encoded '{sample_text}': {tokens}")
    print(f"Decoded back: '{decoded}'")

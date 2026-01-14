import numpy as np

class Tokenizer:
    def __init__(self, chars, max_text_length):
        self.chars = sorted(set(chars))
        self.max_text_length = max_text_length

        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars) + 1

    def encode(self, text):
        x = np.zeros(self.max_text_length, dtype=np.int32)
        for i, ch in enumerate(text[:self.max_text_length]):
            x[i] = self.char2idx.get(ch, 0)
        return x

    def decode(self, seq):
        return "".join(self.idx2char.get(i, "") for i in seq if i != 0)
import h5py
import numpy as np
import random
import math
from .tokenizer import Tokenizer


class DataGenerator:
    def __init__(self, source, batch_size, charset, max_text_length, predict=False):
        self.source = source
        self.batch_size = batch_size
        self.charset = charset
        self.max_text_length = max_text_length
        self.predict = predict

        self.tokenizer = Tokenizer(chars=charset, max_text_length=max_text_length)
        self.hf = h5py.File(self.source, "r")

        self.size = {s: len(self.hf[s]["dt"]) for s in ["train", "valid", "test"]}
        self.steps = {s: math.ceil(self.size[s] / self.batch_size) for s in self.size}

    def _batch(self, split):
        idxs = list(range(self.size[split]))
        random.shuffle(idxs)

        for i in range(0, self.size[split], self.batch_size):
            batch = idxs[i:i + self.batch_size]
            batch_sorted = sorted(batch)

            X = self.hf[split]["dt"][batch_sorted]
            if len(X) == 0:
                continue

            Y = [self.tokenizer.encode(t.decode()) for t in self.hf[split]["gt"][batch_sorted]]


            yield np.array(X), np.array(Y)

    def next_train_batch(self):
        while True:
            for x, y in self._batch("train"):
                yield x, y

    def next_valid_batch(self):
        while True:
            for x, y in self._batch("valid"):
                yield x, y

    def next_test_batch(self):
        while True:
            for x, y in self._batch("test"):
                yield x, y

import re
from collections import Counter

import math


class BM25:

    SITE = re.compile(r"\(?-?\bsite:\S+", re.IGNORECASE)

    def __init__(self, k1: float, b: float):
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold())

    @staticmethod
    def _idf(n_docs: int, n_docs_with: int) -> float:
        return math.log(1 + (n_docs - n_docs_with + 0.5) / (n_docs_with + 0.5))

    def scores(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        tfs = [Counter(self._tokenize(doc)) for doc in docs]
        lens = [sum(tf.values()) for tf in tfs]
        n_docs = len(docs)
        avg = sum(lens) / n_docs
        if avg == 0:
            avg = 1

        df = Counter()
        for tf in tfs:
            df.update(tf.keys())

        query = self.SITE.sub(" ", query)
        words = set(self._tokenize(query))

        scores = [0.0] * n_docs
        for word in words:
            if df[word] == 0:
                continue
            w = self._idf(n_docs, df[word])

            for i in range(n_docs):
                f = tfs[i][word]
                if f == 0:
                    continue
                norm = 1 - self.b + self.b * lens[i] / avg
                scores[i] += w * f * (self.k1 + 1) / (f + self.k1 * norm)

        return scores





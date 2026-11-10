import logging
import math

from models.schema import SearchQuery
from rank.bm25 import BM25
from rank.cut import TextChunker
from rank.rerank import CrossEncoderReranker
from search.schema import Candidate


class Ranker:

    def __init__(
        self,
        k1: float,
        b: float,
        max_chunk_size: int,
        chunk_overlap: int,
        alpha: float,
        model_name: str,
        model_cache_dir: str,
        rerank_per_page: int,
    ):
        self.bm25 = BM25(k1, b)
        self.chunker = TextChunker(max_chunk_size, chunk_overlap)
        self.alpha = alpha
        try:
            self.reranker = CrossEncoderReranker(model_name, model_cache_dir)
        except Exception as e:
            logging.warning(f"Unable to load reranker model: {e}")
            self.reranker = None
        self.rerank_per_page = rerank_per_page

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def rank_candidates(self, candidates: list[Candidate], q: SearchQuery) -> list[Candidate]:

        chunks = []
        owner = []
        if not candidates:
            return []
        for i, c in enumerate(candidates):
            for ch in self.chunker.chunk_text(c.text or ""):
                chunks.append(ch)
                owner.append(i)

        scores = self.bm25.scores(q.query, chunks)
        k = q.chunks_per_source if q.search_depth == "advanced" else 1
        best_scores = [0.0] * len(candidates)
        used_ce = False
        if q.search_depth == "advanced" and self.reranker:
            used_ce = True
            q_clean = BM25.SITE.sub(" ", q.query).strip()
            n = max(self.rerank_per_page, q.chunks_per_source)
            for i, c in enumerate(candidates):
                idxs = [idx for idx in range(len(owner)) if owner[idx] == i]
                if not idxs:
                    c.content = c.snippet or ""
                    continue
                top_chunk_idxs = sorted(idxs, key=lambda x: scores[x], reverse=True)[:n]
                rerank_chunks = [chunks[j] for j in top_chunk_idxs]
                rerank_scores = self.reranker.scores(q_clean, rerank_chunks)
                best = max(rerank_scores)
                if best > 0:
                    ce = dict(zip(top_chunk_idxs, rerank_scores))  # chunk position → ce score
                    chosen = sorted(top_chunk_idxs, key=lambda j: ce[j], reverse=True)[:k]
                    chosen.sort()
                    c.content = " [...] ".join(chunks[j] for j in chosen)
                else:
                    c.content = c.snippet or (chunks[idxs[0]] if idxs else "")
                best_scores[i] = self.sigmoid(best)

        else:
            for i, c in enumerate(candidates):
                idxs = [idx for idx in range(len(owner)) if owner[idx] == i]
                best = max((scores[j] for j in idxs), default=0.0)
                if best > 0:
                    top = sorted(idxs, key=lambda x: scores[x], reverse=True)[:k]
                    top.sort()
                    c.content = " [...] ".join(chunks[j] for j in top)
                else:
                    c.content = c.snippet or (chunks[idxs[0]] if idxs else "")

                best_scores[i] = best

        top_best = max(best_scores)
        priors = [c.prior_score for c in candidates]
        hi, lo = max(priors), min(priors)
        for i, c in enumerate(candidates):
            bm25_part = best_scores[i] if used_ce else (best_scores[i] / top_best if top_best > 0 else 0.0)
            prior_part = (priors[i] - lo) / (hi - lo) if hi > lo else 1.0
            c.score = self.alpha * bm25_part + (1 - self.alpha) * prior_part

        return sorted(candidates, key=lambda c: (c.score, c.prior_score), reverse=True)


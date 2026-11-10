from fastembed.rerank.cross_encoder import TextCrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: str):
        self.model = TextCrossEncoder(model_name, cache_dir=cache_dir)

    def scores(self, query: str, docs: list[str]) -> list[float]:
        return list(self.model.rerank(query, docs))
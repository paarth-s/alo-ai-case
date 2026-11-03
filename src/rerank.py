"""Cross-encoder reranker: reads query and passage together, so it can separate near-duplicate chunks
that a bi-encoder (which embeds them independently) scores almost identically."""
from dataclasses import replace

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .store import Hit

MODELS = {
    "minilm": "Xenova/ms-marco-MiniLM-L-6-v2",  # 22M params, trained on MS MARCO web passages
    "bge-base": "BAAI/bge-reranker-base",  # 278M params, multilingual
}


class Reranker:
    def __init__(self, model: str = "minilm"):
        repo = MODELS[model]
        self.session = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding()

    def rerank(self, query: str, hits: list[Hit], batch_size: int = 16) -> list[Hit]:
        """Return hits re-sorted by relevance logit (higher is better); the score field is replaced."""
        scores = []
        for i in range(0, len(hits), batch_size):
            enc = self.tokenizer.encode_batch([(query, h.text) for h in hits[i:i + batch_size]])
            feed = {"input_ids": np.array([e.ids for e in enc]), "attention_mask": np.array([e.attention_mask for e in enc])}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc])
            scores += self.session.run(None, feed)[0].reshape(len(enc), -1)[:, 0].tolist()
        return sorted((replace(h, score=s) for h, s in zip(hits, scores)), key=lambda h: -h.score)

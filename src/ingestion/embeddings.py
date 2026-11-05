"""Local embeddings via onnxruntime (no PyTorch, no API calls, CPU only)."""
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

NAME = "bge-small"  # the one embedder: BAAI/bge-small-en-v1.5, 33M params, 384 dims, CLS pooling (see ADR Q2 for the ones rejected)
REPO = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE matches short queries to long passages with this
MAX_TOKENS = 512
INDEX_DIR = "index"


class Embedder:
    def __init__(self):
        self.name, self.max_tokens = NAME, MAX_TOKENS
        self.session = ort.InferenceSession(hf_hub_download(REPO, "onnx/model.onnx"), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()
        self._counter = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))  # no truncation: lengths stay honest
        self.dim = self._embed(["probe"]).shape[1]

    def n_tokens(self, text: str) -> int:
        """Tokens the model would need for this passage, including special tokens."""
        return len(self._counter.encode(text).ids)

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        mask = np.array([e.attention_mask for e in enc])
        feed = {"input_ids": np.array([e.ids for e in enc]), "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        out = self.session.run(None, feed)[0]
        pooled = out[:, 0]  # CLS pooling
        return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype("float32")

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), "float32")
        return np.vstack([self._embed(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + query])

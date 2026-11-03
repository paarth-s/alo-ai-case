"""Local embeddings via onnxruntime (no PyTorch, no API calls, CPU only). Models differ in pooling and
in the prefixes they were trained with, so each is described by a Spec."""
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DEFAULT = "bge-small"
BGE_QUERY = "Represent this sentence for searching relevant passages: "  # BGE matches short queries to long passages with this


@dataclass(frozen=True)
class Spec:
    repo: str
    onnx: str = "onnx/model.onnx"
    pooling: str = "cls"  # "cls" or "mean"
    query_prefix: str = ""
    passage_prefix: str = ""


SPECS = {
    "bge-small": Spec("BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY),  # 33M params, 384 dims
    "bge-base": Spec("BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY),  # 110M params, 768 dims
    "gte-small": Spec("thenlper/gte-small", pooling="mean"),  # 33M params, 384 dims
    "minilm": Spec("sentence-transformers/all-MiniLM-L6-v2", pooling="mean"),  # 22M params, 384 dims
    "e5-small": Spec("intfloat/e5-small-v2", onnx="model.onnx", pooling="mean",
                     query_prefix="query: ", passage_prefix="passage: "),  # 33M params, 384 dims
}


def index_dir_for(name: str) -> str:
    return "index" if name == DEFAULT else f"index_{name}"


class Embedder:
    def __init__(self, name: str = DEFAULT):
        self.name, spec = name, SPECS[name]
        self.spec = spec
        self.session = ort.InferenceSession(hf_hub_download(spec.repo, spec.onnx), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(hf_hub_download(spec.repo, "tokenizer.json"))
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding()
        self.max_tokens = 512
        self._counter = Tokenizer.from_file(hf_hub_download(spec.repo, "tokenizer.json"))  # no truncation: lengths stay honest
        self.dim = self._embed(["probe"]).shape[1]

    def n_tokens(self, text: str) -> int:
        """Tokens the model would need for this passage, including special tokens and any passage prefix."""
        return len(self._counter.encode(self.spec.passage_prefix + text).ids)

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        mask = np.array([e.attention_mask for e in enc])
        feed = {"input_ids": np.array([e.ids for e in enc]), "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        out = self.session.run(None, feed)[0]
        pooled = out[:, 0] if self.spec.pooling == "cls" else (out * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype("float32")

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), "float32")
        texts = [self.spec.passage_prefix + t for t in texts]
        return np.vstack([self._embed(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([self.spec.query_prefix + query])

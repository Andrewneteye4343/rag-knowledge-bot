"""自訂 Embeddings：把 fastembed 包成 LangChain 的 Embeddings 介面。

教學重點：LangChain 允許你無痛接入自己的元件——只要實作
embed_documents / embed_query 兩個方法即可。
"""
import warnings

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

# 濾掉 fastembed 的 mean pooling 無害警告，保持輸出乾淨（與主專案一致）
warnings.filterwarnings("ignore", message=".*mean pooling.*")


class FastEmbedLocal(Embeddings):
    def __init__(self, model_name: str, cache_dir: str):
        # 與主專案相同的 fastembed（ONNX、CPU、快取可共用）
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # e5 系列需要 query 前綴，query_embed 會自動處理
        return list(self._model.query_embed([text]))[0].tolist()

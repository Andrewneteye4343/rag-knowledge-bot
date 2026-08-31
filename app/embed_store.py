"""向量資料庫：fastembed 產生向量 + ChromaDB 儲存與檢索。"""
import logging
import os
import warnings
from pathlib import Path

# 必須在 import chromadb 之前設定，才能關閉其匿名統計回報（避免 telemetry 雜訊）
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

import config

# 濾掉 fastembed 的 mean pooling 無害警告，保持輸出乾淨
warnings.filterwarnings("ignore", message=".*mean pooling.*")

# chromadb 0.6.x 的 telemetry 回報有 bug（忽略 anonymized_telemetry 設定），
# 直接停用該 logger，避免每次執行都出現 "Failed to send telemetry" 雜訊
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


def _clean_text(s: str) -> str:
    """消毒文字：確保為 str、移除 NUL 與無效代理字元，避免 tokenizer 崩潰。

    主要是防 Windows 終端機送中文進 Docker TTY 時夾帶的控制字元。
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "").replace("\r", " ")
    return s.encode("utf-8", errors="ignore").decode("utf-8")


class VectorStore:
    def __init__(self, embedding_model: str, chroma_dir: Path, collection: str):
        # 第一次執行會自動下載 embedding 模型到 CACHE_DIR（掛載的持久化目錄，
        # 避免每次 docker compose run 都重新下載）；之後每次執行仍需載入模型
        print("⏳ 載入 embedding 模型與向量庫中（模型越大越久，約 5~15 秒）...")
        self.embedder = TextEmbedding(
            model_name=embedding_model,
            cache_dir=str(config.CACHE_DIR),
        )
        # 關閉 ChromaDB 匿名統計回報（避免輸出 telemetry 雜訊）
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """將區塊向量化後寫入（id 相同會覆蓋，可重複執行）。"""
        if not chunks:
            return 0
        texts = [_clean_text(c["text"]) for c in chunks]
        embeddings = [e.tolist() for e in self.embedder.embed(texts)]
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks
            ],
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, question: str, top_k: int) -> list[dict]:
        """語意檢索，回傳 [{text, source, chunk_index, distance}]。"""
        # e5 系列模型需要 query 前綴，fastembed 的 query_embed 會自動處理；
        # 其他模型（如 bge）query_embed 與 embed 行為相同
        question = _clean_text(question)
        emb = list(self.embedder.query_embed([question]))[0]
        res = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            items.append({
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": round(dist, 4),
            })
        return items

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )

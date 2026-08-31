"""向量資料庫：fastembed 產生向量 + ChromaDB 儲存與檢索。"""
from pathlib import Path

import chromadb
from fastembed import TextEmbedding


class VectorStore:
    def __init__(self, embedding_model: str, chroma_dir: Path, collection: str):
        # 第一次執行會自動下載 embedding 模型（約 100MB）
        self.embedder = TextEmbedding(model_name=embedding_model)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """將區塊向量化後寫入（id 相同會覆蓋，可重複執行）。"""
        if not chunks:
            return 0
        embeddings = [e.tolist() for e in self.embedder.embed([c["text"] for c in chunks])]
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
        emb = list(self.embedder.embed([question]))[0]
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

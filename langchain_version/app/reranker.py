"""二階段檢索的自訂元件：reranker compressor + 組合式 retriever。

背景：LangChain 1.x 已移除 ContextualCompressionRetriever / BaseDocumentCompressor，
因此我們用框架現存的 BaseRetriever 抽象自行組合——這同時示範了
「框架版本演進時，自訂元件如何讓程式不受影響」。
"""
import warnings
from typing import Sequence

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# 濾掉 fastembed 的 mean pooling 無害警告
warnings.filterwarnings("ignore", message=".*mean pooling.*")


class FastEmbedReranker:
    """以 cross-encoder 對「問題＋候選區塊」重新打分排序（compressor 角色）。"""

    def __init__(self, model_name: str, cache_dir: str):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None  # 延遲載入：沒用到不拖慢速度

    def _ensure_model(self) -> TextCrossEncoder:
        if self._model is None:
            print(
                f"⏳ 載入 reranker 模型（{self._model_name}，首次使用需下載約 1.1GB）..."
            )
            self._model = TextCrossEncoder(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
            )
        return self._model

    def compress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        """精排：依 rerank 分數遞減排序（不截斷，由 retriever 決定數量）。"""
        if not documents:
            return []
        model = self._ensure_model()
        texts = [d.page_content for d in documents]
        scores = list(model.rerank(query, texts))
        for doc, s in zip(documents, scores):
            doc.metadata["rerank_score"] = round(float(s), 4)
        return sorted(
            documents,
            key=lambda d: d.metadata["rerank_score"],
            reverse=True,
        )


class RerankRetriever(BaseRetriever):
    """二階段檢索器：base retriever 海選 → reranker 精排 → 取 top_k。"""

    base_retriever: BaseRetriever
    reranker: FastEmbedReranker
    top_k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        ranked = self.reranker.compress_documents(candidates, query)
        return list(ranked[: self.top_k])

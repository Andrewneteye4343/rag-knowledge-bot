"""RAG 主流程：檢索（Retrieval）+ 生成（Generation）。"""
import config
import llm
from embed_store import VectorStore


def ask(question: str) -> tuple[str, list[dict]]:
    store = VectorStore(config.EMBEDDING_MODEL, config.CHROMA_DIR, config.CHROMA_COLLECTION)
    if store.count() == 0:
        raise SystemExit("知識庫是空的！請先執行: python main.py ingest")

    chunks = store.query(question, config.TOP_K)
    answer = llm.generate(question, chunks)
    return answer, chunks

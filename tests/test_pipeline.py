"""端對端冒煙測試：使用 mock LLM，不呼叫外部 API。

執行方式（在專案根目錄）：
    LLM_BACKEND=mock python -m pytest tests/test_pipeline.py -v
或:
    LLM_BACKEND=mock python tests/test_pipeline.py
"""
import os
import sys
import tempfile
from pathlib import Path

# 讓測試可直接執行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

# 隔離測試資料目錄
_TMP = tempfile.mkdtemp(prefix="rag_test_")
os.environ.setdefault("CHROMA_DIR", str(Path(_TMP) / "chroma"))
os.environ.setdefault("LLM_BACKEND", "mock")

import config  # noqa: E402
import rag  # noqa: E402
from embed_store import VectorStore  # noqa: E402
from loader import load_all, make_chunks  # noqa: E402


def _build_index() -> int:
    docs = load_all(config.KNOWLEDGE_DIR)
    chunks = make_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    store = VectorStore(config.EMBEDDING_MODEL, config.CHROMA_DIR, config.CHROMA_COLLECTION)
    store.reset()
    return store.add_chunks(chunks)


def test_ingest_and_ask():
    n = _build_index()
    assert n > 0, "索引應寫入至少一個區塊"

    store = VectorStore(config.EMBEDDING_MODEL, config.CHROMA_DIR, config.CHROMA_COLLECTION)
    assert store.count() == n

    answer, chunks = rag.ask("員工的特休假有幾天？")
    assert answer, "應有回答"
    assert len(chunks) > 0, "應檢索到區塊"
    # 檢索結果應包含人事制度文件
    sources = {c["source"] for c in chunks}
    assert any("人事制度" in s for s in sources), f"檢索結果應含人事制度，實際: {sources}"
    print("✓ 索引與檢索正常")
    print("  Mock 回答:", answer.splitlines()[0])
    print("  檢索來源:", sources)


if __name__ == "__main__":
    test_ingest_and_ask()
    print("\n全部測試通過 ✅")

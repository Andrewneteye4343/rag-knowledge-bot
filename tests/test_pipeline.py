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
# 基礎測試停用 rerank（RERANK_TOP_K <= TOP_K）；rerank 路徑由獨立測試覆蓋
os.environ.setdefault("RERANK_TOP_K", "4")
os.environ.setdefault("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

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

    answer, chunks, confidence = rag.ask("員工的特休假有幾天？")
    assert answer, "應有回答"
    assert len(chunks) > 0, "應檢索到區塊"
    # 檢索結果應包含人事制度文件
    sources = {c["source"] for c in chunks}
    assert any("人事制度" in s for s in sources), f"檢索結果應含人事制度，實際: {sources}"
    # 信心分數應在 0-100 之間且有標籤
    assert 0 <= confidence["score"] <= 100, f"信心分數越界: {confidence['score']}"
    assert confidence["label"] in {"高", "中", "低"}
    print("✓ 索引與檢索正常")
    print("  信心分數:", confidence["score"], "/ 100（", confidence["label"], "）")
    print("  Mock 回答:", answer.splitlines()[0])
    print("  檢索來源:", sources)


def test_rerank_path():
    """二階段檢索路徑：海選 RERANK_TOP_K 個 → rerank 精排回 TOP_K 個。"""
    # 開啟 rerank（8 個候選 → 精排回 4 個）
    config.RERANK_TOP_K = 8

    answer, chunks, confidence = rag.ask("員工的特休假有幾天？")
    assert 0 < len(chunks) <= 4, f"rerank 後應回傳 ≤4 個區塊，實際 {len(chunks)}"
    assert all("rerank_score" in c for c in chunks), "每個區塊都應有 rerank_score"
    # rerank 分數應遞減排序
    scores = [c["rerank_score"] for c in chunks]
    assert scores == sorted(scores, reverse=True), f"應依 rerank 分數遞減: {scores}"
    assert "Reranker" in confidence["composition"], "啟用 rerank 時組成應標註"
    print("✓ Reranker 二階段檢索正常")
    print("  rerank 分數:", scores)

    # 還原設定
    config.RERANK_TOP_K = 4


if __name__ == "__main__":
    test_ingest_and_ask()
    test_rerank_path()
    print("\n全部測試通過 ✅")

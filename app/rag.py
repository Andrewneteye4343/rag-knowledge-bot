"""RAG 主流程：檢索（Retrieval）+ 生成（Generation）+ 信心評估。"""
import config
import llm
from embed_store import VectorStore


def _retrieval_score(chunks: list[dict]) -> float:
    """檢索信心分數（0-100）。

    依據：top-1 餘弦距離越近越高分；top-1 與 top-2 的距離差距越大
    （第一名明顯領先）代表問題越「明確命中」，額外加分。
    """
    if not chunks:
        return 0.0
    top1 = chunks[0]["distance"]
    # 距離 0 → 100 分；距離 ≥ 0.5 → 0 分（0.5 為餘弦距離的經驗閾值）
    base = max(0.0, min(1.0, (0.5 - top1) / 0.5)) * 100
    if len(chunks) >= 2:
        gap = chunks[1]["distance"] - top1
        base = min(100.0, base + min(15.0, gap * 150))
    return round(base, 1)


def _confidence_label(score: int) -> str:
    if score >= 75:
        return "高"
    if score >= 50:
        return "中"
    return "低"


def ask(question: str) -> tuple[str, list[dict], dict]:
    """回傳 (answer, chunks, confidence)。

    confidence = {score, label, reason, retrieval, llm, composition}
    """
    store = VectorStore(config.EMBEDDING_MODEL, config.CHROMA_DIR, config.CHROMA_COLLECTION)
    if store.count() == 0:
        raise SystemExit("知識庫是空的！請先執行: python main.py ingest")

    chunks = store.query(question, config.TOP_K)
    result = llm.generate(question, chunks)

    retrieval_score = _retrieval_score(chunks)
    llm_score = result["score"]
    if llm_score is None:
        # 模型未照格式輸出自評 → 退回只用檢索評估
        final = round(retrieval_score)
        composition = "檢索評估（模型未提供自評）"
    else:
        final = round(0.4 * retrieval_score + 0.6 * llm_score)
        composition = "檢索 40% + 模型自評 60%"

    confidence = {
        "score": final,
        "label": _confidence_label(final),
        "reason": result["reason"],
        "retrieval": retrieval_score,
        "llm": llm_score,
        "composition": composition,
    }
    return result["answer"], chunks, confidence

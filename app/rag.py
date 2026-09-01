"""RAG 主流程：檢索（Retrieval）+ Rerank 精排 + 生成（Generation）+ 信心評估。"""
import config
import llm
from embed_store import VectorStore


def _retrieval_score(chunks: list[dict]) -> float:
    """檢索信心分數（0-100）。

    基礎：top-1 餘弦距離（越近越高分）＋ top-1 與 top-2 的距離間距加分。
    有 rerank 時：rerank 負責排序品質，其「第一名領先幅度」（margin）額外加分。
    注意：不直接用 rerank 原始分數換算——cross-encoder 的分數刻度因模型而異
    （可能全為負值），直接用會嚴重誤判，這是實務上常見的校準陷阱。
    """
    if not chunks:
        return 0.0
    top1 = chunks[0]["distance"]
    # 距離 0 → 100 分；距離 ≥ 0.5 → 0 分（0.5 為餘弦距離的經驗閾值）
    base = max(0.0, min(1.0, (0.5 - top1) / 0.5)) * 100
    if len(chunks) >= 2:
        gap = chunks[1]["distance"] - top1
        base = min(100.0, base + min(15.0, gap * 150))
    if "rerank_score" in chunks[0] and len(chunks) >= 2:
        # 第一名領先幅度越大越有信心（加分上限 15）
        margin = chunks[0]["rerank_score"] - chunks[1]["rerank_score"]
        base = min(100.0, base + max(0.0, min(15.0, margin * 20)))
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

    # 二階段檢索：RERANK_TOP_K 大於 TOP_K 時，先海選再精排
    # （候選數不超過索引總數，避免 ChromaDB 印警告）
    candidate_k = max(config.RERANK_TOP_K, config.TOP_K)
    n = min(candidate_k, store.count())
    chunks = store.query(question, n)
    rerank_used = config.RERANK_TOP_K > config.TOP_K and len(chunks) > config.TOP_K
    if rerank_used:
        chunks = store.rerank(question, chunks, config.TOP_K)

    result = llm.generate(question, chunks)

    retrieval_score = _retrieval_score(chunks)
    llm_score = result["score"]
    if llm_score is None:
        # 模型未照格式輸出自評 → 退回只用檢索評估
        final = round(retrieval_score)
        composition = "檢索評估（模型未提供自評）"
    else:
        final = round(0.4 * retrieval_score + 0.6 * llm_score)
        composition = (
            "檢索 40%（Reranker 精排） + 模型自評 60%"
            if rerank_used
            else "檢索 40% + 模型自評 60%"
        )

    confidence = {
        "score": final,
        "label": _confidence_label(final),
        "reason": result["reason"],
        "retrieval": retrieval_score,
        "llm": llm_score,
        "composition": composition,
    }
    return result["answer"], chunks, confidence

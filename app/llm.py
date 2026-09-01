"""LLM 介面：目前支援 Google Gemini，另有 mock 模式供測試。"""
import re

import config

SYSTEM_PROMPT = (
    "你是企業知識庫問答助手。請嚴格遵守以下規則：\n"
    "1. 只根據【上下文】提供的資料回答問題，不要使用上下文之外的知識。\n"
    "2. 若上下文中沒有答案，請直接說「根據知識庫無法回答此問題」，不要編造。\n"
    "3. 回答時請在相關句子的結尾附上引用來源，格式：[來源: 檔案名稱#區塊編號]。\n"
    "4. 使用繁體中文回答，條列清楚、精簡。\n"
    "5. 回答結束後，請在最末行輸出信心自評，格式為：\n"
    "   【信心分數】0-100 的整數\n"
    "   【理由】一句話說明評估依據（例如：所有數字都有來源引用、或部分資訊需推論）。\n"
    "   注意：若你只能回答「無法回答」，信心分數必須小於等於 20。"
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[區塊{i}] 檔案: {c['source']}#{c['chunk_index']}\n{c['text']}")
    return "\n\n".join(parts)


def _parse_confidence(text: str) -> tuple[str, int | None, str | None]:
    """從模型輸出解析【信心分數】與【理由】，並回傳純回答文字。"""
    score = None
    reason = None
    m = re.search(r"【信心分數】\s*(\d{1,3})", text)
    if m:
        score = min(100, int(m.group(1)))
    m2 = re.search(r"【理由】\s*(.+)", text, re.S)
    if m2:
        reason = m2.group(1).strip().split("\n")[0]
    # 移除格式標記行，保留純回答
    cleaned = re.sub(r"【信心分數】.*?(?=【理由】|$)", "", text, flags=re.S).strip()
    cleaned = re.sub(r"【理由】.*$", "", cleaned, flags=re.S).strip()
    return cleaned, score, reason


def generate(question: str, chunks: list[dict]) -> dict:
    """依檢索結果生成回答。回傳 {answer, score, reason}。

    chunks 為 [{text, source, chunk_index, distance}]。
    """
    if config.LLM_BACKEND == "mock":
        return _mock_answer(question, chunks)

    if not config.GEMINI_API_KEY:
        raise SystemExit(
            "尚未設定 GEMINI_API_KEY！\n"
            "請到 https://aistudio.google.com/apikey 申請金鑰，"
            "然後填入專案根目錄的 .env 檔案。"
        )

    from google import genai
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"【上下文】\n{build_context(chunks)}\n\n"
        f"【問題】\n{question}"
    )
    try:
        resp = client.models.generate_content(model=config.LLM_MODEL, contents=prompt)
    except genai_errors.APIError as e:
        raise SystemExit(
            f"❌ Gemini API 錯誤（HTTP {e.code}）：{e.message}\n"
            "請將此訊息貼給你的助教協助排除。"
        ) from e
    except Exception as e:
        raise SystemExit(
            f"❌ Gemini API 呼叫失敗：{type(e).__name__}: {e}"
        ) from e

    raw = resp.text or "(模型無回覆內容)"
    answer, score, reason = _parse_confidence(raw)
    return {"answer": answer, "score": score, "reason": reason}


def _mock_answer(question: str, chunks: list[dict]) -> dict:
    """測試用：不呼叫 API，回傳範本答案以驗證 RAG 流程。"""
    if chunks:
        # 依檢索距離給一個合理的 mock 分數
        top1 = chunks[0]["distance"]
        score = max(0, min(100, round(100 * (1 - top1))))
    else:
        score = 0
    lines = [f"（Mock 答案）針對問題「{question}」，檢索到 {len(chunks)} 個區塊："]
    for c in chunks:
        preview = c["text"][:60].replace("\n", " ")
        lines.append(f"- [來源: {c['source']}#{c['chunk_index']}] {preview}…")
    return {
        "answer": "\n".join(lines),
        "score": score,
        "reason": "Mock 模式：未實際呼叫模型，分數僅依檢索距離估算",
    }

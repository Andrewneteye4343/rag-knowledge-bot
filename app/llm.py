"""LLM 介面：目前支援 Google Gemini，另有 mock 模式供測試。"""
import config

SYSTEM_PROMPT = (
    "你是企業知識庫問答助手。請嚴格遵守以下規則：\n"
    "1. 只根據【上下文】提供的資料回答問題，不要使用上下文之外的知識。\n"
    "2. 若上下文中沒有答案，請直接說「根據知識庫無法回答此問題」，不要編造。\n"
    "3. 回答時請在相關句子的結尾附上引用來源，格式：[來源: 檔案名稱#區塊編號]。\n"
    "4. 使用繁體中文回答，條列清楚、精簡。"
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[區塊{i}] 檔案: {c['source']}#{c['chunk_index']}\n{c['text']}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[dict]) -> str:
    """依檢索結果生成回答。chunks 為 [{text, source, chunk_index, distance}]。"""
    if config.LLM_BACKEND == "mock":
        return _mock_answer(question, chunks)

    if not config.GEMINI_API_KEY:
        raise SystemExit(
            "尚未設定 GEMINI_API_KEY！\n"
            "請到 https://aistudio.google.com/apikey 申請金鑰，"
            "然後填入專案根目錄的 .env 檔案。"
        )

    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"【上下文】\n{build_context(chunks)}\n\n"
        f"【問題】\n{question}"
    )
    resp = client.models.generate_content(model=config.LLM_MODEL, contents=prompt)
    return resp.text or "(模型無回覆內容)"


def _mock_answer(question: str, chunks: list[dict]) -> str:
    """測試用：不呼叫 API，回傳範本答案以驗證 RAG 流程。"""
    lines = [f"（Mock 答案）針對問題「{question}」，檢索到 {len(chunks)} 個區塊："]
    for c in chunks:
        preview = c["text"][:60].replace("\n", " ")
        lines.append(f"- [來源: {c['source']}#{c['chunk_index']}] {preview}…")
    return "\n".join(lines)

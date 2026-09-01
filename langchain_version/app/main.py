"""LangChain 版 RAG 知識庫問答機器人。

用法（與主專案相同介面）：
    python main.py ingest
    python main.py ask "問題"
    python main.py chat
    python main.py ask "問題" --mock   # 不呼叫 API 的測試模式

本檔示範 LangChain 慣用寫法：
- RecursiveCharacterTextSplitter：切塊
- Chroma（langchain-chroma）：向量庫
- ChatPromptTemplate + LCEL（|）：組合提示詞與模型
- with_structured_output(Pydantic)：結構化輸出（取代手寫 regex 解析）
"""
import argparse
import sys

# 注意：命名為 lc_config 而非 config，避免與主專案的 config.py 同名衝突
# （loader.py 內部 import config 需要拿到主專案的那份）
import lc_config as config
from doc_loader import load_documents
from embeddings import FastEmbedLocal
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from reranker import FastEmbedReranker, RerankRetriever

SYSTEM_PROMPT = (
    "你是企業知識庫問答助手。請嚴格遵守以下規則：\n"
    "1. 只根據【上下文】提供的資料回答問題，不要使用上下文之外的知識。\n"
    "2. 若上下文中沒有答案，請直接說「根據知識庫無法回答此問題」，不要編造。\n"
    "3. 回答時請在相關句子的結尾附上引用來源，格式：[來源: 檔案名稱#區塊編號]。\n"
    "4. 使用繁體中文回答，條列清楚、精簡。"
)


class AnswerWithConfidence(BaseModel):
    """結構化輸出：LangChain 會自動讓模型依此 schema 回傳。"""

    answer: str = Field(description="完整回答，含 [來源: 檔案名稱#區塊] 引用")
    confidence_score: int = Field(ge=0, le=100, description="0-100 信心分數；若只能回答「無法回答」必須 ≤ 20")
    reason: str = Field(description="一句話說明信心評估依據")


def _build_embeddings() -> FastEmbedLocal:
    print(f"⏳ 載入 embedding 模型（{config.EMBEDDING_MODEL}）...")
    return FastEmbedLocal(config.EMBEDDING_MODEL, str(config.CACHE_DIR))


def _build_store(embeddings) -> Chroma:
    return Chroma(
        collection_name="knowledge_base_lc",
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # 與主專案一致的餘弦相似度
    )


def _build_retriever(store: Chroma) -> tuple:
    """建構二階段檢索器。

    rerank 啟用時：RerankRetriever（海選 RERANK_TOP_K → reranker 精排回 TOP_K）；
    停用時：直接取 TOP_K。
    """
    rerank_enabled = config.RERANK_TOP_K > config.TOP_K
    if not rerank_enabled:
        return store.as_retriever(search_kwargs={"k": config.TOP_K}), None

    reranker = FastEmbedReranker(
        model_name=config.RERANK_MODEL,
        cache_dir=str(config.CACHE_DIR),
    )
    base_retriever = store.as_retriever(
        search_kwargs={"k": config.RERANK_TOP_K}
    )
    retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker=reranker,
        top_k=config.TOP_K,
    )
    return retriever, reranker


def _retrieve(store: Chroma, question: str) -> tuple[list[Document], bool]:
    """回傳 (精排後的 docs, 是否啟用 rerank)。

    距離顯示用的 scores 另外查詢（compressor 不提供距離）。
    """
    rerank_enabled = config.RERANK_TOP_K > config.TOP_K
    if rerank_enabled:
        candidate_k = min(config.RERANK_TOP_K, len(store.get()["ids"]))
        docs_with_scores = store.similarity_search_with_score(question, k=candidate_k)
        docs = [d for d, _ in docs_with_scores]
        retriever, _ = _build_retriever(store)
        docs = retriever.invoke(question)
        # 用 page_content 對應回距離（精排後的文件仍是同一批物件）
        score_map = {d.page_content: s for d, s in docs_with_scores}
        for d in docs:
            d.metadata["distance"] = round(score_map.get(d.page_content, 0.0), 4)
        return docs, True
    docs_with_scores = store.similarity_search_with_score(question, k=config.TOP_K)
    docs = [d for d, _ in docs_with_scores]
    for d, s in zip(docs, [s for _, s in docs_with_scores]):
        d.metadata["distance"] = round(s, 4)
    return docs, False


def _build_chain(mock: bool):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "【上下文】\n{context}\n\n【問題】\n{question}"),
        ]
    )
    if mock:
        # 測試模式：固定回覆，驗證管線（不支援結構化輸出，無信心分數）
        llm = FakeListChatModel(responses=["（Mock 答案）管線測試成功"])
        return prompt | llm, False
    if not config.GEMINI_API_KEY:
        raise SystemExit(
            "尚未設定 GEMINI_API_KEY！請填入專案根目錄的 .env（申請: https://aistudio.google.com/apikey）"
        )
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        api_key=config.GEMINI_API_KEY,
        temperature=0,
    )
    # with_structured_output：框架直接產生 schema → 模型回傳 Pydantic 物件
    return prompt | llm.with_structured_output(AnswerWithConfidence), True


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        idx = d.metadata.get("chunk_index")
        # 與主專案一致的引用格式：檔案名稱#區塊編號
        label = f"{src}#{idx}" if idx is not None else src
        parts.append(f"[區塊{i}] 檔案: {label}\n{d.page_content}")
    return "\n\n".join(parts)


def cmd_ingest(args) -> None:
    print(f"📂 讀取知識庫: {config.KNOWLEDGE_DIR}")
    docs = load_documents(config.KNOWLEDGE_DIR)
    if not docs:
        print("⚠ 沒有找到任何支援的文件（.txt/.md/.pdf/.docx）")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "。", "！", "？", "；", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    # 為每個切塊加上區塊編號（每個來源檔獨立編號，與主專案一致）
    counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "?")
        idx = counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        counters[src] = idx + 1
    print(f"✂ 切分完成：{len(docs)} 份文件 → {len(chunks)} 個區塊")

    embeddings = _build_embeddings()
    if args.reset:
        import shutil

        if config.CHROMA_DIR.exists():
            shutil.rmtree(config.CHROMA_DIR)
        print("🧹 已清空舊索引")
    store = _build_store(embeddings)
    store.add_documents(chunks)
    print(f"✅ 索引完成！共 {len(chunks)} 個區塊")
    print('現在可以執行: python main.py ask "你的問題"')


def _print_sources(docs: list[Document]) -> None:
    print("\n📎 檢索來源：")
    for d in docs:
        dist = d.metadata.get("distance")
        rk = d.metadata.get("rerank_score")
        dist_str = f"距離 {dist}" if dist is not None else "距離 ?"
        rk_str = f" / rerank {rk}" if rk is not None else ""
        print(f"  • {d.metadata.get('source')}（{dist_str}{rk_str}）")


def cmd_ask(args) -> None:
    embeddings = _build_embeddings()
    store = _build_store(embeddings)
    if len(store.get()["ids"]) == 0:
        raise SystemExit("知識庫是空的！請先執行: python main.py ingest")

    docs, _ = _retrieve(store, args.question)

    chain, structured = _build_chain(args.mock)
    result = chain.invoke({"context": _format_context(docs), "question": args.question})

    if structured:
        print("=" * 50)
        print(result.answer)
        print("=" * 50)
        print(f"\n🎯 信心分數：{result.confidence_score}/100")
        print(f"   模型自評理由：{result.reason}")
    else:
        print("=" * 50)
        print(result.content)
        print("=" * 50)

    _print_sources(docs)


def cmd_chat(args) -> None:
    print("💬 互動模式（輸入 exit 離開）\n")
    embeddings = _build_embeddings()
    store = _build_store(embeddings)
    chain, structured = _build_chain(args.mock)
    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "離開"}:
            break
        docs, _ = _retrieve(store, question)
        result = chain.invoke({"context": _format_context(docs), "question": question})
        print(f"\n🤖 {result.answer if structured else result.content}\n")
        _print_sources(docs)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-langchain", description="LangChain 版 RAG 知識庫問答機器人")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_mock(p) -> None:
        p.add_argument("--mock", action="store_true", help="測試模式：不呼叫 Gemini API")

    p_ingest = sub.add_parser("ingest", help="建立/更新知識庫索引")
    p_ingest.add_argument("--reset", action="store_true", help="先清空索引再重建")
    add_mock(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="單次問答")
    p_ask.add_argument("question", help='問題內容，例如 "請用雙引號包住問題"')
    add_mock(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="互動式問答")
    add_mock(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""命令列入口：ingest（建立索引）/ ask（單次問答）/ chat（互動模式）。

用法：
    python main.py ingest              # 將 data/knowledge 文件建立索引
    python main.py ingest --reset      # 清空索引後重建
    python main.py ask "問題"          # 單次問答
    python main.py chat                # 互動式問答
"""
import argparse
import sys

import config
import rag
from embed_store import VectorStore
from loader import load_all, make_chunks


def cmd_ingest(args) -> None:
    print(f"📂 讀取知識庫: {config.KNOWLEDGE_DIR}")
    docs = load_all(config.KNOWLEDGE_DIR)
    if not docs:
        print("⚠ 沒有找到任何支援的文件（.txt/.md/.pdf/.docx）")
        sys.exit(1)

    chunks = make_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"✂ 切分完成：{len(docs)} 份文件 → {len(chunks)} 個區塊")

    store = VectorStore(config.EMBEDDING_MODEL, config.CHROMA_DIR, config.CHROMA_COLLECTION)
    if args.reset:
        store.reset()
        print("🧹 已清空舊索引")

    added = store.add_chunks(chunks)
    print(f"✅ 索引完成！知識庫現有 {store.count()} 個區塊（本次寫入 {added}）")
    print("現在可以執行: python main.py ask \"你的問題\"")


def _print_confidence(confidence: dict) -> None:
    print(f"\n🎯 信心分數：{confidence['score']}/100（{confidence['label']}）")
    if confidence["reason"]:
        print(f"   模型自評理由：{confidence['reason']}")
    print(
        f"   組成：{confidence['composition']}"
        + (f"（檢索 {confidence['retrieval']} 分" if confidence["llm"] is not None else "（檢索")
        + (f"、模型自評 {confidence['llm']} 分）" if confidence["llm"] is not None else " 分）")
    )


def cmd_ask(args) -> None:
    answer, chunks, confidence = rag.ask(args.question)
    print("=" * 50)
    print(answer)
    print("=" * 50)
    _print_confidence(confidence)
    print("\n📎 檢索來源：")
    for c in chunks:
        print(f"  • {c['source']}#{c['chunk_index']}（距離 {c['distance']}）")


def cmd_chat(args) -> None:
    print("💬 互動模式（輸入 exit 離開）\n")
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
        answer, chunks, confidence = rag.ask(question)
        print(f"\n🤖 {answer}\n")
        _print_confidence(confidence)
        print("\n📎 檢索來源：")
        for c in chunks:
            print(f"  • {c['source']}#{c['chunk_index']}（距離 {c['distance']}）")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-knowledge-bot", description="企業知識庫 RAG 問答機器人")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="建立/更新知識庫索引")
    p_ingest.add_argument("--reset", action="store_true", help="先清空索引再重建")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="單次問答")
    p_ask.add_argument("question", help='問題內容，例如 "請用雙引號包住問題"')
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="互動式問答")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

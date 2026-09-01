# 🧩 LangChain 版 RAG（對照實作）

用 **LangChain 框架**重寫同一支 RAG 知識庫問答機器人，與主專案（手寫版）共用相同的知識庫、embedding 模型與 Gemini 模型。

**這份對照的意義**：主專案從零手寫（懂了底層），這裡看框架怎麼抽象——兩者功能等價，但寫法截然不同。

## 🚀 使用方式

```powershell
# 在專案根目錄執行（會自動讀取根目錄 .env）
docker compose -f langchain_version/docker-compose.yml build
docker compose -f langchain_version/docker-compose.yml run --rm rag-langchain ingest
docker compose -f langchain_version/docker-compose.yml run --rm rag-langchain ask "員工的特休假有幾天？"
```

測試模式（不耗 API 配額）：
```powershell
docker compose -f langchain_version/docker-compose.yml run --rm rag-langchain ask "員工的特休假有幾天？" --mock
```

> 注意：LangChain 版使用獨立的索引（`data/chroma_lc/`），需先 `ingest`。

## 🔄 元件對照表（手寫版 vs LangChain 版）

| 功能 | 手寫版（主專案） | LangChain 版 | LangChain 幫你做了什麼 |
|---|---|---|---|
| 文件載入 | `loader.py`（自訂） | **共用同一個 loader.py**，包成 `Document` | 什麼都沒藏——展示自訂元件可混用 |
| 切塊 | `split_text()` 自訂遞迴切塊 | `RecursiveCharacterTextSplitter` | 套裝切塊器（含中文分隔符設定） |
| Embedding | fastembed 直接呼叫 | **自訂 `FastEmbedLocal`** 包成 `Embeddings` 介面 | 只要實作 `embed_documents` / `embed_query` 兩個方法 |
| 向量庫 | `embed_store.py`（ChromaDB API） | `Chroma`（langchain-chroma） | 向量庫整合、`as_retriever()`、`similarity_search_with_score()` |
| **二階段檢索** | `store.rerank()` 手寫精排 | **自訂 `RerankRetriever`（`BaseRetriever`）＋ `FastEmbedReranker`** | 註：LangChain 1.x 已移除 `ContextualCompressionRetriever`，改用框架的 `BaseRetriever` 抽象自行組合 |
| 提示詞 | 字串拼接 `f-string` | `ChatPromptTemplate` | 提示詞結構化、可重用 |
| LLM 呼叫 | `google-genai` SDK 手寫 | `ChatGoogleGenerativeAI` | 模型 API 統一介面 |
| **結構化輸出** | 手寫 `regex` 解析【信心分數】 | **`with_structured_output(Pydantic)`** | 框架自動讓模型依 schema 回傳物件——**這是框架最大賣點** |
| 管線組合 | 手寫函式 `ask()` | **LCEL：`prompt \| llm`** | 宣告式組合、可串接、可除錯（`chain.invoke()`） |

## ⚖️ 比較心得

**LangChain 的優點：**
- **元件可替換性**：換 LLM（Gemini → OpenAI → 本地模型）只要換一行
- **結構化輸出**：`with_structured_output` 取代手寫 regex——框架價值最明顯的地方
- **生態系**：數百個整合（loaders、vectorstores、tools），寫企業應用省大量時間

**LangChain 的代價：**
- **抽象包袱**：出問題時要追框架內部（例如 chroma 距離刻度、版本差異）
- **依賴較重**：安裝體積與升級風險（`langchain-community` 已被官方 sunset，2026 生態正移轉到獨立套件）
- **客製化受限**：像「rerank 領先幅度加分」這種自訂邏輯，手寫版更直接
- **框架會演化**：實作時發現 LangChain 1.x 已移除 `ContextualCompressionRetriever`——熟悉 `BaseRetriever` 等底層抽象，才能讓自訂元件不受框架版本影響

**結論（面試金句）：**
> *「我先手寫實作 RAG 理解底層原理，再用 LangChain 重構驗證框架抽象——兩者功能等價。手寫版讓我懂每個環節，框架版讓我理解業界如何快速開發與替換元件。」*

## 📁 檔案結構

```
langchain_version/
├── app/
│   ├── main.py        # CLI + LCEL 管線 + 結構化輸出 + 二階段檢索
│   ├── lc_config.py   # 共用根目錄 .env（命名避免與主專案 config 衝突）
│   ├── embeddings.py  # 自訂 Embeddings（包 fastembed）
│   ├── reranker.py    # 自訂 RerankRetriever + FastEmbedReranker（二階段檢索）
│   └── doc_loader.py  # 重用主專案 loader，轉成 Document
├── Dockerfile
├── docker-compose.yml
└── README.md
```

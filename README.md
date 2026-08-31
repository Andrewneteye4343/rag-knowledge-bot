# 📚 企業知識庫 RAG 問答機器人（rag-knowledge-bot）

以 **RAG（Retrieval-Augmented Generation, 檢索增強生成）** 技術打造的企業知識庫問答機器人：把公司文件（PDF、Word、Markdown）變成可自然語言查詢的知識庫，回答時附上引用來源。

本專案為學習專案，刻意**不依賴 LangChain 等大型框架**，用最少的套件親手實作 RAG 每個環節，適合理解底層原理。

## ✨ 功能

- 支援 `.txt` / `.md` / `.pdf` / `.docx` 文件自動載入
- 中文感知的遞迴式切塊（段落 → 句子 → 硬切，含重疊）
- 本地 embedding 向量化（`BAAI/bge-small-zh-v1.5`，不需 GPU、不需上傳文件）
- ChromaDB 向量檢索（餘弦相似度）
- Google Gemini 生成回答，附引用來源
- 三種使用方式：單次問答 / 互動對話 / 批次索引

## 🏗 架構

```
你的問題
   │
   ▼
┌──────────────┐   ┌──────────────────┐   ┌─────────────┐
│ Embedding    │──▶│ ChromaDB 向量檢索 │──▶│ 相關區塊     │
│ (本地模型)    │   │ (語意相似度 TOP_K) │   │ + 來源標記    │
└──────────────┘   └──────────────────┘   └──────┬──────┘
                                                 ▼
┌──────────────┐   ┌──────────────────────────────────────┐
│ 知識庫文件     │──▶│ 提示詞組裝（系統指令+上下文+問題）        │
│ data/knowledge│   │ ──▶ Gemini LLM ──▶ 附引用的回答        │
└──────────────┘   └──────────────────────────────────────┘
```

資料流：`ingest` 將文件切塊 → 向量化 → 存入 ChromaDB；`ask` 將問題向量化 → 檢索最相似區塊 → 連同問題丟給 LLM 生成回答。

## 🖥 環境需求（你的筆電）

- Windows 11 + **Docker Desktop**（已安裝 ✅）
- 一個 **Google Gemini API Key**（免費）：
  1. 到 <https://aistudio.google.com/apikey> 用 Google 帳號登入
  2. 點「建立 API 金鑰」→ 複製金鑰

## 🚀 快速開始

### 1. 下載與解壓縮

將專案壓縮檔解壓縮到筆電的某個資料夾，例如 `C:\projects\rag-knowledge-bot`。

### 2. 設定 API Key

在專案資料夾內，把 `.env.example` 複製成 `.env`，編輯內容：

```
GEMINI_API_KEY=你的金鑰貼在這裡
```

### 3. 建立索引（讀懂你的文件）

開啟 **PowerShell**（或 Windows Terminal），進入專案資料夾後執行：

```powershell
docker compose run --rm rag ingest
```

第一次執行會下載 Python 映像檔與 embedding 模型（約 1GB），之後就很快。
看到 `✅ 索引完成！` 代表知識庫建立成功。

### 4. 問問題

```powershell
docker compose run --rm rag ask "員工的特休假有幾天？"
```

互動模式：

```powershell
docker compose run --rm rag chat
```

## 📂 加入你自己的文件

1. 把文件（`.pdf` / `.docx` / `.txt` / `.md`）丟進 `data\knowledge\` 資料夾
2. 重新執行 `docker compose run --rm rag ingest`
3. 完成！直接用 `ask` 問吧

## ⚙️ 可調參數（.env）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `LLM_MODEL` | `gemini-3.6-flash` | Gemini 模型名稱（新帳號請勿用 2.5-flash，已停止開放） |
| `LLM_BACKEND` | `gemini` | `gemini`＝正式 API；`mock`＝測試用不呼叫 API |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | 中英雙語 embedding（繁中實測開源最強；下載 2.2GB）。輕量替代：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `CHUNK_SIZE` | `500` | 切塊大小（字元數） |
| `CHUNK_OVERLAP` | `50` | 區塊重疊（字元數） |
| `TOP_K` | `4` | 每次檢索回傳的區塊數 |

## 🧪 本機測試（不需 Docker）

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
LLM_BACKEND=mock python tests/test_pipeline.py
```

## ⚠️ 常見問題

- **回答不準確？** 更新文件後務必重新 `ingest`；可調大 `TOP_K`、換更好的 embedding 模型。
- **答案說「無法回答」？** 表示檢索到的區塊沒有相關內容，請確認文件已 ingest、問題在知識庫範圍內。
- **免費額度**：Gemini API 有每日免費額度，個人練習綽綽有餘。

## 📈 延伸方向（後續專案）

- 加入網頁爬蟲，讓 Agent 自動更新知識庫
- 換成 OpenAI / DeepSeek（改 `llm.py` 即可）
- 串接 Line / Telegram 變成客服機器人
- 加上「答案可不可信」的信心分數評估

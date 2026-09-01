# 📚 企業知識庫 RAG 問答機器人（rag-knowledge-bot）

以 **RAG（Retrieval-Augmented Generation, 檢索增強生成）** 技術打造的企業知識庫問答機器人：把公司文件（PDF、Word、Markdown）變成可自然語言查詢的知識庫，回答時附上引用來源。

本專案為學習專案，刻意**不依賴 LangChain 等大型框架**，用最少的套件親手實作 RAG 每個環節，適合理解底層原理。

## ✨ 功能

- 支援 `.txt` / `.md` / `.pdf` / `.docx` 文件自動載入
- **掃描型 PDF 自動 OCR**（偵測到無文字層時，自動用 Tesseract 辨識繁中＋英文）
- 中文感知的遞迴式切塊（段落 → 句子 → 硬切，含重疊）
- 本地 embedding 向量化（`intfloat/multilingual-e5-large`，中英雙語、不需 GPU、不需上傳文件）
- ChromaDB 向量檢索（餘弦相似度）
- **Reranker 二階段檢索**（bi-encoder 海選 → cross-encoder 精排，提升檢索精準度）
- Google Gemini 生成回答，附引用來源
- **信心分數評估**（檢索距離 + 模型自評加權合成，回答附可信度）
- 三種使用方式：單次問答 / 互動對話 / 批次索引

## 🏗 架構

```
你的問題
   │
   ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌─────────────┐
│ Embedding    │──▶│ ChromaDB 向量檢索 │──▶│ Reranker 精排 │──▶│ 相關區塊     │
│ (本地模型)    │   │ (海選 RERANK_TOP_K) │   │ (cross-encoder)│   │ + 來源標記    │
└──────────────┘   └──────────────────┘   └──────────────┘   └──────┬──────┘
                                                                     ▼
┌──────────────┐   ┌──────────────────────────────────────────────────────┐
│ 知識庫文件     │──▶│ 提示詞組裝（系統指令+上下文+問題）                       │
│ data/knowledge│   │ ──▶ Gemini LLM ──▶ 附引用＋信心分數的回答               │
└──────────────┘   └──────────────────────────────────────────────────────┘
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
| `TOP_K` | `4` | 每次檢索回傳的區塊數（最終給 LLM） |
| `RERANK_TOP_K` | `20` | Reranker 第一階段海選數量；`<= TOP_K` 即停用 |
| `RERANK_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | 多語言 reranker（1.1GB）；輕量替代 `Xenova/ms-marco-MiniLM-L-6-v2`（僅英文） |

## 🔄 Reranker 二階段檢索

向量檢索（bi-encoder）速度快但交互淺——「距離近」不代表「真的回答問題」。因此先海選出 `RERANK_TOP_K` 個候選，再用 cross-encoder（問題＋區塊同時輸入，完整語意交互）精排回 `TOP_K` 個，品質顯著提升。

- 啟用 rerank 後，檢索來源會多顯示 `rerank 分數`；rerank 負責改善排序，信心分數的檢索部分以距離為基礎、rerank「領先幅度」做加分（不直接換算 rerank 原始分數——cross-encoder 刻度因模型而異，直接換算會誤判）
- 停用方式：`.env` 設 `RERANK_TOP_K=4`（或任何 `<= TOP_K` 的值）

## 🎯 信心分數（可信度評估）

每個回答都會附上 0-100 的信心分數與評估理由，組成透明可解釋：

```
🎯 信心分數：85/100（高）
   模型自評理由：所有數字皆有來源引用，檢索距離低。
   組成：檢索 40% + 模型自評 60%（檢索 78 分、模型自評 90 分）
```

**計算方式：**

| 項目 | 權重 | 說明 |
|---|---|---|
| 檢索評估 | 40% | top-1 餘弦距離越近越高分；top-1 明顯領先 top-2 時額外加分（最多 15 分） |
| 模型自評 | 60% | 要求 LLM 回答後自評 0-100 並附一句理由；「無法回答」時自評必須 ≤ 20 |

分級：`≥75 高`、`50-74 中`、`<50 低`。若模型未照格式提供自評，自動退回「僅檢索評估」並於組成欄位標註。

> 為什麼不只用檢索距離？距離是模型相關的相對值，無法代表「答案正確性」；搭配 LLM 自評可捕捉「上下文不足但仍硬答」的狀況。

## 🧪 本機測試（不需 Docker）

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
LLM_BACKEND=mock python tests/test_pipeline.py
```

## ⚠️ 常見問題

- **回答不準確？** 更新文件後務必重新 `ingest`；可調大 `TOP_K`、換更好的 embedding 模型。
- **掃描型 PDF 讀不到？** 已內建 OCR 自動後備：偵測到無文字層的 PDF 會自動用 Tesseract 辨識（`OCR_LANG=chi_tra+eng` 可調）。OCR 較慢，多頁文件請耐心等待。
- **答案說「無法回答」？** 表示檢索到的區塊沒有相關內容，請確認文件已 ingest、問題在知識庫範圍內。
- **免費額度**：Gemini API 有每日免費額度，個人練習綽綽有餘。

## 📈 延伸方向（後續專案）

- 加入網頁爬蟲，讓 Agent 自動更新知識庫
- 換成 OpenAI / DeepSeek（改 `llm.py` 即可）
- 串接 Line / Telegram 變成客服機器人
- 加上「答案可不可信」的信心分數評估

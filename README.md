# 台股策略分析系統 (Stock Strategy Twe)

針對台灣上市/上櫃股票的多策略掃描、回測、AI 排名與持股管理工具。以 `Strategy_twe.py` 為主入口，每天執行可自動完成：資料抓取 → 13 種選股策略 → 回測 → 三層融合排名（規則 + ML + Deep）→ 互動 HTML 報告 → 持股風險監控 → 排程重訓。

## 主要特色

- **13 種選股策略**：法人連續買超、量價突破、RSI 超賣、布林通道、營收成長、融資下降、MACD/KD 黃金交叉、達華斯箱突破、底部量價背離、相對強弱…等
- **三層融合排名系統**
  - Rule-based 綜合分數（策略信號 + 技術 + 法人）
  - XGBoost GPU Learning-to-Rank（`ranker_model.json`）
  - PyTorch GRU 序列深度模型（`deep_model.pt`）
- **GPU 加速**（需 CUDA 12.8+，RTX 30 系列以上）
  - Numba CUDA 參數網格回測（`gpu_backtest.py`）
  - XGBoost device=cuda
  - PyTorch GRU 訓練
- **持股管理**（純前端 LocalStorage，無需伺服器）
  - 動態停損（ATR Chandelier / 保本 / 分層）
  - 健檢分數、加碼訊號、投組風險卡
  - 交易歷史自動記錄至 `trade_history.csv`
  - 現價離線 fallback（yfinance 漏抓時以 max_high 估算）
- **互動 HTML 報告**：Plotly 互動圖表、排序表格、暗色主題
- **自動排程** (`scheduler.py`)：每天執行主程式時，自動偵測是否到期重訓 ML/Deep 模型或重跑 GPU 網格

## 目錄結構

```
Strategy_twe.py          # 主入口：日常執行此檔
scheduler.py             # 定時任務排程器
backtest.py              # 回測引擎 + 策略模擬
gpu_backtest.py          # GPU 參數網格回測 (Numba CUDA)
ml_ranker.py             # XGBoost GPU Learning-to-Rank
deep_ranker.py           # PyTorch GRU 序列深度模型

best_params.json         # GPU 網格搜尋結果（季度更新）
ranker_model.json        # 訓練好的 XGBoost 模型
deep_model.pt            # 訓練好的 PyTorch 模型
deep_model_meta.json     # 深度模型中繼資料
ranker_features.json     # ML 特徵欄位定義

holdings.example.csv     # 持股格式範例（實際請用 holdings.csv，已 gitignore）
```

## 環境需求

- Python 3.10+（開發環境使用 3.13）
- CUDA 12.8+（GPU 訓練用；純 CPU 可用但速度較慢）
- Windows / Linux / macOS

### 安裝依賴

```bash
pip install pandas numpy requests yfinance xgboost plotly

# PyTorch (CUDA 12.8 for RTX 50 系列)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Numba CUDA (GPU 網格回測)
pip install numba numba-cuda nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12 \
            nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
```

## 快速開始

```bash
# 1. 每日執行：自動跑策略 + 排程器自動判斷是否需要重訓模型
python Strategy_twe.py

# 2. 開啟生成的 HTML 報告（檔名形如 2026-04-21_上市上櫃策略報告.html）

# 3. 管理持股：複製 holdings.example.csv 為 holdings.csv 後編輯，或直接在 HTML 報告的「我的持股」區塊新增
cp holdings.example.csv holdings.csv
```

## 排程器

```bash
# 查看任務狀態
python scheduler.py status

# 強制重訓（例：立即重訓 ML + Deep）
python scheduler.py run --force ml_train,deep_train

# 模擬（不實際執行）
python scheduler.py run --force all --dry-run
```

| 任務 | 預設週期 | 條件 |
|------|---------|------|
| `gpu_grid`    | 90 天（季度） | 僅週末 |
| `ml_train`    | 14 天         | 僅週末 |
| `deep_train`  | 14 天         | 僅週末 |

### 環境變數

```bash
# 暫時跳過排程器
STRATEGY_SKIP_SCHEDULER=1 python Strategy_twe.py

# 強制執行指定任務
STRATEGY_FORCE_SCHEDULER=ml_train python Strategy_twe.py
```

## 手動訓練

```bash
# GPU 參數網格（產出 best_params.json）
python gpu_backtest.py --all --period 2y --top-k 10

# XGBoost Ranker
python ml_ranker.py train --stocks auto --period 3y

# PyTorch Deep Ranker
python deep_ranker.py train --stocks auto --period 3y --epochs 40
```

## 注意事項

- `holdings.csv` 與 `trade_history.csv` **含個人交易資料**，已列入 `.gitignore` 不會提交
- 每日 HTML 報告（`YYYY-MM-DD_*.html`）與執行 log 也已 gitignore
- 本系統僅供研究使用，不構成任何投資建議

## 授權

個人專案，未定義授權條款。若需使用請聯絡作者。

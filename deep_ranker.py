"""
PyTorch GPU 序列深度模型：路線 3
————————————————————————————
以 GRU + multi-head 直接學習股票最近 N 日的時序模式，輸出 T+5/10/20 預期報酬
與 20 日上漲機率，補足規則/ML ranker 無法捕捉的非線性時序依賴。

使用方法：
    python deep_ranker.py train --stocks demo --period 3y --epochs 40
    python deep_ranker.py predict --stocks auto --top-n 30

在 Strategy_twe.py 中會被 apply_deep_ranking 自動載入。

設計：
    - 滑動視窗長度 N=60 天，14 個時序特徵
    - GRU(14→64) × 2-layer + LayerNorm + Dropout
    - 四個預測頭：T+5 / T+10 / T+20 報酬（回歸）+ T+20>5% 分類
    - 損失 = Σ Huber(報酬) + BCE(分類)，時間序列切分訓練/驗證
    - 排名分數 = 0.6 × σ(cls_logit) × 100 + 0.4 × clip(pred_20, -15%, +30%) 正規化
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yfinance as yf

import backtest as bt


MODEL_FILE = 'deep_model.pt'
META_FILE = 'deep_model_meta.json'

# 序列設定
WINDOW = 60                  # 取近 60 天作為輸入
HORIZONS = (5, 10, 20)       # 預測 T+5, T+10, T+20 報酬
UP_THRESHOLD = 0.05          # T+20 > 5% 視為正樣本
MIN_HISTORY = WINDOW + max(HORIZONS) + 20

# 基礎特徵（14 個，原版相容）
BASE_SEQ_FEATURES = [
    'logret_1d',
    'close_ma5', 'close_ma20', 'close_ma60',
    'vol_ratio',
    'rsi_n', 'k_n', 'd_n',
    'macd_hist_n', 'bb_pos', 'atr_pct', 'hl_range',
    'dist_52w', 'log_vol_z',
]

# Tier C 進階價量衍生特徵（10 個）：從價量單獨衍生、無需外部資料
EXTRA_SEQ_FEATURES = [
    'ret_5d_cum', 'ret_20d_cum',     # 短/中期累積報酬
    'vol_of_vol_20d',                # 波動度的波動度（Volatility-of-Volatility）
    'pv_corr_20d',                   # 價量 Pearson 相關（+=齊漲，−=背離）
    'range_expand',                  # HL 範圍 20 日 z-score（突破前放大）
    'volume_surge_5d',               # 近 5 日量 / 20 日量（爆量訊號）
    'rsi_slope_5d',                  # RSI 5 日斜率（動能反轉）
    'macd_hist_slope',               # MACD 柱狀體 5 日斜率
    'squeeze_intensity',             # 布林收斂強度：is_squeeze 的 EMA(span=10)，0~1（C15 重新命名）
    'trend_strength',                # ADX-like: MA5/MA20/MA60 方向一致性
]

# 預設訓練特徵 = base + extra（24 個）
SEQ_FEATURES = BASE_SEQ_FEATURES + EXTRA_SEQ_FEATURES


# ──────────────────────────────────────────────────────────
# 特徵工程（逐時間步）
# ──────────────────────────────────────────────────────────

def build_seq_features(df: pd.DataFrame) -> pd.DataFrame:
    """df 需已經過 bt._add_indicators，此函式會回傳同樣 index 的特徵 DataFrame。"""
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    out = pd.DataFrame(index=df.index)

    out['logret_1d'] = np.log(c / c.shift(1))

    ma60 = c.rolling(60, min_periods=20).mean()
    out['close_ma5'] = c / df['MA5'].replace(0, np.nan) - 1
    out['close_ma20'] = c / df['MA20'].replace(0, np.nan) - 1
    out['close_ma60'] = c / ma60.replace(0, np.nan) - 1

    out['vol_ratio'] = df['Vol_ratio']

    out['rsi_n'] = df['RSI'] / 100.0
    out['k_n'] = df['K'] / 100.0
    out['d_n'] = df['D'] / 100.0

    out['macd_hist_n'] = df['MACD_hist'] / c.replace(0, np.nan)

    bb_range = (df['BB_upper'] - df['BB_lower']).replace(0, np.nan)
    out['bb_pos'] = (c - df['BB_lower']) / bb_range

    out['atr_pct'] = df['ATR'] / c.replace(0, np.nan)
    out['hl_range'] = (h - l) / c.replace(0, np.nan)

    out['dist_52w'] = c / df['High_52w'].replace(0, np.nan) - 1

    log_v = np.log(v.replace(0, np.nan))
    vol_mean = log_v.rolling(60, min_periods=20).mean()
    vol_std = log_v.rolling(60, min_periods=20).std().replace(0, np.nan)
    out['log_vol_z'] = (log_v - vol_mean) / vol_std

    # ========== Tier C 進階特徵 ==========
    logret = out['logret_1d']
    out['ret_5d_cum'] = logret.rolling(5, min_periods=3).sum()
    out['ret_20d_cum'] = logret.rolling(20, min_periods=10).sum()

    ret_std_20 = logret.rolling(20, min_periods=10).std()
    out['vol_of_vol_20d'] = (
        ret_std_20.rolling(20, min_periods=10).std()
        / ret_std_20.rolling(20, min_periods=10).mean().replace(0, np.nan)
    )

    out['pv_corr_20d'] = (
        logret.rolling(20, min_periods=10).corr(log_v.diff())
    )

    hl_rng = (h - l) / c.replace(0, np.nan)
    rng_mean = hl_rng.rolling(20, min_periods=10).mean()
    rng_std = hl_rng.rolling(20, min_periods=10).std().replace(0, np.nan)
    out['range_expand'] = (hl_rng - rng_mean) / rng_std

    v_5 = v.rolling(5, min_periods=3).mean()
    v_20 = v.rolling(20, min_periods=10).mean().replace(0, np.nan)
    out['volume_surge_5d'] = (v_5 / v_20).apply(np.log)

    rsi_raw = df['RSI']
    out['rsi_slope_5d'] = (rsi_raw - rsi_raw.shift(5)) / 5.0 / 100.0

    mhist = df['MACD_hist']
    out['macd_hist_slope'] = (mhist - mhist.shift(5)) / 5.0 / c.replace(0, np.nan)

    # 布林收斂強度：BB_width 是否低於 60 日中位數，再做 EMA(span=10) 平滑為 0~1
    # （越接近 1 表示近期持續處於收斂狀態，準備突破）
    bb_w_pct = (df['BB_upper'] - df['BB_lower']) / c.replace(0, np.nan)
    bb_w_med = bb_w_pct.rolling(60, min_periods=20).median()
    is_squeeze = (bb_w_pct < bb_w_med).astype(float)
    out['squeeze_intensity'] = is_squeeze.ewm(span=10, adjust=False).mean()

    # 趨勢一致性：MA5 > MA20 > MA60 = +1；全部相反 = -1
    ma5 = df['MA5']
    ma20 = df['MA20']
    trend = (
        np.sign(ma5 - ma20).fillna(0)
        + np.sign(ma20 - ma60).fillna(0)
    ) / 2.0
    out['trend_strength'] = trend

    # 剪掉極端值以穩定訓練
    out = out.clip(-5, 5)
    return out


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    c = df['Close']
    out = pd.DataFrame(index=df.index)
    for h in HORIZONS:
        out[f'ret_{h}d'] = c.shift(-h) / c - 1
    out['cls_up'] = (out[f'ret_{HORIZONS[-1]}d'] > UP_THRESHOLD).astype(np.float32)
    return out


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y_reg: np.ndarray, Y_cls: np.ndarray,
                 dates: np.ndarray, stock_ids: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y_reg = torch.from_numpy(Y_reg).float()
        self.Y_cls = torch.from_numpy(Y_cls).float()
        self.dates = dates
        self.stock_ids = stock_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y_reg[i], self.Y_cls[i]


def _windowize(feat: np.ndarray, labels_reg: np.ndarray, labels_cls: np.ndarray,
               dates: np.ndarray, sid: str) -> Tuple[np.ndarray, ...]:
    """從一支股票時序產出 (n_windows, WINDOW, n_feat) 以及對齊的標籤、日期。
    只取特徵與標籤皆不含 NaN 的視窗末端。"""
    n = len(feat)
    if n < WINDOW + max(HORIZONS):
        return None

    X_list, Yr_list, Yc_list, d_list = [], [], [], []
    for end in range(WINDOW - 1, n):
        # 標籤對齊「視窗最後一天」
        if np.isnan(labels_reg[end]).any() or np.isnan(labels_cls[end]):
            continue
        window = feat[end - WINDOW + 1: end + 1]
        if np.isnan(window).any():
            continue
        X_list.append(window)
        Yr_list.append(labels_reg[end])
        Yc_list.append(labels_cls[end])
        d_list.append(dates[end])

    if not X_list:
        return None
    return (np.stack(X_list).astype(np.float32),
            np.stack(Yr_list).astype(np.float32),
            np.array(Yc_list, dtype=np.float32),
            np.array(d_list),
            np.array([sid] * len(X_list)))


def build_training_dataset(
    stock_ids: Sequence[str],
    period: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """回傳 (X, Y_reg, Y_cls, dates, stock_ids)。"""
    print(f"\n===== 建立序列訓練資料 ({len(stock_ids)} 股, {period}) =====")
    t0 = time.time()
    cache = bt._batch_download(list(stock_ids), period=period)
    print(f"  成功下載 {len(cache)} 支")

    xs, yr, yc, ds, ss = [], [], [], [], []
    for sid, df in cache.items():
        if len(df) < MIN_HISTORY:
            continue
        bt._add_indicators(df)
        feats = build_seq_features(df)
        labels = build_labels(df)

        feat_arr = feats[SEQ_FEATURES].values.astype(np.float32)
        lreg = labels[[f'ret_{h}d' for h in HORIZONS]].values.astype(np.float32)
        lcls = labels['cls_up'].values.astype(np.float32)

        res = _windowize(feat_arr, lreg, lcls, df.index.values, sid)
        if res is None:
            continue
        X, Yr, Yc, dates, sids = res
        xs.append(X); yr.append(Yr); yc.append(Yc); ds.append(dates); ss.append(sids)

    if not xs:
        raise RuntimeError("資料不足，無法建訓練集")
    X = np.concatenate(xs)
    Yr = np.concatenate(yr)
    Yc = np.concatenate(yc)
    D = np.concatenate(ds)
    S = np.concatenate(ss)
    print(f"  共 {len(X):,} 個視窗 × {WINDOW} 天 × {len(SEQ_FEATURES)} 特徵")
    print(f"  涵蓋 {pd.Timestamp(D.min()).date()} ~ {pd.Timestamp(D.max()).date()}")
    print(f"  正樣本比例 (T+20 > {UP_THRESHOLD:.0%}): {Yc.mean():.2%}")
    print(f"  建立耗時 {time.time() - t0:.1f}s")
    return X, Yr, Yc, D, S


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────

class PriceGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.25,
                 n_horizons: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, n_horizons),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.norm(out[:, -1, :])
        h = self.dropout(h)
        return self.reg_head(h), self.cls_head(h).squeeze(-1)


class PriceTransformer(nn.Module):
    """Tier B2: Transformer Encoder 替代 GRU。
    同樣吃 (B, T, F) 序列；加上 sinusoidal positional encoding。
    """
    def __init__(self, n_features: int, hidden: int = 64,
                 layers: int = 3, dropout: float = 0.25,
                 n_horizons: int = 3, n_heads: int = 4, window: int = 60):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        pe = torch.zeros(window, hidden)
        pos = torch.arange(0, window, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden, 2).float() * -(math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_enc', pe.unsqueeze(0))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, n_horizons),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.pos_enc[:, :h.size(1)]
        h = self.encoder(h)
        # 取最後一個 time-step 作為 query
        h = self.norm(h[:, -1, :])
        h = self.dropout(h)
        return self.reg_head(h), self.cls_head(h).squeeze(-1)


def build_model(model_type: str, n_features: int, hidden: int, layers: int,
                dropout: float, n_horizons: int, window: int = WINDOW) -> nn.Module:
    if model_type == 'transformer':
        return PriceTransformer(
            n_features=n_features, hidden=hidden, layers=layers,
            dropout=dropout, n_horizons=n_horizons, window=window,
        )
    return PriceGRU(
        n_features=n_features, hidden=hidden,
        layers=layers, dropout=dropout, n_horizons=n_horizons,
    )


# ──────────────────────────────────────────────────────────
# 訓練
# ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 40
    patience: int = 8
    # Tier A3 修正：train / val / calib 三段時間序列切分
    # 預設 70% train + 15% val（early-stopping）+ 15% calib（conformal quantile）
    train_pct: float = 0.70
    val_pct: float = 0.15  # 接著 train_pct，剩下做 calib
    reg_weight: float = 1.0
    cls_weight: float = 0.5
    huber_delta: float = 0.05
    grad_clip: float = 1.0
    model_type: str = 'gru'  # 'gru' or 'transformer'
    n_heads: int = 4         # 僅 transformer 使用


def train_model(X, Yr, Yc, D, cfg: TrainConfig,
                device: str = 'cuda') -> Tuple[nn.Module, dict]:
    # Tier A3 修正：時間序列切分為 train / val / calib（避免 conformal data leakage）
    order = np.argsort(D)
    X, Yr, Yc, D = X[order], Yr[order], Yc[order], D[order]
    unique_dates = np.unique(D)
    n_dates = len(unique_dates)
    train_end = int(n_dates * cfg.train_pct)
    val_end = int(n_dates * (cfg.train_pct + cfg.val_pct))
    # 至少留 1 天做 calib，否則退回兩段切分
    if val_end >= n_dates:
        val_end = max(train_end + 1, n_dates - 1)
    train_split_date = unique_dates[train_end]
    val_split_date = unique_dates[val_end]

    train_mask = D < train_split_date
    valid_mask = (D >= train_split_date) & (D < val_split_date)
    calib_mask = D >= val_split_date
    if calib_mask.sum() == 0:
        # fallback: 沒有 calib 樣本（資料太短），與 valid 共用（會有輕微洩漏）
        calib_mask = valid_mask
        print("  ⚠ 資料量不足獨立 calibration set，conformal 退回 valid set")

    print(f"\n===== 訓練 PriceGRU =====")
    print(f"  train: {train_mask.sum():,}  valid: {valid_mask.sum():,}  "
          f"calib: {calib_mask.sum():,}")
    print(f"  splits: train→{pd.Timestamp(train_split_date).date()}  "
          f"val→{pd.Timestamp(val_split_date).date()}")

    train_ds = SlidingWindowDataset(X[train_mask], Yr[train_mask], Yc[train_mask],
                                    D[train_mask], None)
    valid_ds = SlidingWindowDataset(X[valid_mask], Yr[valid_mask], Yc[valid_mask],
                                    D[valid_mask], None)
    calib_ds = SlidingWindowDataset(X[calib_mask], Yr[calib_mask], Yc[calib_mask],
                                    D[calib_mask], None)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size * 2,
                              shuffle=False)
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size * 2,
                              shuffle=False)

    model = build_model(
        cfg.model_type, n_features=X.shape[2], hidden=cfg.hidden,
        layers=cfg.layers, dropout=cfg.dropout, n_horizons=Yr.shape[1],
        window=X.shape[1],
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=cfg.lr * 0.05)
    reg_loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
    cls_loss_fn = nn.BCEWithLogitsLoss()

    print(f"  裝置: {device} | model_params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr} dropout={cfg.dropout}")

    best_valid = float('inf')
    best_state = None
    patience_left = cfg.patience
    history = {'train': [], 'valid': [], 'valid_ic': []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_ep = time.time()
        train_losses = []
        for xb, yrb, ycb in train_loader:
            xb, yrb, ycb = xb.to(device, non_blocking=True), \
                           yrb.to(device, non_blocking=True), \
                           ycb.to(device, non_blocking=True)
            opt.zero_grad()
            pred_r, pred_c = model(xb)
            lr_ = reg_loss_fn(pred_r, yrb)
            lc_ = cls_loss_fn(pred_c, ycb)
            loss = cfg.reg_weight * lr_ + cfg.cls_weight * lc_
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_losses.append(loss.item())

        sched.step()

        # Valid
        model.eval()
        v_losses = []
        all_pred20, all_true20 = [], []
        with torch.no_grad():
            for xb, yrb, ycb in valid_loader:
                xb, yrb, ycb = xb.to(device), yrb.to(device), ycb.to(device)
                pred_r, pred_c = model(xb)
                lr_ = reg_loss_fn(pred_r, yrb)
                lc_ = cls_loss_fn(pred_c, ycb)
                loss = cfg.reg_weight * lr_ + cfg.cls_weight * lc_
                v_losses.append(loss.item())
                all_pred20.append(pred_r[:, -1].cpu().numpy())
                all_true20.append(yrb[:, -1].cpu().numpy())

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(v_losses))
        pred20 = np.concatenate(all_pred20)
        true20 = np.concatenate(all_true20)
        # Spearman IC (rank correlation)
        ic = float(pd.Series(pred20).corr(pd.Series(true20), method='spearman'))
        history['train'].append(train_loss)
        history['valid'].append(valid_loss)
        history['valid_ic'].append(ic)

        improved = valid_loss < best_valid - 1e-5
        if improved:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
            mark = ' *'
        else:
            patience_left -= 1
            mark = ''

        print(f"  ep{epoch:>3d} | train {train_loss:.5f} | "
              f"valid {valid_loss:.5f} | IC@20 {ic:+.4f} | "
              f"lr {sched.get_last_lr()[0]:.5f} | "
              f"{time.time() - t_ep:.1f}s{mark}")

        if patience_left <= 0:
            print(f"  early stop at ep{epoch} (best valid {best_valid:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ========== Tier D2 (A3 fixed): Conformal residuals ==========
    # 使用獨立 calibration set（不參與 early stopping）以避免區間被低估
    model.eval()
    all_pred_r, all_true_r = [], []
    with torch.no_grad():
        for xb, yrb, _ in calib_loader:
            xb = xb.to(device)
            pr, _ = model(xb)
            all_pred_r.append(pr.cpu().numpy())
            all_true_r.append(yrb.numpy())
    if all_pred_r:
        pred_r_v = np.concatenate(all_pred_r)
        true_r_v = np.concatenate(all_true_r)
        residuals = true_r_v - pred_r_v
        conformal = {
            'q_low': np.percentile(residuals, 2.5, axis=0).tolist(),
            'q_high': np.percentile(residuals, 97.5, axis=0).tolist(),
            'mae': np.mean(np.abs(residuals), axis=0).tolist(),
            'n_samples': int(len(residuals)),
            'source': 'calib' if calib_mask.sum() > 0 and not np.array_equal(calib_mask, valid_mask) else 'valid',
        }
    else:
        conformal = {'q_low': [0]*Yr.shape[1], 'q_high': [0]*Yr.shape[1],
                     'mae': [0]*Yr.shape[1], 'n_samples': 0, 'source': 'empty'}
    history['conformal'] = conformal
    print(f"  Conformal 95% 區間 (T+20, source={conformal.get('source','?')}, n={conformal.get('n_samples',0)}): "
          f"[{conformal['q_low'][-1]*100:+.2f}%, {conformal['q_high'][-1]*100:+.2f}%] "
          f"(MAE {conformal['mae'][-1]*100:.2f}%)")

    # ========== Tier D17：模型校準診斷（reliability + IC + threshold hit-rate） ==========
    # 在 calib set（或 valid 退路）上量測：
    #   • Spearman IC（rank correlation）@5/10/20
    #   • up_prob 分桶後實際 T+20 > 5% 命中率（reliability diagram）
    #   • up_prob > {0.5, 0.6, 0.7} 的 precision
    diag_loader = calib_loader if calib_mask.sum() > 0 and not np.array_equal(calib_mask, valid_mask) else valid_loader
    model.eval()
    pr_all, pc_all, yr_all, yc_all = [], [], [], []
    with torch.no_grad():
        for xb, yrb, ycb in diag_loader:
            xb = xb.to(device)
            pr, pc = model(xb)
            pr_all.append(pr.cpu().numpy())
            pc_all.append(torch.sigmoid(pc).cpu().numpy())
            yr_all.append(yrb.numpy())
            yc_all.append(ycb.numpy())
    if pr_all:
        pr_all = np.concatenate(pr_all)
        pc_all = np.concatenate(pc_all)
        yr_all = np.concatenate(yr_all)
        yc_all = np.concatenate(yc_all)
        ic = {f"h{HORIZONS[i]}": float(pd.Series(pr_all[:, i]).corr(
                pd.Series(yr_all[:, i]), method='spearman'))
              for i in range(pr_all.shape[1])}
        # Reliability: 把 up_prob 切 5 桶
        bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
        bucket_idx = np.digitize(pc_all, bins) - 1
        reliability = []
        for b in range(len(bins) - 1):
            mask = bucket_idx == b
            if mask.sum() < 10:
                continue
            reliability.append({
                'bin': f"[{bins[b]:.2f}, {bins[b+1]:.2f})",
                'pred_mean': float(pc_all[mask].mean()),
                'actual_rate': float(yc_all[mask].mean()),
                'n': int(mask.sum()),
            })
        precisions = {}
        for thr in (0.5, 0.6, 0.7):
            sel = pc_all >= thr
            precisions[f"p>={thr}"] = (
                {'n': int(sel.sum()),
                 'precision': float(yc_all[sel].mean()) if sel.sum() else None}
            )
        history['calibration'] = {
            'ic_spearman': ic,
            'reliability': reliability,
            'precision_at_threshold': precisions,
        }
        print("\n  ── 校準診斷（Calibration Check）──")
        ic_str = ' | '.join(f"{k}={v:+.3f}" for k, v in ic.items())
        print(f"   Spearman IC: {ic_str}")
        for r in reliability:
            print(f"   bin {r['bin']}: pred={r['pred_mean']:.2f} | "
                  f"actual={r['actual_rate']:.2f} | n={r['n']}")
        for k, v in precisions.items():
            if v['precision'] is not None:
                print(f"   {k} → precision={v['precision']:.2f} (n={v['n']})")

    return model, history


# ──────────────────────────────────────────────────────────
# 儲存 / 載入 / 推論
# ──────────────────────────────────────────────────────────

def save_model(model: nn.Module, path: str = MODEL_FILE,
               cfg: Optional[TrainConfig] = None,
               conformal: Optional[dict] = None):
    torch.save({
        'state_dict': model.state_dict(),
        'features': SEQ_FEATURES,
        'window': WINDOW,
        'horizons': list(HORIZONS),
        'hidden': cfg.hidden if cfg else 64,
        'layers': cfg.layers if cfg else 2,
        'model_type': (cfg.model_type if cfg else 'gru'),
        'n_heads': (cfg.n_heads if cfg else 4),
        'conformal': conformal or {},
    }, path)
    meta = {
        'features': SEQ_FEATURES, 'window': WINDOW,
        'horizons': list(HORIZONS),
        'model_type': (cfg.model_type if cfg else 'gru'),
        'conformal': conformal or {},
        'trained_at': pd.Timestamp.now().isoformat(),
    }
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  模型已儲存: {path}")


_LOADED_CKPT_CACHE: dict = {}
# D16：每個 ohlcv_cache 對應一份「全部 SEQ_FEATURES 的 DataFrame」快取
# key = id(ohlcv_cache)，避免 deep / breakout 重算同一份特徵
_FEATURE_CACHE: dict = {}


def load_model(path: str = MODEL_FILE, device: str = 'cuda',
               return_ckpt: bool = False):
    if not os.path.exists(path):
        return (None, None) if return_ckpt else None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_model(
            model_type=ckpt.get('model_type', 'gru'),
            n_features=len(ckpt['features']),
            hidden=ckpt.get('hidden', 64),
            layers=ckpt.get('layers', 2),
            dropout=0.0,
            n_horizons=len(ckpt['horizons']),
            window=ckpt.get('window', WINDOW),
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        _LOADED_CKPT_CACHE[path] = ckpt
        if return_ckpt:
            return model, ckpt
        return model
    except Exception as e:
        print(f"⚠ 載入 {path} 失敗: {e}")
        return (None, None) if return_ckpt else None


def get_loaded_ckpt(path: str = MODEL_FILE) -> Optional[dict]:
    return _LOADED_CKPT_CACHE.get(path)


def _get_feature_df(sid: str, df: pd.DataFrame, ohlcv_cache_id: int):
    """D16：對 (ohlcv_cache, sid) 快取 build_seq_features 結果，避免 deep/breakout 重算。"""
    bucket = _FEATURE_CACHE.setdefault(ohlcv_cache_id, {})
    if sid in bucket:
        return bucket[sid]
    dfc = df.copy()
    if 'ATR' not in dfc.columns:
        bt._add_indicators(dfc)
    feats = build_seq_features(dfc)
    bucket[sid] = (feats, dfc)
    return bucket[sid]


def clear_feature_cache(ohlcv_cache_id: Optional[int] = None):
    """釋放快取（呼叫者保有 ohlcv_cache 物件就能拿到對應 id）。"""
    if ohlcv_cache_id is None:
        _FEATURE_CACHE.clear()
    else:
        _FEATURE_CACHE.pop(ohlcv_cache_id, None)


def build_live_windows(
    stock_ids: Sequence[str],
    ohlcv_cache: Optional[dict] = None,
    period: str = '9mo',
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[pd.Timestamp], List[float]]:
    """建立推論用的最新視窗：每支股票輸出最後一個 60 天視窗。
    feature_cols 預設使用模組 SEQ_FEATURES；若模型是舊版只有 14 特徵，
    傳入 ckpt['features'] 可保持相容。

    D16：對相同 ohlcv_cache 物件，特徵運算結果會被快取共用（deep / breakout 都受惠）。
    """
    if ohlcv_cache is None:
        ohlcv_cache = bt._batch_download(list(stock_ids), period=period)

    use_cols = feature_cols or SEQ_FEATURES
    cache_id = id(ohlcv_cache)

    Xs, sids, dates, closes = [], [], [], []
    for sid in stock_ids:
        df = ohlcv_cache.get(sid)
        if df is None or len(df) < WINDOW + 10:
            continue
        feats, dfc = _get_feature_df(sid, df, cache_id)
        # 若 checkpoint 要的欄位本次沒算出（舊模型相容），補 0 避免炸裂
        missing = [c for c in use_cols if c not in feats.columns]
        if missing:
            for c in missing:
                feats = feats.assign(**{c: 0.0})
            # 寫回 cache 以便下次直接取
            _FEATURE_CACHE[cache_id][sid] = (feats, dfc)
        arr = feats[use_cols].values.astype(np.float32)
        idx = len(arr) - 1
        while idx >= WINDOW - 1:
            w = arr[idx - WINDOW + 1: idx + 1]
            if not np.isnan(w).any():
                Xs.append(w)
                sids.append(str(sid))
                dates.append(pd.Timestamp(dfc.index[idx]))
                closes.append(float(dfc['Close'].iloc[idx]))
                break
            idx -= 1

    if not Xs:
        return np.zeros((0, WINDOW, len(use_cols)), dtype=np.float32), [], [], []
    return np.stack(Xs), sids, dates, closes


def predict_scores(model: nn.Module, X: np.ndarray,
                   device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """回傳 (pred_returns[N, n_horizons], up_prob[N])"""
    if len(X) == 0:
        return np.zeros((0, len(HORIZONS))), np.zeros(0)
    model.eval()
    xb = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        pred_r, pred_c = model(xb)
        return pred_r.cpu().numpy(), torch.sigmoid(pred_c).cpu().numpy()


# ──────────────────────────────────────────────────────────
# 統一預測 helper：Tier A / D2 下游使用
# ──────────────────────────────────────────────────────────

def predict_all_for_stocks(
    stock_ids: Sequence[str],
    ohlcv_cache: Optional[dict] = None,
    period: str = '9mo',
    model_path: str = MODEL_FILE,
    device: Optional[str] = None,
) -> dict:
    """下游統一入口：給代號清單，回傳
        { sid: {close, pred_5d, pred_10d, pred_20d, up_prob,
                lower_20d, upper_20d, mae_20d, deep_score, target_price} }
    若模型不存在回傳 {}。
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_model(model_path, device=device, return_ckpt=True)
    if model is None:
        return {}
    features = ckpt.get('features', SEQ_FEATURES)
    horizons = ckpt.get('horizons', list(HORIZONS))
    conformal = ckpt.get('conformal') or {}
    X, sids, _dates, closes = build_live_windows(
        stock_ids, ohlcv_cache=ohlcv_cache, period=period,
        feature_cols=features,
    )
    if len(X) == 0:
        return {}
    pred_r, up_p = predict_scores(model, X, device=device)
    # ret_20d clip + normalize 做 deep_score
    ret20 = pred_r[:, -1]
    ret_clip = np.clip(ret20, -0.15, 0.30)
    if ret_clip.max() > ret_clip.min():
        ret_norm = (ret_clip - ret_clip.min()) / (ret_clip.max() - ret_clip.min()) * 100
    else:
        ret_norm = np.full_like(ret_clip, 50.0)
    deep_score = 0.4 * ret_norm + 0.6 * up_p * 100

    q_low = conformal.get('q_low') or [None] * len(horizons)
    q_high = conformal.get('q_high') or [None] * len(horizons)
    mae = conformal.get('mae') or [None] * len(horizons)

    out: dict = {}
    skipped_nan = 0
    for i, sid in enumerate(sids):
        p20 = float(pred_r[i, -1])
        c = float(closes[i])
        up = float(up_p[i])
        ds = float(deep_score[i])
        # B8：過濾 NaN / Inf 預測（防止下游 cohort/stacking 傳染）
        if not (np.isfinite(p20) and np.isfinite(up) and np.isfinite(ds) and np.isfinite(c)):
            skipped_nan += 1
            continue
        # Conformal interval on ret_20d
        lo = p20 + q_low[-1] if q_low[-1] is not None else None
        hi = p20 + q_high[-1] if q_high[-1] is not None else None
        out[str(sid)] = {
            'close': c,
            'pred_5d': float(pred_r[i, 0]),
            'pred_10d': float(pred_r[i, 1]) if pred_r.shape[1] > 2 else None,
            'pred_20d': p20,
            'up_prob': up,
            'lower_20d': float(lo) if lo is not None else None,
            'upper_20d': float(hi) if hi is not None else None,
            'mae_20d': float(mae[-1]) if mae[-1] is not None else None,
            'deep_score': ds,
            'target_price': round(c * (1 + p20), 2) if c > 0 else None,
            'lower_price': round(c * (1 + lo), 2) if (lo is not None and c > 0) else None,
            'upper_price': round(c * (1 + hi), 2) if (hi is not None and c > 0) else None,
        }
    if skipped_nan:
        print(f"  · 略過 {skipped_nan} 檔（NaN/Inf 預測）")
    # B7：模型推論完釋放 GPU 快取
    try:
        del model
        if device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def explain_prediction(
    model: nn.Module, x_window: np.ndarray,
    feature_names: List[str], target: str = 'cls',
    n_steps: int = 32, device: str = 'cuda',
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Tier D3: Integrated Gradients 手寫實作，回傳每個特徵的平均貢獻 top-K。
    target ∈ {'cls', 'ret20'}。"""
    if x_window.ndim == 2:
        x_window = x_window[None, ...]
    x = torch.from_numpy(x_window.astype(np.float32)).to(device)
    baseline = torch.zeros_like(x)
    steps = torch.linspace(0.0, 1.0, n_steps, device=device).view(-1, 1, 1, 1)
    interp = baseline + steps * (x - baseline)  # (S, B, T, F)
    interp = interp.view(-1, x.size(1), x.size(2))
    interp.requires_grad_(True)

    pred_r, pred_c = model(interp)
    if target == 'cls':
        out_sum = pred_c.sum()
    else:
        out_sum = pred_r[:, -1].sum()
    grads = torch.autograd.grad(out_sum, interp)[0]
    grads = grads.view(n_steps, x.size(0), x.size(1), x.size(2))
    avg_grads = grads.mean(dim=0)  # (B, T, F)
    attributions = (x - baseline) * avg_grads  # IG
    # 沿時間軸平均 → 每特徵一個貢獻值
    contrib = attributions.mean(dim=1).squeeze(0).detach().cpu().numpy()
    pairs = sorted(
        [(feature_names[i], float(contrib[i])) for i in range(len(feature_names))],
        key=lambda x: abs(x[1]), reverse=True,
    )
    return pairs[:top_k]


def predict_ranks(model: nn.Module, stock_ids: Sequence[str],
                  top_n: int = 30, period: str = '9mo',
                  device: str = 'cuda',
                  feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    X, sids, dates, closes = build_live_windows(stock_ids, period=period,
                                                feature_cols=feature_cols)
    if len(X) == 0:
        return pd.DataFrame()
    pred_r, up_p = predict_scores(model, X, device=device)
    # 排名分數
    ret20 = pred_r[:, -1]
    # 裁切後線性正規化到 0-100，與分類機率（0-100）加權
    ret_clip = np.clip(ret20, -0.15, 0.30)
    if ret_clip.max() > ret_clip.min():
        ret_norm = (ret_clip - ret_clip.min()) / (ret_clip.max() - ret_clip.min()) * 100
    else:
        ret_norm = np.full_like(ret_clip, 50.0)
    deep_score = 0.4 * ret_norm + 0.6 * up_p * 100

    df = pd.DataFrame({
        'stock_id': sids,
        'date': dates,
        'close': closes,
        'pred_5d': pred_r[:, 0],
        'pred_10d': pred_r[:, 1],
        'pred_20d': pred_r[:, -1],
        'up_prob': up_p,
        'deep_score': deep_score,
    }).sort_values('deep_score', ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def _demo_stocks() -> List[str]:
    # 與 ml_ranker 相同，以維持可比較性
    try:
        from ml_ranker import _demo_stocks as _dm
        return _dm()
    except Exception:
        return [
            '2330', '2317', '2454', '2308', '2881', '2882', '2303', '3711',
            '2412', '2886', '1301', '2891', '3008', '2357', '6505', '2002',
            '2884', '1303', '2885', '1216', '2892', '2207', '2880', '2887',
            '5871', '2890', '2609', '1590', '3045', '2912', '1326', '2603',
            '2610', '2615', '3231', '2379', '6669', '3034', '4938', '2408',
            '3037', '2327', '2474', '1402', '1102', '2801', '2834',
            '2345', '2301', '2377', '2354', '2395', '2382', '2383', '3706',
            '3017', '8046', '6488', '8454',
        ]


def _resolve_stocks(choice: str, extra: List[str]) -> List[str]:
    if choice == 'auto':
        try:
            import Strategy_twe as stw
            all_data = stw.get_recent_institutional()
            return all_data['證券代號'].astype(str).str.strip().unique().tolist()
        except Exception as e:
            print(f"  ⚠ auto 載入失敗 ({e})，改用 demo")
            return _demo_stocks()
    if choice == 'demo':
        return _demo_stocks()
    return extra or _demo_stocks()


def cmd_train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stock_ids = _resolve_stocks(args.stocks, args.extra)
    X, Yr, Yc, D, _ = build_training_dataset(stock_ids, period=args.period)
    cfg = TrainConfig(
        hidden=args.hidden, layers=args.layers,
        dropout=args.dropout, lr=args.lr,
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience,
        model_type=getattr(args, 'model', 'gru'),
        n_heads=getattr(args, 'n_heads', 4),
    )
    model, history = train_model(X, Yr, Yc, D, cfg, device=device)
    save_model(model, args.model_out, cfg, conformal=history.get('conformal'))
    # history 中有 numpy array 要轉純 python
    hist_serializable = {
        'train': history.get('train', []),
        'valid': history.get('valid', []),
        'valid_ic': history.get('valid_ic', []),
        'conformal': history.get('conformal', {}),
        'calibration': history.get('calibration', {}),
    }
    with open('deep_model_history.json', 'w', encoding='utf-8') as f:
        json.dump(hist_serializable, f, ensure_ascii=False, indent=2)


def cmd_predict(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_model(args.model_out, device=device, return_ckpt=True)
    if model is None:
        raise SystemExit(f"❌ 模型 {args.model_out} 不存在，請先 train")
    stock_ids = _resolve_stocks(args.stocks, args.extra)
    result = predict_ranks(model, stock_ids, top_n=args.top_n,
                           period=args.period, device=device,
                           feature_cols=ckpt.get('features'))
    if result.empty:
        print("❌ 無可用特徵")
        return
    print(f"\n===== Deep TOP {args.top_n} =====")
    print(result.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description='PyTorch GPU 序列深度模型')
    sub = p.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train')
    pt.add_argument('--stocks', default='demo', choices=['auto', 'demo', 'custom'])
    pt.add_argument('--extra', nargs='*', default=[])
    pt.add_argument('--period', default='3y')
    pt.add_argument('--epochs', type=int, default=40)
    pt.add_argument('--batch-size', type=int, default=256)
    pt.add_argument('--hidden', type=int, default=64)
    pt.add_argument('--layers', type=int, default=2)
    pt.add_argument('--dropout', type=float, default=0.25)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--patience', type=int, default=8)
    pt.add_argument('--model-out', default=MODEL_FILE)
    pt.add_argument('--model', default='gru', choices=['gru', 'transformer'],
                    help='模型架構：gru (預設) 或 transformer (Tier B2)')
    pt.add_argument('--n-heads', type=int, default=4,
                    help='Transformer attention head 數')
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser('predict')
    pp.add_argument('--stocks', default='auto', choices=['auto', 'demo', 'custom'])
    pp.add_argument('--extra', nargs='*', default=[])
    pp.add_argument('--period', default='9mo')
    pp.add_argument('--top-n', type=int, default=30)
    pp.add_argument('--model-out', default=MODEL_FILE)
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

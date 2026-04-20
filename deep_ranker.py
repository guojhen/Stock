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

# 特徵欄位（每個時間步一個 vector）
SEQ_FEATURES = [
    'logret_1d',
    'close_ma5', 'close_ma20', 'close_ma60',
    'vol_ratio',
    'rsi_n', 'k_n', 'd_n',
    'macd_hist_n', 'bb_pos', 'atr_pct', 'hl_range',
    'dist_52w', 'log_vol_z',
]


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
    train_pct: float = 0.8
    reg_weight: float = 1.0
    cls_weight: float = 0.5
    huber_delta: float = 0.05
    grad_clip: float = 1.0


def train_model(X, Yr, Yc, D, cfg: TrainConfig,
                device: str = 'cuda') -> Tuple[nn.Module, dict]:
    # 時間序列切分
    order = np.argsort(D)
    X, Yr, Yc, D = X[order], Yr[order], Yc[order], D[order]
    unique_dates = np.unique(D)
    split_date = unique_dates[int(len(unique_dates) * cfg.train_pct)]
    train_mask = D < split_date
    valid_mask = ~train_mask

    print(f"\n===== 訓練 PriceGRU =====")
    print(f"  train: {train_mask.sum():,}  valid: {valid_mask.sum():,}  "
          f"(split at {pd.Timestamp(split_date).date()})")

    train_ds = SlidingWindowDataset(X[train_mask], Yr[train_mask], Yc[train_mask],
                                    D[train_mask], None)
    valid_ds = SlidingWindowDataset(X[valid_mask], Yr[valid_mask], Yc[valid_mask],
                                    D[valid_mask], None)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size * 2,
                              shuffle=False)

    model = PriceGRU(
        n_features=X.shape[2], hidden=cfg.hidden,
        layers=cfg.layers, dropout=cfg.dropout, n_horizons=Yr.shape[1],
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
    return model, history


# ──────────────────────────────────────────────────────────
# 儲存 / 載入 / 推論
# ──────────────────────────────────────────────────────────

def save_model(model: nn.Module, path: str = MODEL_FILE, cfg: Optional[TrainConfig] = None):
    torch.save({
        'state_dict': model.state_dict(),
        'features': SEQ_FEATURES,
        'window': WINDOW,
        'horizons': list(HORIZONS),
        'hidden': cfg.hidden if cfg else 64,
        'layers': cfg.layers if cfg else 2,
    }, path)
    meta = {
        'features': SEQ_FEATURES, 'window': WINDOW,
        'horizons': list(HORIZONS),
        'trained_at': pd.Timestamp.now().isoformat(),
    }
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  模型已儲存: {path}")


def load_model(path: str = MODEL_FILE, device: str = 'cuda') -> Optional[nn.Module]:
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = PriceGRU(
            n_features=len(ckpt['features']),
            hidden=ckpt.get('hidden', 64),
            layers=ckpt.get('layers', 2),
            dropout=0.0,
            n_horizons=len(ckpt['horizons']),
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"⚠ 載入 {path} 失敗: {e}")
        return None


def build_live_windows(
    stock_ids: Sequence[str],
    ohlcv_cache: Optional[dict] = None,
    period: str = '9mo',
) -> Tuple[np.ndarray, List[str], List[pd.Timestamp], List[float]]:
    """建立推論用的最新視窗：每支股票輸出最後一個 60 天視窗。"""
    if ohlcv_cache is None:
        ohlcv_cache = bt._batch_download(list(stock_ids), period=period)

    Xs, sids, dates, closes = [], [], [], []
    for sid in stock_ids:
        df = ohlcv_cache.get(sid)
        if df is None or len(df) < WINDOW + 10:
            continue
        dfc = df.copy()
        if 'ATR' not in dfc.columns:
            bt._add_indicators(dfc)
        feats = build_seq_features(dfc)
        arr = feats[SEQ_FEATURES].values.astype(np.float32)
        # 從尾端往前找一個沒有 NaN 的視窗
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
        return np.zeros((0, WINDOW, len(SEQ_FEATURES)), dtype=np.float32), [], [], []
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


def predict_ranks(model: nn.Module, stock_ids: Sequence[str],
                  top_n: int = 30, period: str = '9mo',
                  device: str = 'cuda') -> pd.DataFrame:
    X, sids, dates, closes = build_live_windows(stock_ids, period=period)
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
    )
    model, history = train_model(X, Yr, Yc, D, cfg, device=device)
    save_model(model, args.model_out, cfg)
    with open('deep_model_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def cmd_predict(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_out, device=device)
    if model is None:
        raise SystemExit(f"❌ 模型 {args.model_out} 不存在，請先 train")
    stock_ids = _resolve_stocks(args.stocks, args.extra)
    result = predict_ranks(model, stock_ids, top_n=args.top_n,
                           period=args.period, device=device)
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

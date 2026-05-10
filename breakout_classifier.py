"""Tier B1：突破分類器 (Breakout Classifier)

目標：回答「這檔 T+10 日內最高價 ≥ +10% 的機率是多少？」
比 deep_ranker 的迴歸任務更直接，訓練樣本稀少 → 用 **Focal Loss** 應付不平衡。

輸入：與 deep_ranker 相同的 60-day × F 特徵視窗（重用 build_seq_features）
輸出：單一 sigmoid 機率 `p_breakout ∈ [0, 1]`

使用：
    python breakout_classifier.py train --stocks auto --period 3y --epochs 30
    python breakout_classifier.py predict --stocks auto
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import backtest as bt
import deep_ranker as dr


MODEL_FILE = 'breakout_model.pt'
META_FILE = 'breakout_model_meta.json'

WINDOW = dr.WINDOW
BREAKOUT_WINDOW = 10        # 10 個交易日內
BREAKOUT_THRESHOLD = 0.10   # ≥ +10% 視為正樣本
MIN_HISTORY = WINDOW + BREAKOUT_WINDOW + 20


# ──────────────────────────────────────────────────────────
# Label builder
# ──────────────────────────────────────────────────────────

def build_breakout_labels(df: pd.DataFrame) -> pd.Series:
    """對每個 end day：未來 BREAKOUT_WINDOW 天內最高價 / 當日收盤 - 1 ≥ THRESHOLD"""
    c = df['Close']
    h = df['High']
    # 未來 N 天的最高（不含當天）
    future_max = h.shift(-1).rolling(BREAKOUT_WINDOW).max().shift(-(BREAKOUT_WINDOW - 1))
    ret = future_max / c - 1
    label = (ret >= BREAKOUT_THRESHOLD).astype(np.float32)
    label[ret.isna()] = np.nan
    return label


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

class BreakoutDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def build_training_dataset(stock_ids, period='3y'):
    print(f"\n===== Breakout 訓練集 ({len(stock_ids)} 股, {period}) =====")
    t0 = time.time()
    cache = bt._batch_download(list(stock_ids), period=period)
    print(f"  成功下載 {len(cache)} 支")

    xs, ys, ds = [], [], []
    for sid, df in cache.items():
        if len(df) < MIN_HISTORY:
            continue
        bt._add_indicators(df)
        feats = dr.build_seq_features(df)
        labels = build_breakout_labels(df).values

        feat_arr = feats[dr.SEQ_FEATURES].values.astype(np.float32)
        for end in range(WINDOW - 1, len(feat_arr)):
            lbl = labels[end]
            if np.isnan(lbl):
                continue
            w = feat_arr[end - WINDOW + 1: end + 1]
            if np.isnan(w).any():
                continue
            xs.append(w)
            ys.append(lbl)
            ds.append(df.index[end])

    if not xs:
        raise RuntimeError("資料不足")
    X = np.stack(xs)
    Y = np.array(ys, dtype=np.float32)
    D = np.array(ds)
    pos_ratio = Y.mean()
    print(f"  共 {len(X):,} 個視窗, 正樣本比例={pos_ratio:.2%}, "
          f"耗時 {time.time()-t0:.1f}s")
    return X, Y, D


# ──────────────────────────────────────────────────────────
# Model (小型 GRU) + Focal Loss
# ──────────────────────────────────────────────────────────

class BreakoutGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden,
                          num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 48), nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.norm(out[:, -1, :])
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


class FocalLoss(nn.Module):
    """FocalLoss(γ=2, α=auto)：適合極度不平衡類別。
    C2 強化：alpha 改為「正樣本權重」（一般 = 1 - pos_ratio），
    使少數類獲得更大 loss 權重；α 過小會壓抑正類，α 過大會誤報暴增。
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pw, reduction='none')
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = a_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


# ── C2：自動 tune alpha + 找最佳 F1 閾值 ──

def _auto_focal_alpha(y_train: np.ndarray) -> float:
    """C2：依正樣本比例反比設 alpha。
    pos_ratio=0.10 → alpha=0.90；pos_ratio=0.30 → alpha=0.70。
    限制範圍 [0.25, 0.85] 避免極端。"""
    pos_ratio = float(np.clip(y_train.mean(), 0.01, 0.99))
    alpha = 1.0 - pos_ratio
    return float(np.clip(alpha, 0.25, 0.85))


def _calibrate_threshold(probs: np.ndarray, y_true: np.ndarray):
    """C2：在 valid set 上掃 0.05~0.95 找最佳 F1 / 同時記錄 P@K。
    回傳 dict: {'best_thr', 'best_f1', 'precision', 'recall',
                'p_at_5', 'p_at_10', 'p_at_20'}
    """
    y = y_true.astype(int)
    best = {'best_thr': 0.5, 'best_f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best['best_f1']:
            best = {'best_thr': float(thr), 'best_f1': float(f1),
                    'precision': float(prec), 'recall': float(rec)}
    # Precision @ Top-K%
    order = np.argsort(-probs)
    for k_pct in (5, 10, 20):
        k = max(1, int(len(probs) * k_pct / 100))
        topk = order[:k]
        prec_k = float(y[topk].sum() / k)
        best[f'p_at_{k_pct}'] = prec_k
    return best


# ──────────────────────────────────────────────────────────
# 訓練
# ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    hidden: int = 48
    layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 30
    patience: int = 6
    train_pct: float = 0.8
    focal_alpha: float = -1.0       # C2：< 0 表示自動依正樣本比例 tune
    focal_gamma: float = 2.0
    grad_clip: float = 1.0
    use_pos_weight: bool = True     # C2：BCE 內也加 pos_weight 進一步平衡


def train_model(X, Y, D, cfg: TrainConfig, device='cuda'):
    order = np.argsort(D)
    X, Y, D = X[order], Y[order], D[order]
    unique_dates = np.unique(D)
    split_date = unique_dates[int(len(unique_dates) * cfg.train_pct)]
    tr_mask = D < split_date
    va_mask = ~tr_mask

    print(f"  train: {tr_mask.sum():,}  valid: {va_mask.sum():,}  "
          f"(split at {pd.Timestamp(split_date).date()})")
    pos_ratio_tr = float(Y[tr_mask].mean())
    pos_ratio_va = float(Y[va_mask].mean())
    print(f"  train 正比例: {pos_ratio_tr:.2%}  valid 正比例: {pos_ratio_va:.2%}")

    tr_ds = BreakoutDataset(X[tr_mask], Y[tr_mask])
    va_ds = BreakoutDataset(X[va_mask], Y[va_mask])
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size * 2)

    model = BreakoutGRU(n_features=X.shape[2], hidden=cfg.hidden,
                        layers=cfg.layers, dropout=cfg.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                               eta_min=cfg.lr * 0.05)

    # C2：alpha 自動 tune（依正樣本比例反比）+ pos_weight = (1-p)/p
    if cfg.focal_alpha is None or cfg.focal_alpha < 0:
        alpha = _auto_focal_alpha(Y[tr_mask])
    else:
        alpha = cfg.focal_alpha
    pos_w = ((1 - pos_ratio_tr) / max(pos_ratio_tr, 1e-3)) if cfg.use_pos_weight else None
    print(f"  C2：FocalLoss α={alpha:.3f} γ={cfg.focal_gamma} "
          f"pos_weight={pos_w:.2f}" if pos_w else
          f"  C2：FocalLoss α={alpha:.3f} γ={cfg.focal_gamma}")
    loss_fn = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma, pos_weight=pos_w)

    print(f"  裝置: {device} | params={sum(p.numel() for p in model.parameters()):,}")

    best_val = float('inf')
    best_state = None
    patience = cfg.patience
    history = {'train': [], 'valid': [], 'valid_auc': []}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        tr_losses = []
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lg = model(xb)
            loss = loss_fn(lg, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_losses.append(loss.item())
        sch.step()

        model.eval()
        v_losses, v_preds, v_true = [], [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                lg = model(xb)
                v_losses.append(loss_fn(lg, yb).item())
                v_preds.append(torch.sigmoid(lg).cpu().numpy())
                v_true.append(yb.cpu().numpy())
        v_preds = np.concatenate(v_preds)
        v_true = np.concatenate(v_true)

        # 簡易 AUC（rank-based）
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(v_true, v_preds)) if v_true.std() > 0 else 0.5
        except Exception:
            auc = 0.5

        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(v_losses))
        history['train'].append(tr_loss)
        history['valid'].append(va_loss)
        history['valid_auc'].append(auc)

        improved = va_loss < best_val - 1e-5
        if improved:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.patience
            mark = ' *'
        else:
            patience -= 1
            mark = ''

        print(f"  ep{ep:>3d} | tr {tr_loss:.4f} | va {va_loss:.4f} | AUC {auc:.3f} | "
              f"{time.time()-t0:.1f}s{mark}")
        if patience <= 0:
            print(f"  early stop (best va {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # C2：在 valid set 上做最終閾值校準（找最佳 F1 + P@K）
    model.eval()
    v_probs, v_true = [], []
    with torch.no_grad():
        for xb, yb in va_ld:
            xb, yb = xb.to(device), yb.to(device)
            v_probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            v_true.append(yb.cpu().numpy())
    v_probs = np.concatenate(v_probs)
    v_true = np.concatenate(v_true)
    calib = _calibrate_threshold(v_probs, v_true)
    history['calibration'] = calib
    history['focal_alpha_used'] = alpha
    history['pos_weight_used'] = pos_w
    print(f"  C2 閾值校準（valid）: best_thr={calib['best_thr']:.2f}  "
          f"F1={calib['best_f1']:.3f}  precision={calib['precision']:.3f}  recall={calib['recall']:.3f}")
    print(f"  Precision @ Top 5%/10%/20% = "
          f"{calib['p_at_5']:.2%}/{calib['p_at_10']:.2%}/{calib['p_at_20']:.2%}")

    return model, history


# ──────────────────────────────────────────────────────────
# 儲存 / 載入 / 推論
# ──────────────────────────────────────────────────────────

def save_model(model, path=MODEL_FILE, cfg: Optional[TrainConfig] = None,
               calibration: Optional[dict] = None):
    """C2：保存模型時帶上 calibration（best_thr / Precision@K）。"""
    calib = calibration or {}
    torch.save({
        'state_dict': model.state_dict(),
        'features': dr.SEQ_FEATURES,
        'window': WINDOW,
        'hidden': cfg.hidden if cfg else 48,
        'layers': cfg.layers if cfg else 2,
        'breakout_window': BREAKOUT_WINDOW,
        'breakout_threshold': BREAKOUT_THRESHOLD,
        'calibration': calib,                # C2：保存決策閾值
    }, path)
    meta = {
        'features': dr.SEQ_FEATURES,
        'window': WINDOW,
        'breakout_window': BREAKOUT_WINDOW,
        'breakout_threshold': BREAKOUT_THRESHOLD,
        'calibration': calib,
        'trained_at': pd.Timestamp.now().isoformat(),
    }
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  突破分類器已儲存: {path}")


def load_model(path=MODEL_FILE, device='cuda'):
    if not os.path.exists(path):
        return None, None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = BreakoutGRU(
            n_features=len(ckpt['features']),
            hidden=ckpt.get('hidden', 48),
            layers=ckpt.get('layers', 2),
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model, ckpt
    except Exception as e:
        print(f"⚠ 載入 {path} 失敗: {e}")
        return None, None


def predict_breakout(stock_ids, ohlcv_cache=None, period='9mo',
                     model_path=MODEL_FILE, device=None,
                     return_calibration: bool = False):
    """回傳 {sid: p_breakout} (0~1)；
    C2：return_calibration=True 時，多回傳 (probs_dict, calibration_dict)。
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_model(model_path, device=device)
    if model is None:
        return ({}, {}) if return_calibration else {}
    features = ckpt.get('features', dr.SEQ_FEATURES)
    calibration = ckpt.get('calibration', {}) or {}
    X, sids, _d, _c = dr.build_live_windows(
        stock_ids, ohlcv_cache=ohlcv_cache, period=period,
        feature_cols=features,
    )
    if len(X) == 0:
        return ({}, calibration) if return_calibration else {}
    model.eval()
    xb = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
    out = {}
    for s, p in zip(sids, probs):
        pf = float(p)
        if np.isfinite(pf):
            out[str(s)] = pf
    try:
        del model
        if device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return (out, calibration) if return_calibration else out


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def _resolve_stocks(choice, extra):
    if choice == 'auto':
        try:
            import Strategy_twe as stw
            all_data = stw.get_recent_institutional()
            return all_data['證券代號'].astype(str).str.strip().unique().tolist()
        except Exception:
            return dr._demo_stocks()
    if choice == 'demo':
        return dr._demo_stocks()
    return extra or dr._demo_stocks()


def cmd_train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stocks = _resolve_stocks(args.stocks, args.extra)
    X, Y, D = build_training_dataset(stocks, period=args.period)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      hidden=args.hidden, lr=args.lr,
                      patience=args.patience,
                      focal_alpha=args.focal_alpha,
                      focal_gamma=args.focal_gamma,
                      use_pos_weight=not args.no_pos_weight)
    model, hist = train_model(X, Y, D, cfg, device=device)
    save_model(model, args.model_out, cfg, calibration=hist.get('calibration'))
    with open('breakout_model_history.json', 'w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=False, indent=2, default=float)


def cmd_predict(args):
    stocks = _resolve_stocks(args.stocks, args.extra)
    probs = predict_breakout(stocks, period=args.period)
    if not probs:
        print("❌ 無法預測（模型不存在或資料不足）")
        return
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print(f"\n===== 突破機率 TOP {args.top_n} =====")
    for sid, p in ranked[:args.top_n]:
        print(f"  {sid}  {p*100:5.1f}%")


def main():
    p = argparse.ArgumentParser(description='突破分類器 (Focal Loss + GRU)')
    sub = p.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train')
    pt.add_argument('--stocks', default='demo', choices=['auto', 'demo', 'custom'])
    pt.add_argument('--extra', nargs='*', default=[])
    pt.add_argument('--period', default='3y')
    pt.add_argument('--epochs', type=int, default=30)
    pt.add_argument('--batch-size', type=int, default=256)
    pt.add_argument('--hidden', type=int, default=48)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--patience', type=int, default=6)
    pt.add_argument('--focal-alpha', type=float, default=-1.0,
                    help='-1 (預設) 表示自動依正樣本比例調整')
    pt.add_argument('--focal-gamma', type=float, default=2.0)
    pt.add_argument('--no-pos-weight', action='store_true',
                    help='關掉 BCE pos_weight（仍保留 Focal alpha）')
    pt.add_argument('--model-out', default=MODEL_FILE)
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser('predict')
    pp.add_argument('--stocks', default='auto', choices=['auto', 'demo', 'custom'])
    pp.add_argument('--extra', nargs='*', default=[])
    pp.add_argument('--period', default='9mo')
    pp.add_argument('--top-n', type=int, default=30)
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

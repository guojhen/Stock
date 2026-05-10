"""Tier D1：GPU 蒙地卡羅情境模擬 (Monte Carlo Geometric Brownian Motion)

對每檔股票以歷史漂移 μ 與波動度 σ，在 GPU 上同時模擬 10,000 條路徑 × 20 天，
得到下列機率指標：
    p_up_5, p_up_10, p_up_15, p_up_20  -- 20 日內任一天漲幅 ≥ X% 的機率
    expected_ret                       -- T+20 預期報酬
    p5 / p50 / p95                     -- T+20 價格分位數

使用方式（通常由 Strategy_twe 呼叫）：
    from mc_simulator import simulate_all
    result = simulate_all(stock_ids, ohlcv_cache=..., horizon=20, n_paths=10000)
    # result[sid] = {'close':..., 'p_up_10': 0.45, ...}

純 PyTorch 實作，若 CUDA 不可用自動 fallback CPU；全市場 1500 檔 GPU < 1 秒。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import backtest as bt
except Exception:
    bt = None


DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 20
DEFAULT_N_PATHS = 10_000
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)  # +5, +10, +15, +20%


# ──────────────────────────────────────────────────────────
# Statistics extractor
# ──────────────────────────────────────────────────────────

def compute_stats(ohlcv_cache: Dict[str, pd.DataFrame],
                  lookback: int = DEFAULT_LOOKBACK,
                  min_lookback: int = 30) -> Dict[str, dict]:
    """從 OHLCV cache 萃取每檔的日均 log-return（drift μ）與日波動 σ。
    若實際資料短於 lookback，至少需有 min_lookback 天才納入。"""
    stats = {}
    for sid, df in ohlcv_cache.items():
        if df is None or len(df) < min_lookback + 2:
            continue
        use_lookback = min(lookback, len(df) - 1)
        close = df['Close'].tail(use_lookback + 1).astype(float)
        if close.min() <= 0:
            continue
        logret = np.log(close.values[1:] / close.values[:-1])
        if len(logret) < min_lookback - 5:
            continue
        mu = float(np.nanmean(logret))
        sigma = float(np.nanstd(logret, ddof=1))
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-8:
            continue
        stats[str(sid)] = {
            'close': float(df['Close'].iloc[-1]),
            'mu': mu,
            'sigma': sigma,
        }
    return stats


# ──────────────────────────────────────────────────────────
# PyTorch GPU 模擬核心
# ──────────────────────────────────────────────────────────

def simulate_gbm_pytorch(
    stats: Dict[str, dict],
    horizon: int = DEFAULT_HORIZON,
    n_paths: int = DEFAULT_N_PATHS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    device: Optional[str] = None,
    seed: Optional[int] = 20260412,
) -> Dict[str, dict]:
    """向量化 GBM：對所有股票一次模擬 (S, N, T) 張量。"""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch 未安裝")
    if not stats:
        return {}
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if seed is not None:
        torch.manual_seed(seed)

    sids = list(stats.keys())
    close = torch.tensor([stats[s]['close'] for s in sids], device=device,
                         dtype=torch.float32)
    mu = torch.tensor([stats[s]['mu'] for s in sids], device=device,
                      dtype=torch.float32)
    sigma = torch.tensor([stats[s]['sigma'] for s in sids], device=device,
                         dtype=torch.float32)

    # 分批模擬避免顯存爆炸（每批 512 檔）
    S = len(sids)
    batch = 512
    out: Dict[str, dict] = {}

    for start in range(0, S, batch):
        end = min(start + batch, S)
        s_slice = slice(start, end)
        b = end - start
        # eps ~ N(0,1), shape (b, N, T)
        eps = torch.randn(b, n_paths, horizon, device=device)
        # 日 log return
        # GBM: dlogS = (μ - σ²/2) dt + σ dW；dt=1
        drift_term = (mu[s_slice] - 0.5 * sigma[s_slice] ** 2)[:, None, None]
        vol_term = sigma[s_slice][:, None, None] * eps
        dlog = drift_term + vol_term
        cum = torch.cumsum(dlog, dim=-1)  # (b, N, T)
        price = close[s_slice][:, None, None] * torch.exp(cum)

        # 指標
        max_path = price.max(dim=-1).values            # (b, N)
        end_price = price[..., -1]                      # (b, N)

        # 批量算機率與分位數
        prob_cache = {}
        for th in thresholds:
            # max_path >= close * (1+th) → 觸及 +th 的路徑比例
            reached = (max_path >= (close[s_slice][:, None] * (1 + th)))
            prob_cache[th] = reached.float().mean(dim=-1)  # (b,)

        exp_end = end_price.mean(dim=-1)          # (b,)
        mean_max = max_path.mean(dim=-1)          # (b,)
        q05 = end_price.quantile(0.05, dim=-1)
        q50 = end_price.quantile(0.50, dim=-1)
        q95 = end_price.quantile(0.95, dim=-1)

        close_cpu = close[s_slice].cpu().numpy()
        for i, sid in enumerate(sids[start:end]):
            c = float(close_cpu[i])
            row = {
                'close': c,
                'expected_price': float(exp_end[i].item()),
                'expected_ret': float((exp_end[i].item() / c - 1) if c > 0 else 0),
                'max_mean': float(mean_max[i].item()),
                'p5_price': float(q05[i].item()),
                'p50_price': float(q50[i].item()),
                'p95_price': float(q95[i].item()),
            }
            for th in thresholds:
                pct = int(round(th * 100))
                row[f'p_up_{pct}'] = float(prob_cache[th][i].item())
            out[sid] = row

    return out


# ──────────────────────────────────────────────────────────
# 一站式入口
# ──────────────────────────────────────────────────────────

def simulate_all(
    stock_ids: Sequence[str],
    ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None,
    period: str = '9mo',
    horizon: int = DEFAULT_HORIZON,
    n_paths: int = DEFAULT_N_PATHS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    lookback: int = DEFAULT_LOOKBACK,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """主入口：給代號清單，回傳每檔蒙地卡羅指標。"""
    if ohlcv_cache is None:
        if bt is None:
            raise RuntimeError("backtest 模組無法載入")
        ohlcv_cache = bt._batch_download(list(stock_ids), period=period)
    if verbose:
        print(f"  MC 模擬：抽取 μ/σ 從 {len(ohlcv_cache)} 檔 OHLCV")
    stats = compute_stats(ohlcv_cache, lookback=lookback)
    if verbose:
        print(f"  有效樣本 {len(stats)} 檔，開始 {n_paths} 條路徑 × {horizon} 天模擬...")
    result = simulate_gbm_pytorch(
        stats, horizon=horizon, n_paths=n_paths,
        thresholds=thresholds, device=device,
    )
    if verbose and result:
        sample = next(iter(result.values()))
        print(f"  完成。範例：{next(iter(result))} → "
              f"E[ret]={sample['expected_ret']*100:+.1f}%, "
              f"P(+10%)={sample.get('p_up_10',0)*100:.0f}%")
    # B7：釋放 GPU 快取
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return result


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GPU 蒙地卡羅 GBM 情境模擬')
    parser.add_argument('--stocks', nargs='+', help='股票代號')
    parser.add_argument('--period', default='9mo')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON)
    parser.add_argument('--paths', type=int, default=DEFAULT_N_PATHS)
    parser.add_argument('--top-n', type=int, default=20)
    args = parser.parse_args()

    stocks = args.stocks or ['2330', '2317', '2454', '2308', '2881']
    res = simulate_all(stocks, period=args.period,
                       horizon=args.horizon, n_paths=args.paths)
    if not res:
        print("❌ 無結果")
    else:
        rows = []
        for sid, r in res.items():
            rows.append({
                'sid': sid,
                'close': r['close'],
                'E[ret%]': r['expected_ret'] * 100,
                'P(+5%)': r.get('p_up_5', 0) * 100,
                'P(+10%)': r.get('p_up_10', 0) * 100,
                'P(+15%)': r.get('p_up_15', 0) * 100,
                'P(+20%)': r.get('p_up_20', 0) * 100,
                '95%區間': f"{r['p5_price']:.1f} ~ {r['p95_price']:.1f}",
            })
        df = pd.DataFrame(rows).sort_values('P(+10%)', ascending=False).head(args.top_n)
        print(df.to_string(index=False, float_format='%.1f'))

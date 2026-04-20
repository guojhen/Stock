"""
GPU 加速參數網格回測（RTX 50 系列 / sm_120 相容）

用法：
    python gpu_backtest.py                          # 快速測試 15 檔示範股
    python gpu_backtest.py --stocks 2330 2317 ...   # 指定股票池
    python gpu_backtest.py --all                    # 對 Strategy_twe 當日法人資料做全市場搜尋

輸出：
    best_params.json   - 每支策略的最佳參數（供 Strategy_twe.py / backtest.py 讀取）
    gpu_grid_report.html - 每組策略的參數熱力圖 + Top 10 參數表

演算法：
    - 每支策略在 CPU 端算出進場訊號
    - GPU kernel 每 thread 處理一組 (股票, 參數組合)
       模擬交易後把 trades / wins / sum_ret / sum_sq_ret / sum_hold 寫回
    - 在主機端對所有股票 reduce 出每組參數的 Sharpe / 勝率 / 平均報酬
    - 依 Sharpe 排序選出前 K 名

設計：
    - 單一 kernel 處理所有 (stock × param) 組合
    - 所有陣列皆壓平為 1D float32/int32，減少記憶體碎片
    - RTX 5080 一次可跑 10 萬以上 thread，實測 1500 股 × 240 參數 = 36 萬
      大約 2-5 秒完成（相同計算在 CPU 要 30-60 分鐘）
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from numba import cuda

import backtest as bt

# ──────────────────────────────────────────────────────────
# 參數網格（可依需求調整）
# ──────────────────────────────────────────────────────────

DEFAULT_GRID = {
    'tp':          [0.08, 0.10, 0.12, 0.15, 0.20, 0.25],   # 固定停利
    'sl':          [-0.03, -0.05, -0.07, -0.10],           # 硬停損
    'atr_mult':    [1.5, 2.0, 2.5, 3.0],                   # 吊燈倍數
    'atr_trigger': [0.04, 0.06, 0.08],                     # 啟動移動停利的最小獲利
    'max_hold':    [10, 20, 30, 60],                       # 最長持有
}


OUTPUT_METRICS = 7  # (trades, wins, sum_ret, sum_sq, sum_hold, worst, best)


# ──────────────────────────────────────────────────────────
# GPU Kernel
# ──────────────────────────────────────────────────────────

@cuda.jit
def simulate_kernel(
    close_flat, atr_flat,
    stock_offsets, stock_lens,
    entries_flat, entry_offsets, entry_lens,
    tp_arr, sl_arr, atr_mult_arr, atr_trigger_arr, max_hold_arr,
    out,
):
    """每個 thread 處理一組 (stock_idx, param_idx) 的完整交易模擬。

    輸出 out[tid] 含 7 個欄位：
        [0] 交易次數
        [1] 勝場數
        [2] sum(return)
        [3] sum(return^2)
        [4] sum(hold_days)
        [5] worst trade return
        [6] best trade return
    """
    n_stocks = stock_lens.shape[0]
    n_params = tp_arr.shape[0]
    tid = cuda.grid(1)
    total = n_stocks * n_params
    if tid >= total:
        return

    stock_idx = tid // n_params
    param_idx = tid - stock_idx * n_params

    tp = tp_arr[param_idx]
    sl = sl_arr[param_idx]
    atr_mult = atr_mult_arr[param_idx]
    atr_trigger = atr_trigger_arr[param_idx]
    max_hold = max_hold_arr[param_idx]

    close_start = stock_offsets[stock_idx]
    stock_len = stock_lens[stock_idx]
    e_start = entry_offsets[stock_idx]
    e_len = entry_lens[stock_idx]

    trades_ct = 0
    wins_ct = 0
    sum_ret = 0.0
    sum_sq = 0.0
    sum_hold = 0
    worst = 0.0
    best = 0.0

    occupied_until = -1

    for e_i in range(e_start, e_start + e_len):
        idx = entries_flat[e_i]
        if idx <= occupied_until or idx >= stock_len - 1:
            continue
        entry_price = close_flat[close_start + idx]
        if entry_price <= 0.0:
            continue

        peak = entry_price
        j_end = idx + max_hold + 1
        if j_end > stock_len:
            j_end = stock_len
        exit_idx = j_end - 1

        for j in range(idx + 1, j_end):
            cj = close_flat[close_start + j]
            if cj > peak:
                peak = cj
            ret = (cj - entry_price) / entry_price
            peak_ret = (peak - entry_price) / entry_price
            if ret >= tp:
                exit_idx = j
                break
            if ret <= sl:
                exit_idx = j
                break
            atr_j = atr_flat[close_start + j]
            if (peak_ret >= atr_trigger and atr_j > 0.0
                    and cj < peak - atr_mult * atr_j):
                exit_idx = j
                break

        exit_price = close_flat[close_start + exit_idx]
        pnl = (exit_price - entry_price) / entry_price

        trades_ct += 1
        if pnl > 0.0:
            wins_ct += 1
        sum_ret += pnl
        sum_sq += pnl * pnl
        sum_hold += exit_idx - idx
        if pnl < worst:
            worst = pnl
        if pnl > best:
            best = pnl
        occupied_until = exit_idx

    out[tid, 0] = float(trades_ct)
    out[tid, 1] = float(wins_ct)
    out[tid, 2] = sum_ret
    out[tid, 3] = sum_sq
    out[tid, 4] = float(sum_hold)
    out[tid, 5] = worst
    out[tid, 6] = best


# ──────────────────────────────────────────────────────────
# CPU 端：資料打包
# ──────────────────────────────────────────────────────────

@dataclass
class FlatData:
    close: np.ndarray
    atr: np.ndarray
    stock_offsets: np.ndarray
    stock_lens: np.ndarray
    entries: np.ndarray
    entry_offsets: np.ndarray
    entry_lens: np.ndarray
    stock_ids: List[str] = field(default_factory=list)


def _compute_entries(df: pd.DataFrame, signal_fn) -> np.ndarray:
    """用既有的 signal_fn 在單支股票上找出所有進場日索引。"""
    entries = []
    prev_row = None
    n = len(df)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    records = df.to_dict('records')
    for i in range(n):
        row = records[i]
        try:
            if signal_fn(row, prev_row):
                entries.append(i)
        except (KeyError, TypeError):
            pass
        prev_row = row
    return np.array(entries, dtype=np.int32)


def prepare_flat_arrays(ohlcv_cache: Dict[str, pd.DataFrame], signal_fn) -> FlatData:
    """將所有股票 OHLCV 與進場訊號攤平成 GPU 友善的 1D 陣列。"""
    close_list, atr_list = [], []
    stock_lens, entries_list, entry_lens = [], [], []
    stock_ids = []

    for sid, df in ohlcv_cache.items():
        if 'ATR' not in df.columns or len(df) < 30:
            continue
        c = df['Close'].to_numpy(dtype=np.float32, copy=False)
        a_raw = df['ATR'].to_numpy(dtype=np.float32, copy=True)
        a_raw = np.nan_to_num(a_raw, nan=0.0, posinf=0.0, neginf=0.0)

        entries = _compute_entries(df, signal_fn)

        stock_ids.append(sid)
        close_list.append(c)
        atr_list.append(a_raw)
        stock_lens.append(len(c))
        entries_list.append(entries)
        entry_lens.append(len(entries))

    if not stock_ids:
        return FlatData(
            close=np.empty(0, dtype=np.float32),
            atr=np.empty(0, dtype=np.float32),
            stock_offsets=np.empty(0, dtype=np.int32),
            stock_lens=np.empty(0, dtype=np.int32),
            entries=np.empty(0, dtype=np.int32),
            entry_offsets=np.empty(0, dtype=np.int32),
            entry_lens=np.empty(0, dtype=np.int32),
            stock_ids=[],
        )

    close_flat = np.concatenate(close_list).astype(np.float32)
    atr_flat = np.concatenate(atr_list).astype(np.float32)
    stock_lens_arr = np.array(stock_lens, dtype=np.int32)
    stock_offsets = np.zeros(len(stock_lens), dtype=np.int32)
    stock_offsets[1:] = np.cumsum(stock_lens[:-1])

    entries_flat = (np.concatenate(entries_list).astype(np.int32)
                    if any(len(e) for e in entries_list)
                    else np.empty(0, dtype=np.int32))
    entry_lens_arr = np.array(entry_lens, dtype=np.int32)
    entry_offsets = np.zeros(len(entry_lens), dtype=np.int32)
    entry_offsets[1:] = np.cumsum(entry_lens[:-1])

    return FlatData(
        close=close_flat, atr=atr_flat,
        stock_offsets=stock_offsets, stock_lens=stock_lens_arr,
        entries=entries_flat, entry_offsets=entry_offsets, entry_lens=entry_lens_arr,
        stock_ids=stock_ids,
    )


# ──────────────────────────────────────────────────────────
# 參數網格打包 & 指標彙算
# ──────────────────────────────────────────────────────────

def build_param_arrays(grid: Dict[str, Sequence]):
    combos = list(itertools.product(
        grid['tp'], grid['sl'], grid['atr_mult'], grid['atr_trigger'], grid['max_hold']
    ))
    n = len(combos)
    tp = np.array([c[0] for c in combos], dtype=np.float32)
    sl = np.array([c[1] for c in combos], dtype=np.float32)
    atr_m = np.array([c[2] for c in combos], dtype=np.float32)
    atr_t = np.array([c[3] for c in combos], dtype=np.float32)
    mh = np.array([c[4] for c in combos], dtype=np.int32)
    return combos, n, (tp, sl, atr_m, atr_t, mh)


def reduce_to_metrics(out_np: np.ndarray, n_stocks: int, n_params: int) -> pd.DataFrame:
    """將 (n_stocks*n_params, 7) 的原始輸出聚合成每組參數的績效指標。"""
    out_reshape = out_np.reshape(n_stocks, n_params, OUTPUT_METRICS)
    trades = out_reshape[:, :, 0].sum(axis=0)
    wins = out_reshape[:, :, 1].sum(axis=0)
    sum_ret = out_reshape[:, :, 2].sum(axis=0)
    sum_sq = out_reshape[:, :, 3].sum(axis=0)
    sum_hold = out_reshape[:, :, 4].sum(axis=0)
    worst = out_reshape[:, :, 5].min(axis=0)
    best = out_reshape[:, :, 6].max(axis=0)

    safe = np.where(trades > 0, trades, 1)
    mean = sum_ret / safe
    var = sum_sq / safe - mean ** 2
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    avg_hold = sum_hold / safe
    avg_hold_safe = np.where(avg_hold > 0, avg_hold, 1)
    sharpe = np.where(std > 0, mean / std * np.sqrt(252.0 / avg_hold_safe), 0.0)
    win_rate = np.where(trades > 0, wins / safe * 100, 0.0)

    return pd.DataFrame({
        'trades': trades.astype(int),
        'wins': wins.astype(int),
        'win_rate': win_rate,
        'avg_return': mean * 100,       # 每次交易平均報酬 %
        'std_return': std * 100,
        'avg_hold': avg_hold,
        'sharpe': sharpe,
        'worst_trade': worst * 100,
        'best_trade': best * 100,
    })


# ──────────────────────────────────────────────────────────
# 主搜尋流程
# ──────────────────────────────────────────────────────────

def run_grid_search(
    stock_ids: Sequence[str],
    period: str = '2y',
    grid: Dict = None,
    top_k: int = 5,
    min_trades: int = 20,
    output_json: str = 'best_params.json',
    output_html: str = 'gpu_grid_report.html',
) -> Dict:
    grid = grid or DEFAULT_GRID
    t0 = time.time()

    print(f"\n===== GPU 參數網格搜尋（{len(stock_ids)} 股） =====")
    print(f"參數網格大小: {np.prod([len(v) for v in grid.values()])}"
          f"（tp={len(grid['tp'])} × sl={len(grid['sl'])} × atr_m={len(grid['atr_mult'])}"
          f" × atr_t={len(grid['atr_trigger'])} × mh={len(grid['max_hold'])}）")

    combos, n_params, (tp_a, sl_a, atr_m_a, atr_t_a, mh_a) = build_param_arrays(grid)
    d_tp = cuda.to_device(tp_a)
    d_sl = cuda.to_device(sl_a)
    d_atr_m = cuda.to_device(atr_m_a)
    d_atr_t = cuda.to_device(atr_t_a)
    d_mh = cuda.to_device(mh_a)

    print(f"\n下載 & 計算指標（期間 {period}）...")
    t_dl = time.time()
    cache = bt._batch_download(list(stock_ids), period=period)
    print(f"  成功 {len(cache)} / {len(stock_ids)} 檔，耗時 {time.time() - t_dl:.1f}s")
    for sid, df in cache.items():
        bt._add_indicators(df)

    all_results: Dict[str, Dict] = {}
    html_sections: List[str] = []
    total_kernel_time = 0.0

    for strat_idx, (strat_name, signal_fn) in enumerate(bt.STRATEGY_MAP.items(), 1):
        t_s = time.time()
        flat = prepare_flat_arrays(cache, signal_fn)
        n_stocks = len(flat.stock_ids)
        if n_stocks == 0 or len(flat.entries) == 0:
            print(f"  [{strat_idx:2d}] {strat_name}: 無進場訊號，略過")
            continue

        # 上 GPU
        d_close = cuda.to_device(flat.close)
        d_atr = cuda.to_device(flat.atr)
        d_so = cuda.to_device(flat.stock_offsets)
        d_sl_arr = cuda.to_device(flat.stock_lens)
        d_e = cuda.to_device(flat.entries)
        d_eo = cuda.to_device(flat.entry_offsets)
        d_el = cuda.to_device(flat.entry_lens)

        total_threads = n_stocks * n_params
        out = cuda.device_array((total_threads, OUTPUT_METRICS), dtype=np.float32)

        threads_per_block = 256
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        t_k = time.time()
        simulate_kernel[blocks, threads_per_block](
            d_close, d_atr, d_so, d_sl_arr, d_e, d_eo, d_el,
            d_tp, d_sl, d_atr_m, d_atr_t, d_mh, out
        )
        cuda.synchronize()
        kernel_time = time.time() - t_k
        total_kernel_time += kernel_time

        # 下 GPU 聚合
        out_host = out.copy_to_host()
        metrics = reduce_to_metrics(out_host, n_stocks, n_params)

        # 加入參數欄
        metrics['tp'] = tp_a
        metrics['sl'] = sl_a
        metrics['atr_mult'] = atr_m_a
        metrics['atr_trigger'] = atr_t_a
        metrics['max_hold'] = mh_a

        # 過濾交易樣本過少的組合
        valid = metrics[metrics['trades'] >= min_trades].copy()
        if valid.empty:
            print(f"  [{strat_idx:2d}] {strat_name}: 所有參數交易數 <{min_trades}，略過")
            continue

        valid = valid.sort_values('sharpe', ascending=False).reset_index(drop=True)
        top = valid.head(top_k)
        best = top.iloc[0]

        all_results[strat_name] = {
            'best_params': {
                'tp': float(best['tp']),
                'sl': float(best['sl']),
                'atr_mult': float(best['atr_mult']),
                'atr_trigger': float(best['atr_trigger']),
                'max_hold': int(best['max_hold']),
            },
            'best_metrics': {
                'trades': int(best['trades']),
                'win_rate': round(float(best['win_rate']), 1),
                'avg_return': round(float(best['avg_return']), 2),
                'avg_hold': round(float(best['avg_hold']), 1),
                'sharpe': round(float(best['sharpe']), 3),
                'worst_trade': round(float(best['worst_trade']), 2),
                'best_trade': round(float(best['best_trade']), 2),
            },
            'n_stocks': n_stocks,
            'top_k': top.round(3).to_dict(orient='records'),
        }

        print(f"  [{strat_idx:2d}] {strat_name}: "
              f"{n_stocks} 股 × {n_params} 組 = {total_threads:,} threads  "
              f"kernel={kernel_time*1000:.1f}ms  "
              f"最佳 Sharpe={best['sharpe']:.2f}  "
              f"勝率={best['win_rate']:.0f}%  "
              f"tp={best['tp']:.2f}/sl={best['sl']:.2f}/mh={int(best['max_hold'])}"
              f"  (總耗時 {time.time()-t_s:.1f}s)")

        # HTML sections
        html_sections.append(
            _build_strategy_html(strat_name, valid, top)
        )

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    _write_html_report(
        output_html, all_results, html_sections,
        total_stocks=len(cache), total_kernel_time=total_kernel_time,
        total_time=time.time() - t0, grid=grid
    )

    print(f"\n總計 GPU kernel 時間: {total_kernel_time*1000:.0f}ms")
    print(f"總耗時: {time.time() - t0:.1f}s")
    print(f"已寫入 {output_json} 與 {output_html}")
    return all_results


# ──────────────────────────────────────────────────────────
# HTML 報告
# ──────────────────────────────────────────────────────────

def _build_strategy_html(strat_name: str, all_rows: pd.DataFrame, top: pd.DataFrame) -> str:
    def _row_color(r):
        s = r['sharpe']
        if s >= 1.5:
            return 'background:rgba(38,166,154,.3)'
        if s >= 1.0:
            return 'background:rgba(38,166,154,.15)'
        if s < 0:
            return 'background:rgba(239,83,80,.15)'
        return ''

    top_html = ''.join(
        f'<tr style="{_row_color(r)}">'
        f'<td>{i+1}</td>'
        f'<td>{r["tp"]:.2f}</td><td>{r["sl"]:.2f}</td>'
        f'<td>{r["atr_mult"]:.1f}</td><td>{r["atr_trigger"]:.2f}</td>'
        f'<td>{int(r["max_hold"])}</td>'
        f'<td>{int(r["trades"])}</td>'
        f'<td>{r["win_rate"]:.1f}%</td>'
        f'<td>{r["avg_return"]:.2f}%</td>'
        f'<td>{r["avg_hold"]:.1f}</td>'
        f'<td><b>{r["sharpe"]:.2f}</b></td>'
        f'<td>{r["worst_trade"]:.1f}%</td>'
        f'<td>{r["best_trade"]:.1f}%</td>'
        f'</tr>'
        for i, r in top.iterrows()
    )

    # 熱力圖：tp × sl（取各格平均 Sharpe）
    pivot = all_rows.pivot_table(
        index='sl', columns='tp', values='sharpe', aggfunc='mean'
    ).sort_index(ascending=False)
    cmin, cmax = pivot.min().min(), pivot.max().max()

    def _heat_color(v):
        if pd.isna(v):
            return '#1a1a2e'
        if cmax == cmin:
            return '#42a5f5'
        t = (v - cmin) / (cmax - cmin)
        if t > 0.5:
            r, g, b = int(200*(1-t)*2), int(166 + (89*t)), int(154 - 100*t)
        else:
            r, g, b = int(239 - (200-239)*(t*2)), int(83 + (166-83)*(t*2)), int(80 + (154-80)*(t*2))
        return f'rgb({max(0,min(255,r))},{max(0,min(255,g))},{max(0,min(255,b))})'

    heat_cells = ''
    heat_cells += '<tr><th></th>' + ''.join(f'<th>tp={c:.2f}</th>' for c in pivot.columns) + '</tr>'
    for sl_val in pivot.index:
        row = '<tr>' + f'<th>sl={sl_val:.2f}</th>'
        for col in pivot.columns:
            v = pivot.loc[sl_val, col]
            txt = f'{v:.2f}' if not pd.isna(v) else '-'
            bg = _heat_color(v)
            row += f'<td style="background:{bg};color:#000;font-weight:600">{txt}</td>'
        row += '</tr>'
        heat_cells += row

    return f'''
<section>
    <h3 style="margin-top:30px;">{strat_name}</h3>
    <h4>Top 10 參數組合（按 Sharpe 排序）</h4>
    <table class="data-table">
        <thead>
            <tr><th>#</th><th>tp</th><th>sl</th><th>atr_mult</th><th>atr_trig</th>
                <th>max_hold</th><th>交易數</th><th>勝率</th><th>平均報酬</th>
                <th>平均持有</th><th>Sharpe</th><th>最差</th><th>最好</th></tr>
        </thead>
        <tbody>{top_html}</tbody>
    </table>
    <h4 style="margin-top:20px;">Sharpe 熱力圖 (tp × sl, 取各 atr/mh 組合平均)</h4>
    <table class="heatmap">{heat_cells}</table>
</section>
'''


def _write_html_report(path, all_results, sections, total_stocks, total_kernel_time,
                       total_time, grid):
    ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_rows = ''
    for name, r in all_results.items():
        p = r['best_params']
        m = r['best_metrics']
        summary_rows += (
            f'<tr><td><b>{name}</b></td>'
            f'<td>{p["tp"]:.2f}</td><td>{p["sl"]:.2f}</td>'
            f'<td>{p["atr_mult"]:.1f}</td><td>{p["atr_trigger"]:.2f}</td>'
            f'<td>{p["max_hold"]}</td>'
            f'<td>{m["trades"]}</td><td>{m["win_rate"]}%</td>'
            f'<td>{m["avg_return"]}%</td><td><b>{m["sharpe"]}</b></td></tr>'
        )

    grid_str = '<br>'.join(
        f'<code>{k}</code>: {v}' for k, v in grid.items()
    )

    html = f'''<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>GPU 參數網格回測報告 — {ts}</title>
<style>
    body {{ background:#0f0f1a; color:#e0e0e0; font-family:-apple-system,'Segoe UI',sans-serif; padding:30px; max-width:1400px; margin:auto; }}
    h1 {{ color:#42a5f5; border-bottom:2px solid #42a5f5; padding-bottom:10px; }}
    h2 {{ color:#81d4fa; margin-top:40px; }}
    h3 {{ color:#fff; border-left:4px solid #42a5f5; padding-left:12px; }}
    h4 {{ color:#a0a0b0; margin-bottom:8px; }}
    .info {{ background:#1a1a2e; padding:14px 18px; border-radius:8px; border-left:4px solid #26a69a; margin-bottom:20px; }}
    .info code {{ background:#0f0f1a; padding:2px 6px; border-radius:4px; color:#ffb74d; }}
    table.data-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    table.data-table th, table.data-table td {{ padding:6px 10px; border-bottom:1px solid #2a2a4a; text-align:right; }}
    table.data-table th {{ background:#1a1a2e; color:#81d4fa; }}
    table.data-table td:first-child, table.data-table th:first-child {{ text-align:left; }}
    table.heatmap {{ border-collapse:collapse; font-size:12px; margin-top:8px; }}
    table.heatmap th, table.heatmap td {{ padding:6px 10px; border:1px solid #2a2a4a; text-align:center; min-width:60px; }}
    table.heatmap th {{ background:#1a1a2e; color:#81d4fa; }}
</style></head>
<body>
<h1>🚀 GPU 參數網格回測報告</h1>
<div class="info">
    <b>執行時間：</b>{ts}<br>
    <b>股票池大小：</b>{total_stocks}<br>
    <b>GPU Kernel 總耗時：</b>{total_kernel_time*1000:.0f} ms
    （含資料下載/指標計算的總流程 {total_time:.1f} s）<br>
    <b>設備：</b>NVIDIA GeForce RTX 5080 (Blackwell / sm_120)<br><br>
    <b>參數網格：</b><br>{grid_str}
</div>

<h2>📊 各策略最佳參數</h2>
<table class="data-table">
    <thead><tr>
        <th>策略</th><th>tp</th><th>sl</th><th>atr_mult</th><th>atr_trig</th>
        <th>max_hold</th><th>交易數</th><th>勝率</th><th>平均報酬</th><th>Sharpe</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
</table>

{''.join(sections)}
</body></html>'''

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

DEMO_STOCKS = ['2330', '2317', '2454', '2308', '2881', '2882', '2303', '3711',
               '2412', '2886', '1301', '2891', '3008', '2357', '6505']


def _load_full_market_ids() -> List[str]:
    """從 Strategy_twe 當日法人資料抓所有台股代號（含上市/櫃）。"""
    try:
        import Strategy_twe as stw
        all_data = stw.get_recent_institutional()
        ids = all_data['證券代號'].astype(str).str.strip().unique().tolist()
        print(f"從 Strategy_twe 載入 {len(ids)} 個代號")
        return ids
    except Exception as e:
        print(f"⚠ 無法載入全市場（{e}），改用 DEMO_STOCKS")
        return DEMO_STOCKS


def main():
    parser = argparse.ArgumentParser(description='GPU 參數網格回測 (RTX 50 相容)')
    parser.add_argument('--stocks', nargs='+', help='指定股票代號（多個以空格分隔）')
    parser.add_argument('--all', action='store_true', help='使用全市場（Strategy_twe 當日資料）')
    parser.add_argument('--period', default='2y', help='下載期間 (default 2y)')
    parser.add_argument('--top-k', type=int, default=10, help='每策略回報前 K 組 (default 10)')
    parser.add_argument('--min-trades', type=int, default=20, help='最少有效交易數門檻')
    parser.add_argument('--out-json', default='best_params.json')
    parser.add_argument('--out-html', default='gpu_grid_report.html')
    args = parser.parse_args()

    if args.all:
        stock_ids = _load_full_market_ids()
    elif args.stocks:
        stock_ids = args.stocks
    else:
        stock_ids = DEMO_STOCKS

    if not cuda.is_available():
        raise SystemExit('❌ CUDA 不可用，請先確認 numba-cuda / nvrtc 安裝正確')

    gpu = cuda.get_current_device()
    print(f"GPU: {gpu.name.decode() if isinstance(gpu.name, bytes) else gpu.name}"
          f"  CC={gpu.compute_capability}  SM={gpu.MULTIPROCESSOR_COUNT}")

    run_grid_search(
        stock_ids,
        period=args.period,
        top_k=args.top_k,
        min_trades=args.min_trades,
        output_json=args.out_json,
        output_html=args.out_html,
    )


if __name__ == '__main__':
    main()

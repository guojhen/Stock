"""
回測引擎 — 對 8 種策略以近 1 年歷史資料進行模擬交易與績效評估。

由於法人歷史資料無法透過公開 API 批量取得，
各策略以純技術面條件近似原始進場邏輯。
"""

import json
import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from dataclasses import dataclass, field
from datetime import datetime

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

BEST_PARAMS_FILE = 'best_params.json'

# ── 常數 ──────────────────────────────────────────────

MA_SHORT = 5
MA_LONG = 10
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
KD_PERIOD = 9

# A7：取消 +15% 固定停利，改以 ATR trailing + tier 1R/2R/3R 為主
# TAKE_PROFIT 保留為「硬上限保險」（避免極端 outlier 失真），預設拉到 30%
TAKE_PROFIT = 0.30
STOP_LOSS = -0.05         # 硬停損下限
MAX_HOLD_DAYS = 20
ATR_STOP_MULT = 2.5       # A7：trailing 改 2.5x ATR（原 2.0 過早砍長腳）
ATR_STOP_TRIGGER = 0.04   # A7：獲利 4% 即啟用 trailing（比原 6% 更早保護成本）

# D1：Volatility-aware trailing — 持有期分段 ATR 倍數
ATR_TRAIL_EARLY = 1.5     # 持有 <= 10 日：緊（保護成本）
ATR_TRAIL_MID = 2.5       # 11~30 日：中（讓利潤跑）
ATR_TRAIL_LATE = 3.5      # > 30 日：寬（讓贏家奔跑）
ATR_TRAIL_DAYS_EARLY = 10
ATR_TRAIL_DAYS_LATE = 30


# ── 資料結構 ──────────────────────────────────────────

@dataclass
class BacktestResult:
    summary: pd.DataFrame
    trades: pd.DataFrame
    equity_curves: dict = field(default_factory=dict)
    ohlcv_cache: dict = field(default_factory=dict)


# ── 技術指標計算 ──────────────────────────────────────

def _calc_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _flatten_columns(df):
    """將 MultiIndex 欄位攤平為單層（yfinance 相容）。"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def _add_indicators(df):
    """在 OHLCV DataFrame 上新增技術指標欄位（就地修改）。"""
    _flatten_columns(df)
    c = df['Close']
    df['MA5'] = c.rolling(MA_SHORT).mean()
    df['MA10'] = c.rolling(MA_LONG).mean()
    df['MA20'] = c.rolling(20).mean()
    df['RSI'] = _calc_rsi(c)

    bb_ma = c.rolling(BB_PERIOD).mean()
    bb_std = c.rolling(BB_PERIOD).std()
    df['BB_upper'] = bb_ma + BB_STD * bb_std
    df['BB_lower'] = bb_ma - BB_STD * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_ma

    vol_avg = df['Volume'].rolling(20).mean()
    df['Vol_ratio'] = df['Volume'] / vol_avg

    df['Prev_high_20'] = df['High'].rolling(20).max().shift(1)
    df['High_52w'] = df['High'].rolling(250, min_periods=60).max()

    bw = df['BB_width'].dropna()
    if len(bw) >= 20:
        df['BB_width_pctl'] = bw.rolling(20).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
        )
    else:
        df['BB_width_pctl'] = np.nan

    # ATR
    prev_close = c.shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(ATR_PERIOD).mean()

    # MACD
    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # KD
    low_n = df['Low'].rolling(KD_PERIOD).min()
    high_n = df['High'].rolling(KD_PERIOD).max()
    rsv = 100 * (c - low_n) / (high_n - low_n).replace(0, np.nan)
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    return df


# ── 策略進場訊號 ──────────────────────────────────────

def _signal_s1_s3(row, prev):
    """策略 1-3 近似：MA 金叉 + 量比 > 1.2"""
    if pd.isna(row['MA5']) or pd.isna(row['MA10']):
        return False
    if prev is not None and not pd.isna(prev['MA5']) and prev['MA5'] <= prev['MA10']:
        if row['MA5'] > row['MA10'] and row['Vol_ratio'] > 1.2:
            return True
    return False


def _signal_s4(row, _prev):
    """策略 4：量比 > 2.0 + 突破前 20 日高點 + MA 金叉"""
    if pd.isna(row['Vol_ratio']) or pd.isna(row['Prev_high_20']):
        return False
    return (row['Vol_ratio'] > 2.0
            and row['Close'] > row['Prev_high_20']
            and row['MA5'] > row['MA10'])


def _signal_s5(row, prev):
    """策略 5：RSI < 30 後回升至 35 以上 + 收盤 > MA10"""
    if prev is None or pd.isna(prev['RSI']) or pd.isna(row['RSI']):
        return False
    return (prev['RSI'] < 30
            and row['RSI'] >= 35
            and row['Close'] > row['MA10'])


def _signal_s6(row, _prev):
    """策略 6：布林帶寬百分位 < 20% + 收盤突破上軌"""
    if pd.isna(row['BB_width_pctl']) or pd.isna(row['BB_upper']):
        return False
    return (row['BB_width_pctl'] < 0.20
            and row['Close'] > row['BB_upper'])


def _signal_s7(row, prev):
    """策略 7 近似：MA 金叉 + 量比 > 1.5"""
    if pd.isna(row['MA5']) or pd.isna(row['MA10']):
        return False
    if prev is not None and not pd.isna(prev['MA5']) and prev['MA5'] <= prev['MA10']:
        if row['MA5'] > row['MA10'] and row['Vol_ratio'] > 1.5:
            return True
    return False


def _signal_s8(row, prev):
    """策略 8 近似：MA5 > MA10 + RSI 40-70 + 量縮後量增"""
    if pd.isna(row['RSI']) or pd.isna(row['Vol_ratio']):
        return False
    if prev is None or pd.isna(prev['Vol_ratio']):
        return False
    return (row['MA5'] > row['MA10']
            and 40 <= row['RSI'] <= 70
            and prev['Vol_ratio'] < 0.8
            and row['Vol_ratio'] > 1.0)


def _signal_s9(row, prev):
    """策略 9：MACD 金叉 + 紅柱擴大 + 站上 MA10"""
    if prev is None or pd.isna(row['MACD']) or pd.isna(row['MACD_signal']):
        return False
    if pd.isna(prev['MACD']) or pd.isna(prev['MACD_signal']):
        return False
    # 金叉（昨天 MACD<=signal，今天 > signal）
    cross = prev['MACD'] <= prev['MACD_signal'] and row['MACD'] > row['MACD_signal']
    hist_expand = (not pd.isna(row['MACD_hist']) and not pd.isna(prev['MACD_hist'])
                   and row['MACD_hist'] > prev['MACD_hist'] and row['MACD_hist'] > 0)
    above_ma = not pd.isna(row['MA10']) and row['Close'] > row['MA10']
    return cross and hist_expand and above_ma


def _signal_s10(row, prev):
    """策略 10：KD 低檔黃金交叉（K<50）"""
    if prev is None or pd.isna(row['K']) or pd.isna(row['D']):
        return False
    if pd.isna(prev['K']) or pd.isna(prev['D']):
        return False
    cross = prev['K'] <= prev['D'] and row['K'] > row['D']
    return cross and row['K'] < 50 and row['Close'] > row['MA10'] * 0.97


def _signal_s11(row, prev):
    """策略 11：突破 52 週高點 + 量比 > 1.5 + 多頭排列"""
    if pd.isna(row['High_52w']) or pd.isna(row['Vol_ratio']):
        return False
    if prev is None or pd.isna(prev['High_52w']):
        return False
    # 當日收盤接近或突破前一日 52w 高
    if row['Close'] < prev['High_52w'] * 0.98:
        return False
    if row['Vol_ratio'] < 1.5:
        return False
    return row['MA5'] > row['MA10']


def _signal_s12(row, prev):
    """策略 12：量縮價穩 + RSI 止跌回升（底部價量背離近似）"""
    if prev is None or pd.isna(row['RSI']) or pd.isna(prev['RSI']):
        return False
    if pd.isna(row['Vol_ratio']) or pd.isna(row['MA10']):
        return False
    if row['Vol_ratio'] > 0.8:
        return False
    if row['Close'] > row['MA10'] * 1.02:
        return False
    return row['RSI'] > prev['RSI'] + 2 and row['RSI'] < 55


def _signal_s13(row, prev):
    """策略 13：20 日動能領先（近似相對強度）+ MA 多頭"""
    if pd.isna(row['MA5']) or pd.isna(row['MA10']) or pd.isna(row['MA20']):
        return False
    # 單股 20 日動能（非相對大盤，近似）
    return (row['MA5'] > row['MA10'] > row['MA20']
            and row['Close'] > row['MA20'] * 1.05)


STRATEGY_MAP = {
    '策略1-3 (法人+MA近似)': _signal_s1_s3,
    '策略4 (量價齊揚)': _signal_s4,
    '策略5 (RSI超賣反彈)': _signal_s5,
    '策略6 (布林收斂突破)': _signal_s6,
    '策略7 (營收+MA近似)': _signal_s7,
    '策略8 (融資+法人近似)': _signal_s8,
    '策略9 (MACD金叉)': _signal_s9,
    '策略10 (KD底部黃金叉)': _signal_s10,
    '策略11 (Darvas突破)': _signal_s11,
    '策略12 (價量背離)': _signal_s12,
    '策略13 (動能領漲)': _signal_s13,
}


# ── 最佳參數載入（由 gpu_backtest.py 產生） ──────────────

def load_best_params(path=BEST_PARAMS_FILE):
    """載入 GPU 網格搜尋出的最佳參數，若不存在回傳 None。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return {k: v['best_params'] for k, v in raw.items() if 'best_params' in v}
    except Exception as e:
        print(f"⚠ 讀取 {path} 失敗: {e}")
        return None


def get_strategy_params(strat_name, best_params=None):
    """依策略名稱取得專屬參數，找不到就回預設。"""
    defaults = {
        'tp': TAKE_PROFIT, 'sl': STOP_LOSS, 'max_hold': MAX_HOLD_DAYS,
        'atr_mult': ATR_STOP_MULT, 'atr_trigger': ATR_STOP_TRIGGER,
    }
    if not best_params or strat_name not in best_params:
        return defaults
    p = best_params[strat_name]
    return {
        'tp': float(p.get('tp', defaults['tp'])),
        'sl': float(p.get('sl', defaults['sl'])),
        'max_hold': int(p.get('max_hold', defaults['max_hold'])),
        'atr_mult': float(p.get('atr_mult', defaults['atr_mult'])),
        'atr_trigger': float(p.get('atr_trigger', defaults['atr_trigger'])),
    }


# ── 交易模擬 ──────────────────────────────────────────

def _simulate_trades(df, entry_indices, tp=TAKE_PROFIT, sl=STOP_LOSS,
                     max_hold=MAX_HOLD_DAYS, atr_mult=ATR_STOP_MULT,
                     atr_trigger=ATR_STOP_TRIGGER,
                     volatility_aware=True):
    """從進場點模擬交易。
    A7：tp 改為 hard-cap (預設 30%)，不再把 +15% 當主要出場。
    D1：volatility_aware=True 時，trailing ATR 倍數依持有時間放寬：
        ≤10 日 → 1.5x，11~30 日 → 2.5x，>30 日 → 3.5x。
    """
    trades = []
    close = df['Close'].values
    atr = df['ATR'].values if 'ATR' in df.columns else np.full(len(close), np.nan)
    dates = df.index
    n = len(close)

    occupied_until = -1

    for idx in entry_indices:
        if idx >= n - 1 or idx <= occupied_until:
            continue
        entry_price = close[idx]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        exit_idx = min(idx + max_hold, n - 1)
        exit_reason = '最長持有'
        peak = entry_price

        for j in range(idx + 1, min(idx + max_hold + 1, n)):
            cj = close[j]
            if cj > peak:
                peak = cj
            ret = (cj - entry_price) / entry_price
            peak_ret = (peak - entry_price) / entry_price
            held = j - idx

            # 硬上限（防 outlier）
            if ret >= tp:
                exit_idx = j
                exit_reason = '停利上限'
                break
            if ret <= sl:
                exit_idx = j
                exit_reason = '停損'
                break

            # D1：Volatility-aware trailing
            if volatility_aware:
                if held <= ATR_TRAIL_DAYS_EARLY:
                    eff_mult = ATR_TRAIL_EARLY
                elif held <= ATR_TRAIL_DAYS_LATE:
                    eff_mult = ATR_TRAIL_MID
                else:
                    eff_mult = ATR_TRAIL_LATE
            else:
                eff_mult = atr_mult

            atr_j = atr[j] if j < len(atr) else np.nan
            if (peak_ret >= atr_trigger and not np.isnan(atr_j)
                    and atr_j > 0 and cj < peak - eff_mult * atr_j):
                exit_idx = j
                exit_reason = f'ATR trailing ({eff_mult:.1f}x)'
                break

        exit_price = close[exit_idx]
        pnl = (exit_price - entry_price) / entry_price

        trades.append({
            'entry_date': dates[idx],
            'exit_date': dates[exit_idx],
            'entry_price': round(float(entry_price), 2),
            'exit_price': round(float(exit_price), 2),
            'return': round(float(pnl), 4),
            'hold_days': int(exit_idx - idx),
            'exit_reason': exit_reason,
        })
        occupied_until = exit_idx

    return trades


# ── 績效指標 ──────────────────────────────────────────

def _calc_metrics(trades_df):
    """從交易明細計算績效指標。"""
    if trades_df.empty:
        return {
            '交易次數': 0, '勝率': 0, '平均報酬': 0,
            '最大回撤': 0, 'Sharpe': 0, '平均持股天數': 0,
        }

    returns = trades_df['return'].values
    wins = (returns > 0).sum()
    total = len(returns)

    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0

    mean_r = float(returns.mean())
    std_r = float(returns.std()) if len(returns) > 1 else 0
    sharpe = (mean_r / std_r * np.sqrt(252 / max(trades_df['hold_days'].mean(), 1))
              if std_r > 0 else 0)

    return {
        '交易次數': total,
        '勝率': round(wins / total * 100, 1),
        '平均報酬': round(mean_r * 100, 2),
        '最大回撤': round(max_dd * 100, 2),
        'Sharpe': round(sharpe, 2),
        '平均持股天數': round(float(trades_df['hold_days'].mean()), 1),
    }


def _build_equity_curve(trades_df):
    """建立每日累積報酬序列（以 exit_date 為基準）。"""
    if trades_df.empty:
        return pd.Series(dtype=float)
    by_date = trades_df.groupby('exit_date')['return'].sum()
    by_date = by_date.sort_index()
    cumulative = (1 + by_date).cumprod()
    return cumulative


def _monthly_returns(trades_df):
    """依月份彙總報酬。"""
    if trades_df.empty:
        return pd.Series(dtype=float)
    t = trades_df.copy()
    t['month'] = pd.to_datetime(t['exit_date']).dt.to_period('M')
    return t.groupby('month')['return'].mean()


# ── 批次下載 ──────────────────────────────────────────

def _batch_download(stock_ids, period='1y', batch_size=50):
    """批次下載 OHLCV，回傳 {stock_id: DataFrame}。"""
    import re as _re

    def _is_stock(sid):
        s = str(sid).strip()
        if _re.match(r'^\d{4}$', s):
            return True
        if _re.match(r'^00\d{2,4}[A-Za-z]?$', s):
            return True
        return False

    valid = [str(sid).strip() for sid in stock_ids if _is_stock(sid)]
    valid = list(dict.fromkeys(valid))
    cache = {}

    for suffix in ['.TW', '.TWO']:
        to_fetch = [sid for sid in valid if sid not in cache]
        if not to_fetch:
            break
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i + batch_size]
            tickers = [f"{sid}{suffix}" for sid in batch]
            try:
                if len(tickers) == 1:
                    raw = yf.download(tickers[0], period=period, progress=False)
                    raw = _flatten_columns(raw)
                    if not raw.empty and len(raw) > 30:
                        cache[batch[0]] = raw
                else:
                    raw = yf.download(tickers, period=period, progress=False,
                                      group_by='ticker', threads=True)
                    for sid, ticker in zip(batch, tickers):
                        if sid in cache:
                            continue
                        try:
                            sdf = raw[ticker].copy()
                            sdf = _flatten_columns(sdf)
                            sdf = sdf.dropna(subset=['Close'])
                            if not sdf.empty and len(sdf) > 30:
                                cache[sid] = sdf
                        except (KeyError, TypeError):
                            pass
            except Exception:
                pass
            done = min(i + batch_size, len(to_fetch))
            print(f"  回測資料下載: {done}/{len(to_fetch)} ({suffix})")

    return cache


# ── 主入口 ────────────────────────────────────────────

def run_backtest(stock_ids, period='1y', use_best_params=True):
    """
    對指定股票池執行全策略回測。

    Parameters
    ----------
    stock_ids : list[str]
        要回測的股票代號
    period : str
        yfinance 下載期間（預設 '1y'）
    use_best_params : bool
        是否讀取 best_params.json 套用每策略最佳參數（預設 True）

    Returns
    -------
    BacktestResult
    """
    print(f"\n===== 開始回測（{len(stock_ids)} 支股票，期間 {period}）=====")

    best_params = load_best_params() if use_best_params else None
    if best_params:
        print(f"  已套用 {len(best_params)} 組 GPU 優化參數 (來源: {BEST_PARAMS_FILE})")

    ohlcv_cache = _batch_download(stock_ids, period=period)
    print(f"  成功下載 {len(ohlcv_cache)} 支股票的歷史資料")

    for sid, df in ohlcv_cache.items():
        _add_indicators(df)

    all_trades = []
    equity_curves = {}

    for strat_name, signal_fn in STRATEGY_MAP.items():
        params = get_strategy_params(strat_name, best_params)
        strat_trades = []
        for sid, df in ohlcv_cache.items():
            entries = []
            prev_row = None
            for i in range(len(df)):
                row = df.iloc[i]
                if signal_fn(row, prev_row):
                    entries.append(i)
                prev_row = row

            trades = _simulate_trades(df, entries, **params)
            for t in trades:
                t['stock_id'] = sid
                t['strategy'] = strat_name
            strat_trades.extend(trades)

        strat_df = pd.DataFrame(strat_trades)
        if not strat_df.empty:
            equity_curves[strat_name] = _build_equity_curve(strat_df)
        all_trades.extend(strat_trades)

    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    summary_rows = []
    for strat_name in STRATEGY_MAP:
        sub = trades_df[trades_df['strategy'] == strat_name] if not trades_df.empty else pd.DataFrame()
        metrics = _calc_metrics(sub)
        metrics['策略'] = strat_name
        monthly = _monthly_returns(sub)
        metrics['月度報酬'] = monthly.to_dict() if not monthly.empty else {}
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    cols = ['策略', '交易次數', '勝率', '平均報酬', '最大回撤', 'Sharpe', '平均持股天數']
    summary = summary[[c for c in cols if c in summary.columns]]

    print("\n回測完成！")
    print(summary.to_string(index=False))

    return BacktestResult(
        summary=summary,
        trades=trades_df,
        equity_curves=equity_curves,
        ohlcv_cache=ohlcv_cache,
    )


if __name__ == '__main__':
    test_ids = ['2330', '2317', '2454', '2308', '2881', '2882', '2303', '3711',
                '2412', '2886', '1301', '2891', '3008', '2357', '6505']
    result = run_backtest(test_ids)
    print(result.summary)

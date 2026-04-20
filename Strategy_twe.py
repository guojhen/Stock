import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
import time
import os
import re
import json
import logging
import yfinance as yf

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# === 設定 ===
HOLDINGS_FILE = 'holdings.csv'
TRADE_HISTORY_FILE = 'trade_history.csv'
RECENT_DAYS = 15
CONSEC_BUY_DAYS = 3
CONSEC_SELL_DAYS = 2
MIN_BUY_THRESHOLD = 10000
MA_SHORT = 5
MA_LONG = 10
GOLDEN_CROSS_THRESHOLD = 0.02

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 80
BB_PERIOD = 20
BB_STD = 2
VOLUME_SURGE_RATIO = 2.0
REVENUE_GROWTH_THRESHOLD = 0.20

ATR_PERIOD = 14
KD_PERIOD = 9
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_AVG_VOLUME = 500  # 日均量下限（張），流動性過濾

# === 持股風險管理設定 ===
ATR_INITIAL_STOP_MULT = 1.5      # 初始停損：買進價 - N×ATR
CHANDELIER_ATR_MULT = 3.0         # 吊燈停損：最高價 - N×ATR
CHANDELIER_ATR_MULT_BEAR = 2.0    # 空頭環境下縮緊
BREAKEVEN_PROFIT_TRIGGER = 0.05   # 獲利 N% 後啟動保本停損
ACCOUNT_SIZE_DEFAULT = 1_000_000  # 帳戶總資金（可由 HOLDINGS_ACCOUNT_SIZE 環境變數覆蓋）
POSITION_RISK_PCT = 0.01          # 單筆最多損失比例（凱利簡化版）

_technicals_cache = {}
_taiex_returns_cache = {'return_20d': None, 'bullish': None}


def _is_stock_or_etf(stock_id):
    """過濾掉權證、債券等非股票代碼，只保留股票與 ETF"""
    s = str(stock_id).strip()
    # 一般股票：4 碼數字 (1101~9999, 含 0050 等老 ETF)
    if re.match(r'^\d{4}$', s):
        return True
    # ETF/基金：以 00 開頭，後接 2-4 碼數字，可選尾碼字母 (00878, 006208, 00663L, 00945B)
    if re.match(r'^00\d{2,4}[A-Za-z]?$', s):
        return True
    return False


def _to_roc_date(date_str):
    clean = date_str.replace('-', '')
    y, m, d = int(clean[:4]), clean[4:6], clean[6:8]
    return f"{y - 1911}/{m}/{d}"

# ===================== 三大法人資料 =====================

def get_twse_institutional(date_str):
    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str}&selectType=ALL"
    try:
        r = requests.get(url, timeout=10, verify=False)
        r.raise_for_status()
        if not r.text.strip().startswith('{'):
            return None
        data = r.json()
        if data.get('stat') != 'OK' or 'data' not in data or 'fields' not in data:
            print(f"TWSE {date_str} 無資料或欄位缺失")
            return None
        df = pd.DataFrame(data['data'], columns=data['fields'])
        rename_map = {
            '外陸資買賣超股數(不含外資自營商)': '外資買賣超',
            '投信買賣超股數': '投信買賣超',
            '自營商買賣超股數(自行買賣)': '自營商自行買賣超',
            '自營商買賣超股數(避險)': '自營商避險買賣超',
            '三大法人買賣超股數': '三大法人買賣超'
        }
        df.rename(columns=rename_map, inplace=True)
        numeric_cols = ['外資買賣超', '投信買賣超', '自營商自行買賣超', '自營商避險買賣超', '三大法人買賣超']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['自營商買賣超'] = df.get('自營商自行買賣超', 0) + df.get('自營商避險買賣超', 0)
        df['日期'] = date_str
        df['市場'] = '上市'
        keep = ['證券代號', '證券名稱', '外資買賣超', '投信買賣超', '自營商買賣超', '三大法人買賣超', '日期', '市場']
        return df[[c for c in keep if c in df.columns]]
    except Exception as e:
        print(f"TWSE {date_str} 錯誤: {e}")
        return None


def get_tpex_institutional(date_str):
    date_roc = _to_roc_date(date_str)
    url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&se=AL&t=D&d={date_roc}"
    try:
        r = requests.get(url, timeout=10, verify=False)
        r.raise_for_status()
        data = r.json()
        if data.get('stat') != 'ok':
            return None
        tables = data.get('tables', [])
        if not tables or 'data' not in tables[0]:
            return None
        fields = tables[0]['fields']
        rows = tables[0]['data']
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=fields)
        df['日期'] = date_str
        df['市場'] = '上櫃'
        df['證券代號'] = df.iloc[:, 0].str.strip()
        df['證券名稱'] = df.iloc[:, 1].str.strip()

        def parse_idx(idx):
            if idx < len(fields):
                return pd.to_numeric(df.iloc[:, idx].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            return 0

        # TPEx 欄位名稱重複，需用索引定位：
        # 0:代號, 1:名稱, 2-4:外資, 5-7:外資自營商, 8-10:投信,
        # 11-13:自營商(自行), 14-16:自營商(避險), 17-19:自營商合計,
        # 20-22:合計, 23:三大法人買賣超股數合計
        n = len(fields)
        if n >= 24:
            df['外資買賣超'] = parse_idx(4)
            df['投信買賣超'] = parse_idx(10)
            df['自營商買賣超'] = parse_idx(13) + parse_idx(16)
            df['三大法人買賣超'] = parse_idx(n - 1)
        else:
            print(f"TPEx 欄位數量異常: {n}")
            df['外資買賣超'] = 0
            df['投信買賣超'] = 0
            df['自營商買賣超'] = 0
            df['三大法人買賣超'] = parse_idx(n - 1)
        keep = ['證券代號', '證券名稱', '外資買賣超', '投信買賣超', '自營商買賣超', '三大法人買賣超', '日期', '市場']
        return df[keep]
    except Exception as e:
        print(f"TPEx {date_str} 錯誤: {e}")
        return None


def get_recent_institutional(days=RECENT_DAYS, max_attempts=60):
    data_list = []
    current = datetime.now()
    collected = 0
    attempts = 0
    while collected < days and attempts < max_attempts:
        date_str = current.strftime('%Y-%m-%d')
        print(f"嘗試抓取法人 {date_str}... ({collected}/{days})")
        df_twse = get_twse_institutional(date_str.replace('-', ''))
        if df_twse is not None:
            data_list.append(df_twse)
        df_tpex = get_tpex_institutional(date_str)
        if df_tpex is not None:
            data_list.append(df_tpex)
        if df_twse is not None or df_tpex is not None:
            collected += 1
        current -= timedelta(days=1)
        attempts += 1
        time.sleep(1.5)
    if data_list:
        all_df = pd.concat(data_list, ignore_index=True)
        all_df.sort_values(['證券代號', '日期'], ascending=[True, False], inplace=True)
        print(f"成功合併 {collected} 天上市+上櫃資料")
        return all_df
    raise ValueError("無法抓取資料")


# ===================== 月營收資料 (MOPS) =====================

def get_latest_revenue():
    """從 TWSE/TPEx 開放資料 CSV 取得最新月營收（上市 + 上櫃）"""
    csv_urls = {
        '上市': 'https://mopsfin.twse.com.tw/opendata/t187ap05_L.csv',
        '上櫃': 'https://mopsfin.twse.com.tw/opendata/t187ap05_O.csv',
    }
    all_dfs = []
    for market_label, url in csv_urls.items():
        try:
            r = requests.get(url, timeout=20, verify=False)
            text = r.content.decode('utf-8-sig')
            df = pd.read_csv(StringIO(text))
            if '公司代號' not in df.columns:
                print(f"月營收 {market_label}: 欄位格式異常，跳過")
                continue
            result = pd.DataFrame()
            result['證券代號'] = df['公司代號'].astype(str).str.strip()
            result['當月營收'] = pd.to_numeric(df['營業收入-當月營收'], errors='coerce')
            yoy_col = '營業收入-去年同月增減(%)'
            if yoy_col in df.columns:
                result['營收年增率'] = pd.to_numeric(df[yoy_col], errors='coerce') / 100.0
            mom_col = '營業收入-上月比較增減(%)'
            if mom_col in df.columns:
                result['營收月增率'] = pd.to_numeric(df[mom_col], errors='coerce') / 100.0
            cum_yoy_col = '累計營業收入-去年累計增減(%)'
            if cum_yoy_col in df.columns:
                result['累計年增率'] = pd.to_numeric(df[cum_yoy_col], errors='coerce') / 100.0
            result['市場'] = market_label
            result = result.dropna(subset=['證券代號'])
            result = result[result['證券代號'].str.match(r'^\d{4,6}$')]
            all_dfs.append(result)
            period = df['資料年月'].iloc[0] if '資料年月' in df.columns else '?'
            print(f"取得 {market_label} 月營收（{period}），{len(result)} 筆")
        except Exception as e:
            print(f"月營收 {market_label} 錯誤: {e}")
        time.sleep(1)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    print("無法取得月營收資料")
    return pd.DataFrame()


# ===================== 融資融券資料 =====================

def get_twse_margin(date_str):
    """date_str: YYYYMMDD"""
    url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json&date={date_str}&selectType=ALL"
    try:
        r = requests.get(url, timeout=10, verify=False)
        data = r.json()
        if data.get('stat') != 'OK':
            return None

        fields = data.get('fields') or data.get('fields9')
        rows = data.get('data') or data.get('data9')
        if not fields or not rows:
            for t in data.get('tables', []):
                if 'data' in t and t['data']:
                    fields = t.get('fields', [])
                    rows = t['data']
                    break
        if not fields or not rows:
            return None

        df = pd.DataFrame(rows, columns=fields)
        id_col = df.columns[0]

        margin_today = None
        for col in df.columns:
            c = str(col)
            if '融資' in c and '今' in c and '餘額' in c:
                margin_today = col
                break
        if margin_today is None:
            found = [col for col in df.columns if '融資' in str(col) and '餘額' in str(col)]
            margin_today = found[-1] if found else None
        if margin_today is None:
            return None

        result = pd.DataFrame({
            '證券代號': df[id_col].astype(str).str.strip(),
            '融資餘額': pd.to_numeric(df[margin_today].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        })
        result = result[result['證券代號'].str.match(r'^\d{4,6}$')]
        return result
    except Exception as e:
        print(f"TWSE margin {date_str} 錯誤: {e}")
        return None


def get_tpex_margin(date_str):
    """date_str: YYYY-MM-DD"""
    date_roc = _to_roc_date(date_str)
    url = f"https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&d={date_roc}&o=json"
    try:
        r = requests.get(url, timeout=10, verify=False)
        data = r.json()
        rows = data.get('aaData') or data.get('data', [])
        if not rows:
            for t in data.get('tables', []):
                if 'data' in t and t['data']:
                    rows = t['data']
                    break
        if not rows:
            return None
        # TPEx margin 典型欄位: 代號, 名稱, 前資餘額, 資買, 資賣, 現償, 今資餘額, ...
        result = pd.DataFrame({
            '證券代號': [str(row[0]).strip() for row in rows],
            '融資餘額': pd.to_numeric(
                pd.Series([str(row[6]).replace(',', '') if len(row) > 6 else '0' for row in rows]),
                errors='coerce').fillna(0).values
        })
        result = result[result['證券代號'].str.match(r'^\d{4,6}$')]
        return result
    except Exception as e:
        print(f"TPEx margin {date_str} 錯誤: {e}")
        return None


def get_recent_margin(days=5, max_attempts=20):
    data_list = []
    current = datetime.now()
    collected = 0
    attempts = 0
    while collected < days and attempts < max_attempts:
        date_str = current.strftime('%Y-%m-%d')
        print(f"嘗試抓取融資 {date_str}... ({collected}/{days})")
        dfs = []
        df_t = get_twse_margin(date_str.replace('-', ''))
        if df_t is not None:
            dfs.append(df_t)
        df_p = get_tpex_margin(date_str)
        if df_p is not None:
            dfs.append(df_p)
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined['日期'] = date_str
            data_list.append(combined)
            collected += 1
        current -= timedelta(days=1)
        attempts += 1
        time.sleep(1.5)
    if data_list:
        result = pd.concat(data_list, ignore_index=True)
        result.sort_values(['證券代號', '日期'], ascending=[True, False], inplace=True)
        print(f"成功取得 {collected} 天融資資料")
        return result
    print("無法取得融資資料")
    return pd.DataFrame()


# ===================== 技術分析（含快取） =====================

def _calc_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v.item()) if hasattr(v, 'item') else float(v)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _empty_technicals():
    return {
        'close': None, 'ma_short': None, 'ma_long': None, 'ma_status': '無資料',
        'rsi': None, 'rsi_history': [],
        'bb_upper': None, 'bb_lower': None, 'bb_width': None, 'bb_width_pctl': None,
        'bb_squeeze_days': 0,
        'volume_ratio': None, 'avg_volume_20d': None,
        'prev_high_20d': None, 'prev_close': None,
        'atr': None, 'atr_pct': None,
        'macd': None, 'macd_signal': None, 'macd_hist': None, 'macd_hist_prev': None,
        'kd_k': None, 'kd_d': None, 'k_history': [],
        'high_52w': None, 'low_52w': None, 'dist_from_52w_high': None,
        'return_20d': None, 'rs_vs_taiex': None,
        'upper_shadow_pct': None,
    }


def _compute_technicals(data):
    """從 yfinance DataFrame 計算所有技術指標（含 ATR/MACD/KD/52週高/相對強度）"""
    try:
        if data.empty:
            return _empty_technicals()
        c = data['Close']
        if c.dropna().empty:
            return _empty_technicals()

        t = _empty_technicals()

        # --- MA ---
        ma_s_series = c.rolling(MA_SHORT).mean()
        ma_l_series = c.rolling(MA_LONG).mean()
        close_v = c.iloc[-1].item()
        ma_s = _safe_float(ma_s_series.iloc[-1])
        ma_l = _safe_float(ma_l_series.iloc[-1])

        if ma_s is None or ma_l is None:
            ma_status = "無 MA 資料"
        else:
            diff = (ma_s - ma_l) / ma_l if ma_l != 0 else 0
            prev_ms = _safe_float(ma_s_series.iloc[-2]) if len(data) > 1 else ma_s
            if prev_ms is None:
                prev_ms = ma_s
            if ma_s > ma_l:
                ma_status = "金叉"
            elif abs(diff) <= GOLDEN_CROSS_THRESHOLD and ma_s > prev_ms:
                ma_status = "接近金叉"
            elif ma_s < ma_l:
                ma_status = "死叉"
            else:
                ma_status = "一般"

        t['close'] = _safe_float(close_v)
        t['ma_short'] = ma_s
        t['ma_long'] = ma_l
        t['ma_status'] = ma_status

        # --- RSI ---
        rsi_series = _calc_rsi(c)
        t['rsi'] = _safe_float(rsi_series.iloc[-1])
        t['rsi_history'] = [
            _safe_float(x) for x in rsi_series.dropna().tail(5)
            if _safe_float(x) is not None
        ]

        # --- Bollinger Bands ---
        bb_ma = c.rolling(BB_PERIOD).mean()
        bb_std = c.rolling(BB_PERIOD).std()
        bb_up = bb_ma + BB_STD * bb_std
        bb_lo = bb_ma - BB_STD * bb_std
        bb_w = (bb_up - bb_lo) / bb_ma

        t['bb_upper'] = _safe_float(bb_up.iloc[-1])
        t['bb_lower'] = _safe_float(bb_lo.iloc[-1])
        t['bb_width'] = _safe_float(bb_w.iloc[-1])

        recent_w = bb_w.dropna().tail(20)
        if len(recent_w) >= 5 and t['bb_width'] is not None:
            t['bb_width_pctl'] = float((recent_w < t['bb_width']).sum()) / len(recent_w)
            # 計算持續收斂天數（帶寬百分位 < 30% 的連續天數）
            pctl_series = bb_w.rolling(len(recent_w)).apply(
                lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
            )
            squeeze_days = 0
            for v in reversed(pctl_series.dropna().tail(20).tolist()):
                if v is not None and v < 0.30:
                    squeeze_days += 1
                else:
                    break
            t['bb_squeeze_days'] = squeeze_days

        # --- Volume ---
        if 'Volume' in data.columns and len(data['Volume'].dropna()) > 0:
            vol_avg = data['Volume'].rolling(20).mean()
            vl = _safe_float(data['Volume'].iloc[-1])
            va = _safe_float(vol_avg.iloc[-1])
            if va and va > 0:
                t['avg_volume_20d'] = va / 1000  # 轉為張
                if vl:
                    t['volume_ratio'] = vl / va

        # --- Previous 20-day high & prev close ---
        if 'High' in data.columns and len(data) > 1:
            t['prev_high_20d'] = _safe_float(data['High'].iloc[:-1].tail(20).max())
        t['prev_close'] = _safe_float(c.iloc[-2]) if len(data) > 1 else None

        # --- ATR (14) ---
        if 'High' in data.columns and 'Low' in data.columns and len(data) >= ATR_PERIOD + 1:
            high = data['High']
            low = data['Low']
            prev_close = c.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(ATR_PERIOD).mean()
            t['atr'] = _safe_float(atr_series.iloc[-1])
            if t['atr'] and t['close'] and t['close'] > 0:
                t['atr_pct'] = t['atr'] / t['close']

        # --- MACD ---
        if len(c) >= MACD_SLOW + MACD_SIGNAL:
            ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
            ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
            hist = macd_line - signal_line
            t['macd'] = _safe_float(macd_line.iloc[-1])
            t['macd_signal'] = _safe_float(signal_line.iloc[-1])
            t['macd_hist'] = _safe_float(hist.iloc[-1])
            if len(hist) >= 2:
                t['macd_hist_prev'] = _safe_float(hist.iloc[-2])

        # --- KD (Stochastic) ---
        if 'High' in data.columns and 'Low' in data.columns and len(data) >= KD_PERIOD:
            low_n = data['Low'].rolling(KD_PERIOD).min()
            high_n = data['High'].rolling(KD_PERIOD).max()
            rsv = 100 * (c - low_n) / (high_n - low_n).replace(0, np.nan)
            k_series = rsv.ewm(com=2, adjust=False).mean()
            d_series = k_series.ewm(com=2, adjust=False).mean()
            t['kd_k'] = _safe_float(k_series.iloc[-1])
            t['kd_d'] = _safe_float(d_series.iloc[-1])
            t['k_history'] = [
                _safe_float(x) for x in k_series.dropna().tail(5)
                if _safe_float(x) is not None
            ]

        # --- 52-week high/low & drawdown ---
        if 'High' in data.columns and len(data) >= 30:
            look = data.tail(250)
            t['high_52w'] = _safe_float(look['High'].max())
            t['low_52w'] = _safe_float(look['Low'].min()) if 'Low' in data.columns else None
            if t['high_52w'] and t['close']:
                t['dist_from_52w_high'] = (t['high_52w'] - t['close']) / t['high_52w']

        # --- 20 日報酬（用於相對強度）---
        if len(c) >= 21:
            ref = _safe_float(c.iloc[-21])
            if ref and ref > 0 and t['close']:
                t['return_20d'] = (t['close'] - ref) / ref

        # --- 上影線占比（當日最高點距收盤）---
        if 'High' in data.columns and t['close'] and t['close'] > 0:
            day_high = _safe_float(data['High'].iloc[-1])
            if day_high and day_high > t['close']:
                t['upper_shadow_pct'] = (day_high - t['close']) / t['close']

        return t
    except Exception:
        return _empty_technicals()


# === 批次下載（核心加速） ===

def _batch_download_technicals(stock_ids, suffix, period, batch_size):
    for i in range(0, len(stock_ids), batch_size):
        batch_ids = stock_ids[i:i + batch_size]
        tickers = [f"{sid}{suffix}" for sid in batch_ids]
        try:
            if len(tickers) == 1:
                raw = yf.download(tickers[0], period=period, progress=False)
                if not raw.empty:
                    result = _compute_technicals(raw)
                    if result['close'] is not None:
                        result['rs_vs_taiex'] = _compute_rs_vs_taiex(result)
                        _technicals_cache[batch_ids[0]] = result
            else:
                raw = yf.download(tickers, period=period, progress=False,
                                  group_by='ticker', threads=True)
                for sid, ticker in zip(batch_ids, tickers):
                    if sid in _technicals_cache:
                        continue
                    try:
                        stock_data = raw[ticker]
                        if not stock_data.empty:
                            result = _compute_technicals(stock_data)
                            if result['close'] is not None:
                                result['rs_vs_taiex'] = _compute_rs_vs_taiex(result)
                                _technicals_cache[sid] = result
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            print(f"  批次下載錯誤: {e}")
        done = min(i + batch_size, len(stock_ids))
        print(f"  進度: {done}/{len(stock_ids)}")


def prefetch_technicals(stock_ids, period='1y', batch_size=50):
    """批次下載所有候選股票的技術指標（比逐支下載快 20 倍以上）"""
    all_ids = set(str(sid).strip() for sid in stock_ids)
    fetchable = [sid for sid in all_ids if _is_stock_or_etf(sid) and sid not in _technicals_cache]
    skipped = len(all_ids) - len(fetchable) - sum(1 for sid in all_ids if sid in _technicals_cache)
    if skipped > 0:
        print(f"跳過 {skipped} 個非股票代碼（權證/債券等）")
    # 直接標記不可抓的代碼為空
    for sid in all_ids:
        if not _is_stock_or_etf(sid) and sid not in _technicals_cache:
            _technicals_cache[sid] = _empty_technicals()
    to_fetch = fetchable
    if not to_fetch:
        return
    print(f"\n批次下載 {len(to_fetch)} 支股票技術指標...")
    _batch_download_technicals(to_fetch, '.TW', period, batch_size)

    remaining = [sid for sid in to_fetch if sid not in _technicals_cache]
    if remaining:
        print(f"  嘗試上櫃代碼 ({len(remaining)} 支)...")
        _batch_download_technicals(remaining, '.TWO', period, batch_size)

    for sid in to_fetch:
        if sid not in _technicals_cache:
            _technicals_cache[sid] = _empty_technicals()
    print(f"技術指標下載完成，快取 {len(_technicals_cache)} 支\n")


def get_taiex_state():
    """下載 TAIEX (^TWII) 判斷大盤多空狀態。
    回傳 dict: {'bullish': bool, 'close': float, 'ma_60': float, 'return_20d': float, 'desc': str}"""
    if _taiex_returns_cache.get('desc') is not None:
        return _taiex_returns_cache
    try:
        data = yf.download('^TWII', period='1y', progress=False)
        if data.empty:
            raise ValueError('no data')
        close = data['Close']
        if hasattr(close, 'columns'):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 60:
            raise ValueError('insufficient history')
        ma_60 = close.rolling(60).mean().iloc[-1]
        ma_20 = close.rolling(20).mean().iloc[-1]
        ret_20 = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if len(close) > 21 else 0
        c_now = float(close.iloc[-1])
        ma_60_v = float(ma_60) if pd.notna(ma_60) else None
        ma_20_v = float(ma_20) if pd.notna(ma_20) else None
        bullish = (ma_60_v is not None and c_now > ma_60_v) or (ma_20_v is not None and c_now > ma_20_v and ret_20 > 0)
        desc = '多頭' if bullish else '空頭/震盪'
        _taiex_returns_cache.update({
            'bullish': bullish, 'close': c_now, 'ma_60': ma_60_v,
            'ma_20': ma_20_v, 'return_20d': float(ret_20), 'desc': desc,
        })
    except Exception as e:
        print(f"大盤資料擷取失敗: {e}")
        _taiex_returns_cache.update({
            'bullish': True, 'close': None, 'ma_60': None, 'ma_20': None,
            'return_20d': 0.0, 'desc': '未知（預設多頭）',
        })
    return _taiex_returns_cache


def is_market_bullish():
    state = get_taiex_state()
    return bool(state.get('bullish', True))


def _compute_rs_vs_taiex(t):
    """將個股 20 日報酬轉為相對於大盤的比值"""
    market = get_taiex_state()
    m_ret = market.get('return_20d')
    s_ret = t.get('return_20d')
    if m_ret is None or s_ret is None:
        return None
    if abs(1 + m_ret) < 1e-6:
        return None
    return (1 + s_ret) / (1 + m_ret)


def get_stock_technicals(stock_id, period='1y'):
    if stock_id in _technicals_cache:
        return _technicals_cache[stock_id]
    ticker = f"{stock_id}.TW"
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        ticker = f"{stock_id}.TWO"
        data = yf.download(ticker, period=period, progress=False)
    result = _compute_technicals(data)
    _technicals_cache[stock_id] = result
    return result


def get_stock_price_ma(stock_id, period='1y'):
    t = get_stock_technicals(stock_id, period)
    return t['close'], t['ma_short'], t['ma_long'], t['ma_status']


# ===================== 策略 1-3（法人 + MA） =====================

def _check_ma_ok(t):
    if t['ma_long'] is None:
        return False
    if t['close'] > t['ma_long']:
        return True
    if t['ma_short'] is not None and t['ma_short'] > t['ma_long']:
        return True
    return False


def _is_liquid(t, min_avg_volume=MIN_AVG_VOLUME):
    """流動性過濾：日均量 >= 門檻（張）。
    若技術指標未能計算（資料不足），保守保留（回傳 True）。"""
    av = t.get('avg_volume_20d')
    if av is None:
        return True
    return av >= min_avg_volume


def _bb_position(t):
    if t['bb_upper'] is None or t['bb_lower'] is None or t['close'] is None:
        return '無資料'
    if t['close'] >= t['bb_upper']:
        return '上軌之上'
    if t['close'] <= t['bb_lower']:
        return '下軌之下'
    mid = (t['bb_upper'] + t['bb_lower']) / 2
    return '中軌以上' if t['close'] >= mid else '中軌以下'


def _buy_volume_ratio(total, t):
    """法人買超占 20 日均量百分比。"""
    av = t.get('avg_volume_20d')
    if not av or av <= 0:
        return None
    return total / av  # total(張) / 日均量(張)


def _is_accelerating(series):
    """最近一日買超 >= 前 (n-1) 日平均買超，代表動能加速。"""
    if len(series) < 2:
        return True
    prev_avg = series.iloc[1:].mean() if len(series) > 1 else 0
    return float(series.iloc[0]) >= prev_avg * 0.9


def strategy1(all_df):
    """外資連續買超 + MA（含動能加速 + 買超占日均量過濾）"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(CONSEC_BUY_DAYS)
        if len(recent) < CONSEC_BUY_DAYS:
            continue
        if not (recent['外資買賣超'] > 0).all():
            continue
        total = recent['外資買賣超'].sum()
        if total < MIN_BUY_THRESHOLD * CONSEC_BUY_DAYS:
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or not _check_ma_ok(t):
            continue
        if not _is_liquid(t):
            continue
        ratio = _buy_volume_ratio(total, t)
        if ratio is not None and ratio < 0.05:
            continue
        if not _is_accelerating(recent['外資買賣超']):
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略1',
            '總外資買超': int(total),
            '買超/日均量%': round(ratio * 100, 1) if ratio is not None else None,
            '最新收盤': t['close'],
            'MA_short': t['ma_short'], 'MA_long': t['ma_long'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('總外資買超', ascending=False, inplace=True)
    return df


def strategy2(all_df):
    """投信連續買超 + MA（含動能加速 + 買超占日均量過濾）"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(CONSEC_BUY_DAYS)
        if len(recent) < CONSEC_BUY_DAYS:
            continue
        if not (recent['投信買賣超'] > 0).all():
            continue
        total = recent['投信買賣超'].sum()
        if total < MIN_BUY_THRESHOLD * CONSEC_BUY_DAYS:
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or not _check_ma_ok(t):
            continue
        if not _is_liquid(t):
            continue
        ratio = _buy_volume_ratio(total, t)
        if ratio is not None and ratio < 0.03:
            continue
        if not _is_accelerating(recent['投信買賣超']):
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略2',
            '總投信買超': int(total),
            '買超/日均量%': round(ratio * 100, 1) if ratio is not None else None,
            '最新收盤': t['close'],
            'MA_short': t['ma_short'], 'MA_long': t['ma_long'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('總投信買超', ascending=False, inplace=True)
    return df


def strategy3(all_df):
    """三法人共識買超 + MA（含流動性過濾）"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        cond = (recent['外資買賣超'] > 0) & (recent['投信買賣超'] > 0) & (recent['自營商買賣超'] > 0)
        if not cond.all():
            continue
        total = recent['三大法人買賣超'].sum()
        t = get_stock_technicals(stock_id)
        if t['close'] is None or not _check_ma_ok(t):
            continue
        if not _is_liquid(t):
            continue
        ratio = _buy_volume_ratio(total, t)
        if ratio is not None and ratio < 0.05:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略3',
            '總三大買超': int(total),
            '買超/日均量%': round(ratio * 100, 1) if ratio is not None else None,
            '最新收盤': t['close'],
            'MA_short': t['ma_short'], 'MA_long': t['ma_long'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('總三大買超', ascending=False, inplace=True)
    return df


# ===================== 策略 4：量價齊揚突破 =====================

def strategy4(all_df):
    """量價齊揚突破（含 52 週高點位置 + 上影線過濾 + 流動性）"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        has_buy = (recent['外資買賣超'] > 0).any() or (recent['投信買賣超'] > 0).any()
        if not has_buy:
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t['volume_ratio'] is None:
            continue
        if not _is_liquid(t):
            continue
        if t['volume_ratio'] < VOLUME_SURGE_RATIO:
            continue
        if t['prev_high_20d'] is None or t['close'] < t['prev_high_20d']:
            continue
        if t['ma_short'] is None or t['ma_long'] is None or t['ma_short'] <= t['ma_long']:
            continue
        # 52 週高點：避免追高已經漲過頭的股票（位於 90% 以內）
        if t.get('dist_from_52w_high') is not None and t['dist_from_52w_high'] < 0.03 \
                and t.get('high_52w') and t['close'] >= t['high_52w'] * 0.98:
            # 若已在 52 週高點附近，要求量比更大才進場
            if t['volume_ratio'] < VOLUME_SURGE_RATIO * 1.5:
                continue
        # 過濾上影線（收盤距當日高點 > 3%，代表追高被殺）
        if t.get('upper_shadow_pct') is not None and t['upper_shadow_pct'] > 0.03:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略4',
            '量比': round(t['volume_ratio'], 2),
            '距52週高': f"{t['dist_from_52w_high']*100:.1f}%" if t.get('dist_from_52w_high') is not None else '-',
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
            'RSI': round(t['rsi'], 1) if t['rsi'] else None,
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('量比', ascending=False, inplace=True)
    return df


# ===================== 策略 5：RSI 超賣反彈 + 法人進場 =====================

def strategy5(all_df):
    """RSI 超賣反彈 + MACD/KD 任一底部確認 + 法人轉買"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if not (recent['三大法人買賣超'] > 0).any():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t['rsi'] is None or len(t['rsi_history']) < 3:
            continue
        if not _is_liquid(t):
            continue
        was_oversold = any(r < RSI_OVERSOLD for r in t['rsi_history'][:-1])
        now_recovered = t['rsi'] >= RSI_OVERSOLD
        if not (was_oversold and now_recovered):
            continue
        # 站回 MA_long 或接近（放寬至 3% 內，避免錯過剛突破的股）
        if t['ma_long'] is None:
            continue
        if t['close'] < t['ma_long'] * 0.97:
            continue
        # MACD 或 KD 任一確認底部
        confirmations = []
        if (t.get('macd_hist') is not None and t.get('macd_hist_prev') is not None
                and t['macd_hist'] > t['macd_hist_prev']):
            confirmations.append('MACD 紅柱擴大')
        if t.get('kd_k') is not None and t.get('kd_d') is not None \
                and t['kd_k'] > t['kd_d'] and t['kd_k'] < 50:
            confirmations.append('KD 黃金交叉')
        if not confirmations:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略5',
            'RSI': round(t['rsi'], 1), '最新收盤': t['close'],
            'MA 狀態': t['ma_status'], '布林位置': _bb_position(t),
            '底部訊號': '+'.join(confirmations),
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('RSI', ascending=True, inplace=True)
    return df


# ===================== 策略 6：布林通道收斂突破 =====================

def strategy6(all_df):
    """布林通道收斂（>=5 日）+ 突破上軌 + 法人買超 + 流動性"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if not (recent['三大法人買賣超'] > 0).any():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t['bb_upper'] is None or t['bb_width_pctl'] is None:
            continue
        if not _is_liquid(t):
            continue
        if t['bb_width_pctl'] > 0.25:
            continue
        # 要求持續收斂至少 5 日
        if t.get('bb_squeeze_days', 0) < 5:
            continue
        if t['close'] < t['bb_upper']:
            continue
        # 過濾上影線過長（追高失敗）
        if t.get('upper_shadow_pct') is not None and t['upper_shadow_pct'] > 0.03:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略6',
            '帶寬百分位': round(t['bb_width_pctl'] * 100, 1),
            '收斂天數': t.get('bb_squeeze_days', 0),
            '最新收盤': t['close'],
            '布林上軌': round(t['bb_upper'], 2),
            'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('帶寬百分位', ascending=True, inplace=True)
    return df


# ===================== 策略 7：月營收創高 + 法人買超 =====================

def strategy7(all_df, revenue_df):
    """多層營收過濾：YoY > 20% + 累計 YoY > 10% + MoM > 0 + 法人買超 + MA 金叉"""
    if revenue_df is None or revenue_df.empty:
        return pd.DataFrame()
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(CONSEC_BUY_DAYS)
        if len(recent) < CONSEC_BUY_DAYS:
            continue
        if not (recent['三大法人買賣超'] > 0).any():
            continue
        rev = revenue_df[revenue_df['證券代號'] == stock_id]
        if rev.empty:
            continue
        row = rev.iloc[0]
        yoy = row.get('營收年增率')
        if pd.isna(yoy) or yoy < REVENUE_GROWTH_THRESHOLD:
            continue
        # 累計年增率 > 10%（排除單月跳升但整年不佳的）
        cum_yoy = row.get('累計年增率')
        if pd.notna(cum_yoy) and cum_yoy < 0.10:
            continue
        # 月增率 > 0（排除單月暴衝但下個月就回落的風險）
        mom = row.get('營收月增率')
        if pd.notna(mom) and mom < -0.15:
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None:
            continue
        if not _is_liquid(t):
            continue
        if t['ma_short'] is None or t['ma_long'] is None or t['ma_short'] <= t['ma_long']:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略7',
            '營收年增率': f"{yoy * 100:.1f}%",
            '累計年增率': f"{cum_yoy * 100:.1f}%" if pd.notna(cum_yoy) else '-',
            '月增率': f"{mom * 100:.1f}%" if pd.notna(mom) else '-',
            '_yoy': yoy,
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
            'RSI': round(t['rsi'], 1) if t['rsi'] else None,
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('_yoy', ascending=False, inplace=True)
        df.drop(columns=['_yoy'], inplace=True)
    return df


# ===================== 策略 8：融資減少 + 法人買超 =====================

def strategy8(all_df, margin_df):
    """融資減少（或融券增加→軋空）+ 法人買超 + 股價在 MA_long 之上"""
    if margin_df is None or margin_df.empty:
        return pd.DataFrame()
    grouped_inst = all_df.groupby('證券代號')
    grouped_margin = margin_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped_inst:
        recent = group.head(CONSEC_BUY_DAYS)
        if len(recent) < CONSEC_BUY_DAYS:
            continue
        if not (recent['三大法人買賣超'] > 0).all():
            continue
        if stock_id not in grouped_margin.groups:
            continue
        mg = grouped_margin.get_group(stock_id).head(CONSEC_BUY_DAYS)
        if len(mg) < CONSEC_BUY_DAYS:
            continue
        balances = mg['融資餘額'].tolist()
        signals = []
        margin_drop = 0
        short_rise = 0
        decreasing = all(balances[i] < balances[i + 1] for i in range(len(balances) - 1))
        if decreasing:
            margin_drop = abs(balances[-1] - balances[0])
            signals.append('融資連續減少')
        # 融券增加（軋空訊號）
        if '融券餘額' in mg.columns:
            shorts = mg['融券餘額'].tolist()
            if len(shorts) >= 2 and shorts[0] > shorts[-1] * 1.1:
                short_rise = int(shorts[0] - shorts[-1])
                signals.append('融券增加→軋空')
        if not signals:
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t['ma_long'] is None or t['close'] < t['ma_long']:
            continue
        if not _is_liquid(t):
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略8',
            '融資減少張數': int(margin_drop),
            '融券增加張數': int(short_rise),
            '訊號': '+'.join(signals),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
            'RSI': round(t['rsi'], 1) if t['rsi'] else None,
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values(['融資減少張數', '融券增加張數'], ascending=False, inplace=True)
    return df


# ===================== 策略 9：MACD 金叉 + 紅柱放大 =====================

def strategy9(all_df):
    """MACD 金叉（macd > signal）且紅柱連續擴大 + 站上 MA_long + 法人不賣超"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        # 法人近兩日三大合計不賣超
        if (recent['三大法人買賣超'] < 0).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('macd') is None or t.get('macd_signal') is None:
            continue
        if not _is_liquid(t):
            continue
        # MACD 金叉
        if t['macd'] <= t['macd_signal']:
            continue
        # 紅柱擴大
        if t.get('macd_hist') is None or t.get('macd_hist_prev') is None:
            continue
        if not (t['macd_hist'] > t['macd_hist_prev'] and t['macd_hist'] > 0):
            continue
        # 站上 MA_long
        if t['ma_long'] is None or t['close'] < t['ma_long']:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略9',
            'MACD': round(t['macd'], 3),
            'Signal': round(t['macd_signal'], 3),
            '柱值': round(t['macd_hist'], 3),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('柱值', ascending=False, inplace=True)
    return df


# ===================== 策略 10：KD 低檔黃金交叉 =====================

def strategy10(all_df):
    """KD 於低檔（K<50）發生黃金交叉，股價位於 MA_long 附近（±3%）"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < 0).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('kd_k') is None or t.get('kd_d') is None:
            continue
        if not _is_liquid(t):
            continue
        k_hist = t.get('k_history') or []
        if len(k_hist) < 2:
            continue
        # 昨 K < D, 今 K > D 且 K < 50
        if not (t['kd_k'] > t['kd_d'] and t['kd_k'] < 50):
            continue
        if not (k_hist[-2] < k_hist[-1]):
            continue
        # 股價接近或站上 MA_long
        if t['ma_long'] is None or t['close'] < t['ma_long'] * 0.97:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略10',
            'K': round(t['kd_k'], 1), 'D': round(t['kd_d'], 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
            'RSI': round(t['rsi'], 1) if t['rsi'] else None,
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('K', ascending=True, inplace=True)
    return df


# ===================== 策略 11：Darvas 盒突破（52週高） =====================

def strategy11(all_df):
    """突破近 52 週高點（或距離 <2%）+ 量能放大 + 法人買超 + 非追高"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if not (recent['三大法人買賣超'] > 0).any():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('high_52w') is None:
            continue
        if not _is_liquid(t):
            continue
        dist = t.get('dist_from_52w_high')
        if dist is None or dist > 0.03:
            continue
        if t['volume_ratio'] is None or t['volume_ratio'] < 1.5:
            continue
        # MA 多頭排列
        if t['ma_short'] is None or t['ma_long'] is None or t['ma_short'] <= t['ma_long']:
            continue
        # 過濾長上影線
        if t.get('upper_shadow_pct') is not None and t['upper_shadow_pct'] > 0.03:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略11',
            '52週高': round(t['high_52w'], 2),
            '距高%': round(dist * 100, 2),
            '量比': round(t['volume_ratio'], 2),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('距高%', ascending=True, inplace=True)
    return df


# ===================== 策略 12：底部價量背離 =====================

def strategy12(all_df):
    """近期下跌但量能縮至平均以下 + RSI 較前波高（背離），法人止跌或買入"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(3)
        if len(recent) < 3:
            continue
        # 法人不連續大幅賣超
        if (recent['三大法人買賣超'] < 0).all():
            if abs(recent['三大法人買賣超'].sum()) > 2000:
                continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('rsi') is None or t.get('prev_close') is None:
            continue
        if not _is_liquid(t):
            continue
        # 價跌 / 持平
        if t['close'] > t['prev_close'] * 1.01:
            continue
        # 量縮
        if t['volume_ratio'] is None or t['volume_ratio'] > 0.8:
            continue
        # RSI 未創新低（前 5 日最低 < 當前），但 RSI 在中低區（< 55）
        rhist = t.get('rsi_history') or []
        if len(rhist) < 3:
            continue
        min_prev_rsi = min(rhist[:-1])
        if not (t['rsi'] > min_prev_rsi + 2 and t['rsi'] < 55):
            continue
        # 股價接近 MA_long 下方（底部區）
        if t['ma_long'] is None or t['close'] > t['ma_long'] * 1.02:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略12',
            '量比': round(t['volume_ratio'], 2),
            'RSI': round(t['rsi'], 1),
            '前波最低RSI': round(min_prev_rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('RSI', ascending=True, inplace=True)
    return df


# ===================== 策略 13：相對強度領漲（RS vs TAIEX） =====================

def strategy13(all_df):
    """股價 20 日報酬 vs TAIEX 相對強度 > 1.10，MA 多頭 + 法人買超"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if not (recent['三大法人買賣超'] > 0).any():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('rs_vs_taiex') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['rs_vs_taiex'] < 1.10:
            continue
        if t['ma_short'] is None or t['ma_long'] is None or t['ma_short'] <= t['ma_long']:
            continue
        # 避免追高：52 週高點之下 3% 以內時要求法人買超加強
        dist = t.get('dist_from_52w_high')
        if dist is not None and dist < 0.02:
            if not (recent['三大法人買賣超'] > 0).all():
                continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略13',
            '相對強度': round(t['rs_vs_taiex'], 3),
            '20日報酬': f"{t['return_20d']*100:.1f}%" if t.get('return_20d') is not None else '-',
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('相對強度', ascending=False, inplace=True)
    return df


def _estimate_target_price(t):
    """
    根據技術面指標預估短期目標價。

    方法：以布林中軌（20 日均線）為合理回歸目標，
    加權考量 20 日高點與 MA 趨勢，給出保守 / 樂觀目標。
    回傳 (目標價, '漲幅%字串')。
    """
    close = t['close']
    if close is None or close <= 0:
        return None, None

    targets = []

    # 布林中軌回歸：低估股通常低於中軌，回歸中軌是第一目標
    if t['bb_upper'] is not None and t['bb_lower'] is not None:
        bb_mid = (t['bb_upper'] + t['bb_lower']) / 2
        targets.append(bb_mid)

    # 20 日高點的 85% 位置（保守回彈目標）
    if t['prev_high_20d'] is not None and t['prev_high_20d'] > close:
        targets.append(t['prev_high_20d'] * 0.85)

    # MA 均線回歸
    if t['ma_long'] is not None and t['ma_long'] > close:
        targets.append(t['ma_long'])

    # RSI 回歸加成：RSI 越低，回彈空間越大
    if t['rsi'] is not None and t['rsi'] < 40:
        rsi_boost = 1 + (40 - t['rsi']) / 200  # RSI=20 → +10%, RSI=30 → +5%
        targets = [tgt * rsi_boost for tgt in targets] if targets else []

    if not targets:
        return None, None

    target = round(sum(targets) / len(targets), 2)
    if target <= close:
        return None, None

    upside = round((target - close) / close * 100, 1)
    return target, f'+{upside}%'


# ===================== 潛力股提前佈局 =====================

def find_early_potential(all_df):
    """
    找出「尚未發動但正在蓄勢」的潛力股。

    偵測五大提前訊號，每滿足一項 +1 分，至少 3 分列入：
    1. 量能蓄積：近 3 天量比逐日攀升但股價波動 < 2%（主力悄悄吃貨）
    2. 法人試探性買超：最近 1-2 天法人小量買超（尚未達連 3 天門檻）
    3. 布林極度收斂：帶寬百分位 < 20%，即將噴發
    4. 均線糾結即將突破：MA5 與 MA10 差距 < 1% 且 MA5 趨勢向上
    5. RSI 底部回升：RSI 從 < 35 回升中，尚未到 50（動能正在累積）
    """
    grouped = all_df.groupby('證券代號')
    results = []

    for stock_id, group in grouped:
        t = _technicals_cache.get(stock_id, _empty_technicals())
        if t['close'] is None or t['rsi'] is None:
            continue

        recent = group.head(CONSEC_BUY_DAYS)
        if len(recent) < 2:
            continue

        signals = []
        score = 0

        # 1) 量能蓄積：量比在增加但股價平穩
        if t['volume_ratio'] is not None and t['prev_close'] is not None:
            price_chg = abs(t['close'] - t['prev_close']) / t['prev_close']
            if 0.8 < t['volume_ratio'] < 2.0 and price_chg < 0.02:
                signals.append(f'量能蓄積 (量比{t["volume_ratio"]:.1f}, 波動{price_chg*100:.1f}%)')
                score += 1

        # 2) 法人試探性買超：1-2 天有買但未達連 3 天
        inst_recent = recent.head(3)
        buy_days = (inst_recent['三大法人買賣超'] > 0).sum()
        if 1 <= buy_days <= 2:
            total_buy = inst_recent.loc[inst_recent['三大法人買賣超'] > 0, '三大法人買賣超'].sum()
            signals.append(f'法人試探買超 ({buy_days}天, {int(total_buy)}張)')
            score += 1

        # 3) 布林極度收斂
        if t['bb_width_pctl'] is not None and t['bb_width_pctl'] < 0.20:
            signals.append(f'布林收斂 (百分位{t["bb_width_pctl"]*100:.0f}%)')
            score += 1

        # 4) 均線糾結即將突破：MA5 接近 MA10 且趨勢向上
        if t['ma_short'] is not None and t['ma_long'] is not None and t['ma_long'] > 0:
            ma_gap = abs(t['ma_short'] - t['ma_long']) / t['ma_long']
            if ma_gap < 0.01 and t['ma_status'] in ('接近金叉', '一般'):
                signals.append(f'均線糾結 (差距{ma_gap*100:.2f}%)')
                score += 1

        # 5) RSI 底部回升中
        rsi_hist = t.get('rsi_history', [])
        if t['rsi'] is not None and 30 <= t['rsi'] < 50:
            if len(rsi_hist) >= 2 and rsi_hist[-1] > rsi_hist[-2]:
                signals.append(f'RSI 回升中 ({t["rsi"]:.1f})')
                score += 1

        if score >= 3:
            target_price, upside = _estimate_target_price(t)
            results.append({
                '證券代號': stock_id,
                '證券名稱': group['證券名稱'].iloc[0],
                '市場': group['市場'].iloc[0],
                '潛力分數': score,
                '蓄勢訊號': '; '.join(signals),
                'RSI': round(t['rsi'], 1),
                'MA 狀態': t['ma_status'],
                '量比': round(t['volume_ratio'], 2) if t['volume_ratio'] else None,
                '最新收盤': t['close'],
                '預測目標價': target_price,
                '預估漲幅': upside,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('潛力分數', ascending=False, inplace=True)
    return df


# ===================== 綜合買入潛力排名 =====================

def compute_composite_ranking(buy_dfs_map, all_df, top_n=30, backtest_scores=None):
    """跨策略加權評分，輸出最具買入潛力 TOP N。
    buy_dfs_map: dict {策略名稱: DataFrame}
    backtest_scores: 可選 dict {策略名稱: 綜效分數 0~1}，用於動態加權
    """
    # 策略類別分組（用於相關性懲罰：同類多命中不加分過多）
    strategy_groups = {
        '法人': {'策略1', '策略2', '策略3'},
        '量價': {'策略4', '策略6', '策略11'},
        '反彈': {'策略5', '策略10', '策略12'},
        '基本面': {'策略7'},
        '籌碼': {'策略8'},
        '技術面': {'策略9', '策略13'},
    }

    # 策略權重（可由回測動態調整）
    default_weights = {
        '策略1': 15, '策略2': 15, '策略3': 18,
        '策略4': 14, '策略5': 12, '策略6': 13,
        '策略7': 18, '策略8': 12,
        '策略9': 12, '策略10': 10, '策略11': 14,
        '策略12': 10, '策略13': 14,
    }
    weights = default_weights.copy()
    if backtest_scores:
        for sname, adj in backtest_scores.items():
            if sname in weights:
                weights[sname] = int(weights[sname] * (0.6 + 0.8 * float(adj)))

    # 收集每支股票被哪些策略命中
    hit_map = {}
    name_map = {}
    market_map = {}
    for sname, df in buy_dfs_map.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            sid = row['證券代號']
            hit_map.setdefault(sid, []).append(sname)
            name_map[sid] = row.get('證券名稱', '')
            market_map[sid] = row.get('市場', '')

    if not hit_map:
        return pd.DataFrame()

    # 計算三大法人買賣超百分位（用於法人力道分數）
    grouped = all_df.groupby('證券代號')
    inst_totals = {}
    for sid in hit_map:
        if sid in grouped.groups:
            g = grouped.get_group(sid).head(3)
            inst_totals[sid] = g['三大法人買賣超'].sum()
    if inst_totals:
        vals = pd.Series(inst_totals)
        inst_pctl = vals.rank(pct=True)
    else:
        inst_pctl = pd.Series(dtype=float)

    results = []
    for sid, strategies in hit_map.items():
        t = _technicals_cache.get(sid, _empty_technicals())
        if t['close'] is None:
            continue

        score = 0

        # 策略命中分數（同類別策略加分遞減，降低相關性重複）
        hit_count = len(strategies)
        group_hits = {}
        for s in strategies:
            for gname, gset in strategy_groups.items():
                if s in gset:
                    group_hits.setdefault(gname, []).append(s)
                    break
        for _gname, slist in group_hits.items():
            slist_sorted = sorted(slist, key=lambda x: weights.get(x, 10), reverse=True)
            for idx, sname in enumerate(slist_sorted):
                w = weights.get(sname, 10)
                # 同類內第 2 個 50%，第 3 個以後 30%
                factor = 1.0 if idx == 0 else (0.5 if idx == 1 else 0.3)
                score += w * factor

        # 跨類別多命中獎勵（真正強勢股）
        if len(group_hits) >= 3:
            score += 10
        elif len(group_hits) >= 2:
            score += 5

        # 法人力道 (0-25)
        if sid in inst_pctl.index:
            score += round(inst_pctl[sid] * 25)

        # 技術面動能 (0-25)
        if t['ma_status'] == '金叉':
            score += 10
        elif t['ma_status'] == '接近金叉':
            score += 5
        if t['rsi'] is not None:
            if 40 <= t['rsi'] <= 70:
                score += 5
            elif t['rsi'] < 40 and len(t['rsi_history']) >= 2 and t['rsi_history'][-1] > t['rsi_history'][-2]:
                score += 8
        if t['volume_ratio'] is not None:
            if t['volume_ratio'] >= 2.0:
                score += 7
            elif t['volume_ratio'] >= 1.5:
                score += 5

        # 布林位置 (0-10)
        if t['bb_upper'] is not None and t['close'] >= t['bb_upper']:
            score += 10
        elif t['bb_upper'] is not None and t['bb_lower'] is not None:
            mid = (t['bb_upper'] + t['bb_lower']) / 2
            if t['close'] >= mid:
                score += 5

        # 相對強度加分 (0-8)
        rs = t.get('rs_vs_taiex')
        if rs is not None:
            if rs >= 1.15:
                score += 8
            elif rs >= 1.05:
                score += 4

        # MACD 紅柱擴大加分 (0-5)
        if t.get('macd_hist') is not None and t.get('macd_hist_prev') is not None:
            if t['macd_hist'] > t['macd_hist_prev'] and t['macd_hist'] > 0:
                score += 5

        # 52 週高點距離：過近（<3%）扣分避免追高，適中（5-15%）加分
        dist = t.get('dist_from_52w_high')
        if dist is not None:
            if 0.05 <= dist <= 0.15:
                score += 5
            elif dist < 0.02:
                score -= 3

        # 空頭大盤懲罰
        if not is_market_bullish():
            score = int(score * 0.75)

        results.append({
            '證券代號': sid,
            '證券名稱': name_map.get(sid, ''),
            '市場': market_map.get(sid, ''),
            '綜合分數': int(score),
            '命中策略數': hit_count,
            '命中策略': ', '.join(strategies),
            'MA 狀態': t['ma_status'],
            'RSI': round(t['rsi'], 1) if t['rsi'] else None,
            '量比': round(t['volume_ratio'], 2) if t['volume_ratio'] else None,
            '相對強度': round(rs, 2) if rs is not None else None,
            '最新收盤': t['close'],
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('綜合分數', ascending=False, inplace=True)
        df = df.head(top_n).reset_index(drop=True)
        df.insert(0, '排名', range(1, len(df) + 1))
    return df


# ===================== ML Ranker 融合（路線 2） =====================

def apply_ml_ranking(ranking_df, candidate_pool_df=None, top_n=30, ml_weight=0.6):
    """若 ranker_model.json 存在，將 XGBoost ML 分數與規則分數加權融合。

    ranking_df: compute_composite_ranking 產出（已有 排名/綜合分數 等欄位）
    candidate_pool_df: 可選的更大候選池（如全部命中任一策略的股票），
                      用於讓 ML 對更多標的做分數，避免只對 TOP 30 排名
    ml_weight: ML 分數在混合公式中的權重（0~1）

    回傳 (enhanced_df, used_ml: bool)
    """
    try:
        import ml_ranker as mlr
    except Exception as e:
        print(f"  ⚠ ml_ranker 模組無法載入: {e}")
        return ranking_df, False

    model = mlr.load_model()
    if model is None:
        return ranking_df, False
    if ranking_df is None or ranking_df.empty:
        return ranking_df, False

    # 候選池：優先用 candidate_pool_df，其次用 ranking_df 裡的股票
    if candidate_pool_df is not None and not candidate_pool_df.empty \
            and '證券代號' in candidate_pool_df.columns:
        candidate_ids = (candidate_pool_df['證券代號']
                         .astype(str).str.strip().unique().tolist())
    else:
        candidate_ids = ranking_df['證券代號'].astype(str).str.strip().tolist()

    if not candidate_ids:
        return ranking_df, False

    print(f"  載入 ML 模型 ({mlr.MODEL_FILE})，為 {len(candidate_ids)} 檔候選股計算分數...")
    try:
        features = mlr.build_live_features(candidate_ids, period='6mo')
    except Exception as e:
        print(f"  ⚠ ML 特徵建構失敗: {e}")
        return ranking_df, False

    if features.empty:
        print("  ⚠ 無可用 ML 特徵")
        return ranking_df, False

    try:
        scores = mlr.predict_scores(model, features)
    except Exception as e:
        print(f"  ⚠ ML 推論失敗: {e}")
        return ranking_df, False

    ml_map = dict(zip(features['stock_id'].astype(str).str.strip(),
                      scores.astype(float)))

    # 把 ML 分數歸一化到 0~100
    import numpy as _np
    vals = _np.array(list(ml_map.values()), dtype=float)
    if len(vals) > 1 and vals.max() > vals.min():
        lo, hi = float(vals.min()), float(vals.max())
        ml_norm = {sid: (s - lo) / (hi - lo) * 100 for sid, s in ml_map.items()}
    else:
        ml_norm = {sid: 50.0 for sid in ml_map}

    # 規則分數歸一化
    rule_max = float(ranking_df['綜合分數'].max()) or 1.0

    enhanced = ranking_df.copy()
    sid_key = enhanced['證券代號'].astype(str).str.strip()
    enhanced['ML 分數'] = sid_key.map(lambda s: round(ml_norm.get(s, 0.0), 1))
    enhanced['規則分數'] = enhanced['綜合分數']

    def _blend(row):
        sid = str(row['證券代號']).strip()
        rule_norm = float(row['綜合分數']) / rule_max * 100
        if sid in ml_norm:
            return rule_norm * (1 - ml_weight) + ml_norm[sid] * ml_weight
        return rule_norm

    enhanced['混合分數'] = enhanced.apply(_blend, axis=1).round(1)
    enhanced = enhanced.sort_values('混合分數', ascending=False).reset_index(drop=True)
    enhanced = enhanced.head(top_n).copy()
    enhanced['排名'] = range(1, len(enhanced) + 1)

    # 欄位順序：把 ML/混合 放在分數後面
    cols = enhanced.columns.tolist()
    desired_front = ['排名', '證券代號', '證券名稱', '市場',
                     '混合分數', 'ML 分數', '規則分數',
                     '命中策略數', '命中策略',
                     'MA 狀態', 'RSI', '量比', '相對強度', '最新收盤']
    ordered = [c for c in desired_front if c in cols] + \
              [c for c in cols if c not in desired_front and c != '綜合分數']
    enhanced = enhanced[ordered]

    matched = sum(1 for s in sid_key if s in ml_map)
    print(f"  ML 融合完成：{matched}/{len(enhanced)} 檔有 ML 分數 "
          f"(權重 rule:{1-ml_weight:.1f} / ml:{ml_weight:.1f})")
    return enhanced, True


# ===================== Deep Ranker 融合（路線 3） =====================

def apply_deep_ranking(ranking_df, top_n=30, deep_weight=0.35):
    """若 deep_model.pt 存在，把 GRU 深度模型預測分數加入三層融合。

    ranking_df 需至少含 '證券代號' 與（若存在）'混合分數' 或 '綜合分數'。
    回傳 (new_df, used_deep: bool)
    """
    try:
        import deep_ranker as dr
    except Exception as e:
        print(f"  ⚠ deep_ranker 模組無法載入: {e}")
        return ranking_df, False

    if ranking_df is None or ranking_df.empty:
        return ranking_df, False

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'

    model = dr.load_model(device=device)
    if model is None:
        return ranking_df, False

    sids = ranking_df['證券代號'].astype(str).str.strip().tolist()
    print(f"  載入 Deep 模型 ({dr.MODEL_FILE})，計算 {len(sids)} 檔序列分數...")
    try:
        X, ok_sids, _dates, _closes = dr.build_live_windows(sids, period='9mo')
        if len(X) == 0:
            print("  ⚠ 無可用序列視窗")
            return ranking_df, False
        pred_r, up_p = dr.predict_scores(model, X, device=device)
    except Exception as e:
        print(f"  ⚠ Deep 推論失敗: {e}")
        return ranking_df, False

    # deep_score 歸一化到 0~100
    ret20 = pred_r[:, -1]
    ret_clip = np.clip(ret20, -0.15, 0.30)
    if ret_clip.max() > ret_clip.min():
        ret_norm = (ret_clip - ret_clip.min()) / (ret_clip.max() - ret_clip.min()) * 100
    else:
        ret_norm = np.full_like(ret_clip, 50.0)
    deep_raw = 0.4 * ret_norm + 0.6 * up_p * 100

    deep_map = dict(zip(ok_sids, deep_raw))
    pred20_map = dict(zip(ok_sids, ret20))
    upprob_map = dict(zip(ok_sids, up_p))

    # 計算前的基準分數（優先用混合分數，其次綜合分數）
    enhanced = ranking_df.copy()
    if '混合分數' in enhanced.columns:
        base_col = '混合分數'
    elif '綜合分數' in enhanced.columns:
        base_col = '綜合分數'
    else:
        return ranking_df, False

    base_max = float(enhanced[base_col].max()) or 1.0
    sid_key = enhanced['證券代號'].astype(str).str.strip()
    enhanced['Deep 分數'] = sid_key.map(lambda s: round(deep_map.get(s, 0.0), 1))
    enhanced['Deep Pred20d'] = sid_key.map(
        lambda s: f"{pred20_map.get(s, 0.0) * 100:+.1f}%" if s in pred20_map else '—')
    enhanced['Deep UpProb'] = sid_key.map(
        lambda s: f"{upprob_map.get(s, 0.0) * 100:.0f}%" if s in upprob_map else '—')

    def _fuse(row):
        sid = str(row['證券代號']).strip()
        base_norm = float(row[base_col]) / base_max * 100
        if sid in deep_map:
            return base_norm * (1 - deep_weight) + deep_map[sid] * deep_weight
        return base_norm

    enhanced['三層融合'] = enhanced.apply(_fuse, axis=1).round(1)
    enhanced = enhanced.sort_values('三層融合', ascending=False).reset_index(drop=True)
    enhanced = enhanced.head(top_n).copy()
    enhanced['排名'] = range(1, len(enhanced) + 1)

    # 欄位順序
    cols = enhanced.columns.tolist()
    desired = ['排名', '證券代號', '證券名稱', '市場',
               '三層融合', '混合分數', 'ML 分數', 'Deep 分數',
               'Deep Pred20d', 'Deep UpProb',
               '規則分數', '命中策略數', '命中策略',
               'MA 狀態', 'RSI', '量比', '相對強度', '最新收盤']
    ordered = [c for c in desired if c in cols] + \
              [c for c in cols if c not in desired and c != '綜合分數']
    enhanced = enhanced[ordered]

    matched = sum(1 for s in sid_key if s in deep_map)
    print(f"  Deep 融合完成：{matched}/{len(enhanced)} 檔有 Deep 分數 "
          f"(Deep weight={deep_weight:.2f})")
    return enhanced, True


# ===================== 低估股篩選 =====================

def find_undervalued(all_df, min_score=3):
    """篩選股價可能被低估、且法人開始進場的標的"""
    grouped = all_df.groupby('證券代號')
    results = []

    for stock_id, group in grouped:
        t = _technicals_cache.get(stock_id, _empty_technicals())
        if t['close'] is None or t['rsi'] is None:
            continue

        recent = group.head(2)
        if len(recent) < 2:
            continue

        signals = []
        score = 0

        # RSI 偏低 (< 40)
        if t['rsi'] < 40:
            signals.append(f'RSI 偏低 ({t["rsi"]:.1f})')
            score += 1

        # 股價在布林中軌以下
        if t['bb_upper'] is not None and t['bb_lower'] is not None:
            mid = (t['bb_upper'] + t['bb_lower']) / 2
            if t['close'] < mid:
                signals.append('低於布林中軌')
                score += 1

        # 法人最近 2 天有買超
        if (recent['三大法人買賣超'] > 0).any():
            signals.append('法人進場')
            score += 1

        # MA 短線趨勢向上
        if (t['ma_short'] is not None and len(t.get('rsi_history', [])) >= 2
                and t['ma_status'] in ('金叉', '接近金叉')):
            signals.append('短線趨勢向上')
            score += 1

        # 股價距近 20 日高點跌幅 > 10%
        if t['prev_high_20d'] is not None and t['prev_high_20d'] > 0:
            drawdown = (t['prev_high_20d'] - t['close']) / t['prev_high_20d']
            if drawdown > 0.10:
                signals.append(f'跌深 {drawdown*100:.1f}%')
                score += 1

        if score >= min_score:
            target_price, upside = _estimate_target_price(t)
            results.append({
                '證券代號': stock_id,
                '證券名稱': group['證券名稱'].iloc[0],
                '市場': group['市場'].iloc[0],
                '低估分數': score,
                '低估訊號': '; '.join(signals),
                'RSI': round(t['rsi'], 1),
                '布林位置': _bb_position(t),
                '最新收盤': t['close'],
                '預測目標價': target_price,
                '預估漲幅': upside,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('低估分數', ascending=False, inplace=True)
    return df


# ===================== 賣出警報（強化版） =====================

def check_sell_signals(all_df, holdings_df):
    """第一到第三階段整合版：
    - ATR 動態停損 / 保本 / 分層止盈 / Chandelier
    - 策略專屬出場規則
    - 相對強度轉弱 / 時間停損動態化 / 量價頂背離
    - 大盤空頭強制保護 / 加碼訊號 / 健檢分數
    - 自動寫入 trade_history.csv
    """
    alerts = []
    trade_logs = []
    now = datetime.now()
    bullish = is_market_bullish()

    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame()

    for _, row in holdings_df.iterrows():
        stock_id = str(row.get('證券代號', '')).strip()
        if not stock_id:
            continue
        if all_df is not None and not all_df.empty and '證券代號' in all_df.columns:
            group = all_df[all_df['證券代號'] == stock_id].head(5)
        else:
            group = pd.DataFrame()
        t = get_stock_technicals(stock_id)
        close = _as_num(t.get('close'))
        if close is None:
            continue

        buy_price = _as_num(row.get('買進價'))
        shares = _as_num(row.get('張數')) or 1
        max_high = _as_num(row.get('入場後最高')) or close
        initial_stop = _as_num(row.get('初始停損'))
        initial_atr = _as_num(row.get('初始ATR')) or _as_num(t.get('atr'))
        tier_status = str(row.get('分層狀態') or '未達')
        prev_tier = tier_status
        strategy = str(row.get('策略') or '')
        target_price = _as_num(row.get('目標價'))

        # 缺欄位補算
        if initial_stop is None and buy_price and initial_atr:
            initial_stop = _calc_initial_stop(buy_price, initial_atr, strategy)

        sell_reasons = []
        action = '持有'
        action_priority = 0  # 0=hold, 1=scale_down, 2=sell_half, 3=full_exit
        tier = _calc_tier_levels(buy_price, initial_stop) if (buy_price and initial_stop) else None
        tier_new_hit = None
        profit_pct = None

        if buy_price:
            profit_pct = (close - buy_price) / buy_price

        # ── 1) 動態停損（含 Break-even、分層、Chandelier）──
        current_stop = None
        if buy_price and initial_stop:
            current_stop = _calc_current_stop(
                buy_price, initial_stop, max_high, initial_atr,
                tier_status, bullish=bullish
            )

        # ── 2) 分層止盈偵測（新達到的層級）──
        if tier and max_high is not None:
            if max_high >= tier['3R'] and tier_status != '3R已達':
                tier_new_hit = '3R'
            elif max_high >= tier['2R'] and tier_status not in ('2R已達', '3R已達'):
                tier_new_hit = '2R'
            elif max_high >= tier['1R'] and tier_status == '未達':
                tier_new_hit = '1R'
            if tier_new_hit:
                sell_reasons.append(f'🎯 達 {tier_new_hit} 目標（分批出 1/3）')
                if tier_new_hit == '3R':
                    action, action_priority = '全部出場（末端）', max(action_priority, 3)
                else:
                    action, action_priority = '分批出 1/3', max(action_priority, 1)

        # ── 3) 硬停損：跌破當前停損 ──
        if current_stop is not None and close < current_stop:
            if tier_status in ('1R已達', '2R已達', '3R已達'):
                reason = f'🛑 跌破分層停損 {current_stop:.2f}（落袋為安）'
            elif max_high and max_high > (buy_price or 0):
                reason = f'🛑 跌破 Chandelier 停損 {current_stop:.2f}'
            else:
                reason = f'🛑 跌破初始停損 {current_stop:.2f}'
            sell_reasons.append(reason)
            action, action_priority = '全部出場', 3

        # ── 4) 策略專屬出場 ──
        spec_exits = _strategy_specific_exits(strategy, t, group)
        if spec_exits:
            sell_reasons.extend([f'[{strategy}] {sig}' for sig in spec_exits])
            if action_priority < 1:
                action, action_priority = '分批減碼', 1

        # ── 5) 相對強度轉弱 ──
        rs = t.get('rs_vs_taiex')
        if rs is not None and rs < 1.0 and profit_pct is not None and profit_pct < 0:
            sell_reasons.append(f'相對強度轉弱 RS={rs:.2f} 且虧損')
            if action_priority < 1:
                action, action_priority = '分批減碼', 1

        # ── 6) 時間停損動態化 ──
        buy_date_str = row.get('買進日期')
        held_days = None
        if pd.notna(buy_date_str) and buy_date_str:
            try:
                buy_date = pd.to_datetime(buy_date_str)
                held_days = (now - buy_date).days
                if held_days >= 30 and profit_pct is not None and profit_pct < 0.05:
                    sell_reasons.append(f'⏱ 持有 {held_days} 天未突破 +5%')
                    action, action_priority = '全部出場', max(action_priority, 3)
                elif held_days >= 20 and profit_pct is not None and profit_pct < 0.03:
                    sell_reasons.append(f'⏱ 持有 {held_days} 天未突破 +3%')
                    if action_priority < 2:
                        action, action_priority = '分批出一半', 2
                elif held_days >= 10 and profit_pct is not None and profit_pct < 0:
                    sell_reasons.append(f'⏱ 持有 {held_days} 天仍虧損，建議縮停損')
            except Exception:
                pass

        # ── 7) 量價頂背離 ──
        if (tier_status in ('2R已達', '3R已達')
                and t.get('volume_ratio') is not None and t['volume_ratio'] < 0.8
                and t.get('macd_hist') is not None and t.get('macd_hist_prev') is not None
                and t['macd_hist'] < t['macd_hist_prev'] * 0.7):
            sell_reasons.append('📉 高檔量價頂背離（量縮 + MACD 柱縮）')
            if action_priority < 1:
                action, action_priority = '分批減碼', 1

        # ── 8) 加碼訊號（僅趨勢策略、保本以上、多頭）──
        add_signal = ''
        ids = _extract_strategy_ids(strategy)
        if (bullish and buy_price and profit_pct is not None
                and profit_pct >= 0.05
                and tier and max_high and max_high >= tier['1R']
                and any(i in ids for i in ['4', '11', '13'])
                and t.get('ma_short') and t.get('ma_long')
                and t['ma_short'] > t['ma_long']
                and close > t['ma_short']
                and action_priority == 0):
            add_signal = f'➕ 可加碼 1/3（停損維持 {current_stop:.2f}）'

        # ── 9) 健檢分數 ──
        row_with_stop = dict(row)
        row_with_stop['當前停損'] = current_stop
        health = _calc_health_score(row_with_stop, t, group)
        if health < 40 and action_priority < 1:
            sell_reasons.append(f'⚠ 健檢分數低 ({health}/100)')
            action, action_priority = '重檢風險', 1

        # ── 10) 目標價接近 ──
        if target_price and close >= target_price * 0.95:
            sell_reasons.append(f'🎯 距目標價 {target_price:.2f} 僅 {(target_price-close)/close*100:.1f}%')
            if action_priority < 2:
                action, action_priority = '分批獲利了結', 2

        # ── 11) 大盤空頭提示 ──
        if not bullish:
            sell_reasons.append('⚠ 大盤空頭：停損緊縮至 2×ATR')

        # 組裝輸出（即使只有觀察也輸出完整狀態供 HTML 顯示）
        record = {
            '證券代號': stock_id,
            '證券名稱': row.get('證券名稱', ''),
            '市場': group['市場'].iloc[0] if not group.empty and '市場' in group.columns else '-',
            '現價': close,
            '買進價': buy_price,
            '損益%': f'{profit_pct*100:.1f}%' if profit_pct is not None else '-',
            '分層狀態': f'{prev_tier}→{tier_new_hit}已達' if tier_new_hit else prev_tier,
            '當前停損': current_stop if current_stop else '-',
            'R值': round(tier['R'], 2) if tier else '-',
            '健檢分數': health,
            '賣出訊號': '; '.join(sell_reasons) if sell_reasons else '（持有觀察）',
            '建議動作': action,
            '加碼訊號': add_signal or '-',
            '持有天數': held_days if held_days is not None else '-',
            'MA狀態': t.get('ma_status') or '-',
            'RSI': round(t['rsi'], 1) if t.get('rsi') else '-',
        }
        alerts.append(record)

        # 寫入歷史（實際的出場行動）
        if action_priority >= 2:
            trade_logs.append({
                '日期': now.strftime('%Y-%m-%d'),
                '證券代號': stock_id,
                '證券名稱': row.get('證券名稱', ''),
                '買進日期': str(buy_date_str) if pd.notna(buy_date_str) else '',
                '買進價': buy_price,
                '當時收盤': close,
                '損益%': round(profit_pct * 100, 2) if profit_pct is not None else None,
                '持有天數': held_days if held_days is not None else '',
                '出場原因': '; '.join(sell_reasons),
                '建議動作': action,
                '買進策略': strategy,
                '健檢分數': health,
            })

    if trade_logs:
        _log_trade_history(trade_logs)

    return pd.DataFrame(alerts)


# ===================== 持股管理 =====================

HOLDING_COLUMNS = ['證券代號', '證券名稱', '買進日期', '買進價', '目標價',
                   '停損價', '張數', '策略', '備註',
                   '入場後最高', '初始ATR', '初始停損', '分層狀態']


def load_holdings():
    """讀取持股，自動補齊新欄位（向下相容舊 CSV）。"""
    if os.path.exists(HOLDINGS_FILE):
        df = pd.read_csv(HOLDINGS_FILE, dtype={'證券代號': str})
        df['證券代號'] = df['證券代號'].astype(str).str.strip()
        for c in HOLDING_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[HOLDING_COLUMNS]
    return pd.DataFrame(columns=HOLDING_COLUMNS)


def save_holdings(df):
    for c in HOLDING_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df[HOLDING_COLUMNS].to_csv(HOLDINGS_FILE, index=False, encoding='utf-8-sig')


# ===================== 持股風險管理核心 =====================

def _as_num(v):
    """容錯數字轉型，支援 None / NaN / 空字串 / 字串數字。"""
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        f = float(v)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _get_account_size():
    try:
        v = os.environ.get('HOLDINGS_ACCOUNT_SIZE')
        return float(v) if v else ACCOUNT_SIZE_DEFAULT
    except Exception:
        return ACCOUNT_SIZE_DEFAULT


def _extract_strategy_ids(strategy_str):
    """從策略欄解析出所有 策略N 編號（支援多策略以逗號/空白分隔）。"""
    if not strategy_str:
        return []
    s = str(strategy_str)
    return re.findall(r'策略(\d+)', s)


def _calc_initial_stop(buy_price, atr, strategy=None):
    """初始停損 = 買進價 - N×ATR；趨勢類（策略 4/11/13）給更大空間。"""
    if not buy_price:
        return None
    if not atr or atr <= 0:
        return buy_price * 0.95
    ids = _extract_strategy_ids(strategy)
    mult = ATR_INITIAL_STOP_MULT
    if any(i in ids for i in ['4', '11', '13']):
        mult = 2.0
    return round(buy_price - mult * atr, 2)


def _calc_tier_levels(buy_price, initial_stop):
    """計算 1R / 2R / 3R 價位。R 為單位風險。"""
    if not buy_price or not initial_stop or buy_price <= initial_stop:
        return None
    r = buy_price - initial_stop
    return {
        'R': r,
        '1R': buy_price + r,
        '2R': buy_price + 2 * r,
        '3R': buy_price + 3 * r,
    }


def _calc_chandelier_stop(max_high, atr, mult=CHANDELIER_ATR_MULT):
    """吊燈式移動停損：入場後最高 - N×ATR。"""
    if max_high is None or atr is None or atr <= 0:
        return None
    return max_high - mult * atr


def _tier_status_from_max(max_high, tier):
    """依入場後最高價決定已達到的分層狀態。"""
    if max_high is None or tier is None:
        return '未達'
    if max_high >= tier['3R']:
        return '3R已達'
    if max_high >= tier['2R']:
        return '2R已達'
    if max_high >= tier['1R']:
        return '1R已達'
    return '未達'


def _calc_current_stop(buy_price, initial_stop, max_high, atr, tier_status,
                       bullish=True):
    """動態停損 = 所有適用停損規則取最大值（只上升不下降）。"""
    if buy_price is None or initial_stop is None:
        return None
    stops = [initial_stop]
    # 保本停損：max_high >= buy_price × (1+5%) 啟動
    if max_high is not None and max_high >= buy_price * (1 + BREAKEVEN_PROFIT_TRIGGER):
        stops.append(buy_price)
    # 分層停損
    tier = _calc_tier_levels(buy_price, initial_stop)
    if tier:
        if tier_status == '1R已達':
            stops.append(buy_price)
        elif tier_status == '2R已達':
            stops.append(tier['1R'])
        elif tier_status == '3R已達':
            stops.append(tier['2R'])
    # 吊燈停損（僅在已獲利時啟動）
    if max_high is not None and max_high > buy_price and atr:
        mult = CHANDELIER_ATR_MULT_BEAR if not bullish else CHANDELIER_ATR_MULT
        ch = _calc_chandelier_stop(max_high, atr, mult=mult)
        if ch is not None:
            stops.append(ch)
    return round(max(stops), 2)


def _strategy_specific_exits(strategy, t, group):
    """買進策略對應的專屬出場訊號。"""
    signals = []
    ids = _extract_strategy_ids(strategy)
    recent2 = group.head(2) if group is not None and not group.empty else None

    # 策略 1-3：法人類（各自對應）
    if recent2 is not None and len(recent2) >= 2:
        if '1' in ids and (recent2['外資買賣超'] < 0).all():
            net = recent2['外資買賣超'].sum()
            if net < -2000:
                signals.append(f'外資連賣 2 日淨賣 {int(net)} 張')
        if '2' in ids and (recent2['投信買賣超'] < 0).all():
            net = recent2['投信買賣超'].sum()
            if net < -1000:
                signals.append(f'投信連賣 2 日淨賣 {int(net)} 張')
        if '3' in ids and (recent2['三大法人買賣超'] < 0).all():
            signals.append('三大法人連 2 日賣超')

    # 策略 4：量價突破 → 跌回 MA5 或量能萎縮
    if '4' in ids:
        if t.get('ma_short') and t.get('close') and t['close'] < t['ma_short']:
            signals.append('跌回 MA5 下')
        if t.get('volume_ratio') is not None and t['volume_ratio'] < 0.7:
            signals.append(f'量能萎縮 {t["volume_ratio"]:.1f}×')

    # 策略 5：RSI 反彈 → RSI>75 分批 / <40 停損
    if '5' in ids:
        rsi = t.get('rsi')
        if rsi is not None:
            if rsi >= 75:
                signals.append(f'RSI 超買 {rsi:.1f}')
            elif rsi < 40:
                signals.append(f'RSI 再跌破 40 ({rsi:.1f})')

    # 策略 6：布林突破 → 跌破中軌
    if '6' in ids:
        bu, bl, cv = t.get('bb_upper'), t.get('bb_lower'), t.get('close')
        if bu and bl and cv:
            mid = (bu + bl) / 2
            if cv < mid:
                signals.append('跌破布林中軌')

    # 策略 9：MACD 金叉 → MACD 死叉
    if '9' in ids:
        if (t.get('macd') is not None and t.get('macd_signal') is not None
                and t['macd'] < t['macd_signal']):
            signals.append('MACD 死叉')

    # 策略 10：KD 黃金叉 → KD 高檔死叉
    if '10' in ids:
        k, d = t.get('kd_k'), t.get('kd_d')
        if k is not None and d is not None and k < d and k > 80:
            signals.append(f'KD 高檔死叉 K={k:.1f}')

    # 策略 11：Darvas 突破 → 跌破 MA_long（作為底線）
    if '11' in ids:
        if t.get('close') and t.get('ma_long') and t['close'] < t['ma_long']:
            signals.append('跌破 MA_long (Darvas 結構破)')

    # 策略 13：相對強度 → RS < 1.0
    if '13' in ids:
        rs = t.get('rs_vs_taiex')
        if rs is not None and rs < 1.0:
            signals.append(f'相對強度轉弱 RS={rs:.2f}')

    return signals


def _calc_health_score(h, t, group):
    """持股健檢分數 0-100。"""
    score = 50
    close = _as_num(t.get('close'))
    buy_price = _as_num(h.get('買進價'))

    # 損益（+20 最高）
    if close and buy_price:
        p = (close - buy_price) / buy_price
        score += int(max(min(p * 100 * 2, 20), -20))

    # MA 狀態
    ma_status = t.get('ma_status')
    if ma_status == '金叉':
        score += 8
    elif ma_status == '死叉':
        score -= 12
    elif ma_status == '接近金叉':
        score += 3

    # 相對強度
    rs = t.get('rs_vs_taiex')
    if rs is not None:
        if rs >= 1.10:
            score += 8
        elif rs >= 1.05:
            score += 4
        elif rs < 0.95:
            score -= 8

    # MACD hist 動能
    if t.get('macd_hist') is not None and t.get('macd_hist_prev') is not None:
        if t['macd_hist'] > t['macd_hist_prev']:
            score += 4
        else:
            score -= 3

    # RSI 區間
    rsi = t.get('rsi')
    if rsi is not None:
        if 45 <= rsi <= 70:
            score += 4
        elif rsi > 80:
            score -= 6
        elif rsi < 30:
            score -= 4

    # 法人近 3 日
    if group is not None and not group.empty and '三大法人買賣超' in group.columns:
        net3 = group.head(3)['三大法人買賣超'].sum()
        if net3 > 500:
            score += 6
        elif net3 < -1000:
            score -= 10

    # 距停損
    cstop = _as_num(h.get('當前停損'))
    if cstop and close:
        room = (close - cstop) / close
        if room < 0.02:
            score -= 8  # 已到停損邊緣

    return max(0, min(100, score))


def _suggest_position_size(buy_price, atr, account_size=None, strategy=None):
    """按 ATR 反向縮放建議張數。單筆風險 = POSITION_RISK_PCT × 帳戶資金。"""
    acc = account_size or _get_account_size()
    if not buy_price or not atr or atr <= 0:
        return 1
    ids = _extract_strategy_ids(strategy)
    mult = 2.0 if any(i in ids for i in ['4', '11', '13']) else ATR_INITIAL_STOP_MULT
    risk_per_share = mult * atr
    max_risk_total = acc * POSITION_RISK_PCT
    shares = max_risk_total / (risk_per_share * 1000)
    return max(1, int(round(shares)))


def _update_holdings_state(holdings_df):
    """為每筆持股更新：入場後最高、初始ATR、初始停損、分層狀態。
    就地修改並回傳。"""
    if holdings_df is None or holdings_df.empty:
        return holdings_df
    # 確保欄位齊全
    for c in HOLDING_COLUMNS:
        if c not in holdings_df.columns:
            holdings_df[c] = None

    for idx, row in holdings_df.iterrows():
        sid = str(row.get('證券代號', '')).strip()
        if not sid:
            continue
        t = _technicals_cache.get(sid, _empty_technicals())
        close = _as_num(t.get('close'))
        atr = _as_num(t.get('atr'))
        buy_price = _as_num(row.get('買進價'))

        # 1) 入場後最高
        curr_max = _as_num(row.get('入場後最高'))
        candidates = [v for v in [curr_max, close, buy_price] if v is not None]
        if candidates:
            holdings_df.at[idx, '入場後最高'] = round(max(candidates), 2)

        # 2) 初始 ATR（僅首次寫入）
        if _as_num(row.get('初始ATR')) is None and atr:
            holdings_df.at[idx, '初始ATR'] = round(atr, 3)

        # 3) 初始停損（僅首次寫入）
        init_stop = _as_num(row.get('初始停損'))
        if init_stop is None and buy_price:
            atr_use = _as_num(holdings_df.at[idx, '初始ATR']) or atr
            new_stop = _calc_initial_stop(buy_price, atr_use, row.get('策略'))
            if new_stop:
                holdings_df.at[idx, '初始停損'] = new_stop

        # 4) 分層狀態（根據當前最高價更新，只升不降）
        init_stop = _as_num(holdings_df.at[idx, '初始停損'])
        new_max = _as_num(holdings_df.at[idx, '入場後最高'])
        if buy_price and init_stop and new_max:
            tier = _calc_tier_levels(buy_price, init_stop)
            if tier:
                new_status = _tier_status_from_max(new_max, tier)
                # 狀態只能升不能降
                order = {'未達': 0, '1R已達': 1, '2R已達': 2, '3R已達': 3}
                prev = str(row.get('分層狀態') or '未達')
                if order.get(new_status, 0) >= order.get(prev, 0):
                    holdings_df.at[idx, '分層狀態'] = new_status
                else:
                    holdings_df.at[idx, '分層狀態'] = prev
    return holdings_df


def _log_trade_history(rows):
    """追加出場訊號到 trade_history.csv。"""
    if not rows:
        return
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(TRADE_HISTORY_FILE)
    df.to_csv(TRADE_HISTORY_FILE, mode='a', header=not file_exists,
              index=False, encoding='utf-8-sig')


def check_portfolio_risk(holdings_df):
    """投組風險總覽：總成本、總市值、總風險、集中度警示。"""
    result = {
        'total_cost': 0, 'total_market': 0, 'total_risk': 0,
        'mtm_pct': 0, 'warnings': [], 'positions': 0,
        'max_concentration': 0, 'bullish': is_market_bullish(),
    }
    if holdings_df is None or holdings_df.empty:
        return result
    total_cost = 0
    total_market = 0
    total_risk = 0
    cost_by_sid = {}
    cost_by_strategy = {}
    tracked = 0

    for _, h in holdings_df.iterrows():
        sid = str(h.get('證券代號', '')).strip()
        if not sid:
            continue
        t = _technicals_cache.get(sid, _empty_technicals())
        close = _as_num(t.get('close'))
        buy_price = _as_num(h.get('買進價'))
        shares = _as_num(h.get('張數')) or 1
        strat = str(h.get('策略') or '未分類')
        if not buy_price:
            continue
        cost = buy_price * shares * 1000
        total_cost += cost
        cost_by_sid[sid] = cost_by_sid.get(sid, 0) + cost
        cost_by_strategy[strat] = cost_by_strategy.get(strat, 0) + cost
        if close:
            total_market += close * shares * 1000
            tracked += 1
        init_stop = _as_num(h.get('初始停損'))
        if init_stop and buy_price > init_stop:
            total_risk += (buy_price - init_stop) * shares * 1000

    result['total_cost'] = total_cost
    result['total_market'] = total_market
    result['total_risk'] = total_risk
    result['positions'] = len(holdings_df)

    if total_cost > 0:
        if total_market > 0:
            result['mtm_pct'] = (total_market - total_cost) / total_cost
        # 單股集中度
        max_cost = max(cost_by_sid.values()) if cost_by_sid else 0
        conc = max_cost / total_cost
        result['max_concentration'] = conc
        if conc > 0.4 and len(cost_by_sid) > 1:
            top_sid = max(cost_by_sid, key=cost_by_sid.get)
            result['warnings'].append(
                f'單股集中度過高：{top_sid} 占 {conc * 100:.1f}%'
            )
        # 總虧損
        if tracked > 0 and result['mtm_pct'] < -0.10:
            result['warnings'].append(
                f'投組未實現虧損 {result["mtm_pct"] * 100:.1f}% 超過 10%，建議減倉'
            )
        # 策略集中
        if len(cost_by_strategy) > 1:
            for strat, c in cost_by_strategy.items():
                if c / total_cost > 0.5:
                    result['warnings'].append(
                        f'策略「{strat}」集中度 {c / total_cost * 100:.1f}%'
                    )
        # 風險占資金比例
        acc = _get_account_size()
        if total_risk > acc * 0.05:
            result['warnings'].append(
                f'投組總風險 {total_risk:,.0f} 超過帳戶 5%（{acc * 0.05:,.0f}），建議降倉'
            )

    # 大盤空頭
    if not result['bullish']:
        result['warnings'].append('⚠ 大盤空頭：所有持股停損自動縮緊至 2×ATR，禁止加碼')
    return result


# ===================== 互動式 HTML 報告 =====================

def _df_to_html_table(df):
    """將 DataFrame 轉為 styled HTML table 字串。"""
    if df is None or df.empty:
        return '<p class="empty">無符合</p>'
    html = df.to_html(index=False, classes='data-table', border=0,
                      float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x)
    return html


def _build_stock_charts(stock_ids, all_data, revenue_data, ohlcv_cache):
    """為 TOP 30 個股產生圖表 HTML（Plotly）。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    charts_html = ''
    for sid in stock_ids:
        sid = str(sid).strip()
        ohlcv = ohlcv_cache.get(sid)
        if ohlcv is None or ohlcv.empty:
            continue

        df = ohlcv.copy()
        c = df['Close']
        ma5 = c.rolling(5).mean()
        ma10 = c.rolling(10).mean()
        bb_ma = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_up = bb_ma + 2 * bb_std
        bb_lo = bb_ma - 2 * bb_std

        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_g = gain.rolling(14).mean()
        avg_l = loss.rolling(14).mean()
        rs = avg_g / avg_l
        rsi = 100 - (100 / (1 + rs))

        vol_avg = df['Volume'].rolling(20).mean()

        name_str = sid
        if all_data is not None and not all_data.empty:
            match = all_data[all_data['證券代號'] == sid]
            if not match.empty:
                name_str = f"{sid} {match['證券名稱'].iloc[0]}"

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.45, 0.2, 0.15, 0.2],
            subplot_titles=['', '', '', '']
        )

        # K 線 + MA + BB
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=c,
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
            name='K線', showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma5, name='MA5',
                                 line=dict(width=1, color='#ffa726')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma10, name='MA10',
                                 line=dict(width=1, color='#42a5f5')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_up, name='BB上',
                                 line=dict(width=1, dash='dot', color='#78909c')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_lo, name='BB下',
                                 line=dict(width=1, dash='dot', color='#78909c'),
                                 fill='tonexty', fillcolor='rgba(120,144,156,0.08)'), row=1, col=1)

        # 成交量
        colors = ['#ef5350' if c.iloc[i] >= (c.iloc[i-1] if i > 0 else c.iloc[i])
                  else '#26a69a' for i in range(len(c))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量',
                             marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=vol_avg, name='20日均量',
                                 line=dict(width=1, color='#ffa726')), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI',
                                 line=dict(width=1.5, color='#ab47bc')), row=3, col=1)
        fig.add_hline(y=80, line_dash='dash', line_color='#ef5350',
                      opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='#26a69a',
                      opacity=0.5, row=3, col=1)

        # 法人買賣超
        if all_data is not None and not all_data.empty:
            inst = all_data[all_data['證券代號'] == sid].head(15).sort_values('日期')
            if not inst.empty and '日期' in inst.columns:
                dates = inst['日期']
                for col_name, color in [('外資買賣超', '#42a5f5'),
                                         ('投信買賣超', '#ffa726'),
                                         ('自營商買賣超', '#66bb6a')]:
                    if col_name in inst.columns:
                        fig.add_trace(go.Bar(
                            x=dates, y=inst[col_name], name=col_name,
                            marker_color=color, opacity=0.8
                        ), row=4, col=1)

        fig.update_layout(
            title=dict(text=name_str, font=dict(size=16)),
            template='plotly_dark',
            paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
            height=700, margin=dict(l=50, r=30, t=40, b=30),
            legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center',
                        font=dict(size=9)),
            xaxis_rangeslider_visible=False,
            barmode='group',
        )
        fig.update_yaxes(title_text='價格', row=1, col=1)
        fig.update_yaxes(title_text='量', row=2, col=1)
        fig.update_yaxes(title_text='RSI', row=3, col=1)
        fig.update_yaxes(title_text='張數', row=4, col=1)

        chart_div = fig.to_html(full_html=False, include_plotlyjs=False)
        charts_html += f'''
        <div class="stock-card" id="stock-{sid}">
            <button class="accordion" onclick="toggleAccordion(this)">
                {name_str}
            </button>
            <div class="accordion-content">{chart_div}</div>
        </div>'''

    return charts_html


def _build_revenue_chart(revenue_data, stock_ids):
    """營收增率圖表（僅針對 TOP 30 中有營收資料者）。"""
    import plotly.graph_objects as go
    if revenue_data is None or revenue_data.empty:
        return ''

    sids = [str(s).strip() for s in stock_ids]
    rev = revenue_data[revenue_data['證券代號'].isin(sids)].copy()
    if rev.empty:
        return ''

    yoy = rev.dropna(subset=['營收年增率']).sort_values('營收年增率', ascending=True)
    if yoy.empty:
        return ''

    fig = go.Figure()
    colors = ['#ef5350' if v < 0 else '#26a69a' for v in yoy['營收年增率']]
    fig.add_trace(go.Bar(
        y=yoy['證券代號'], x=(yoy['營收年增率'] * 100).round(1),
        orientation='h', marker_color=colors,
        text=(yoy['營收年增率'] * 100).round(1).astype(str) + '%',
        textposition='outside',
    ))
    fig.update_layout(
        title='營收年增率 (%)',
        template='plotly_dark',
        paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
        height=max(300, len(yoy) * 28),
        margin=dict(l=80, r=60, t=40, b=30),
        xaxis_title='年增率 (%)',
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_backtest_charts(bt_result):
    """產生回測績效區的三張圖表。"""
    import plotly.graph_objects as go

    html_parts = []

    if bt_result is None:
        return '<p class="empty">無回測資料</p>'

    summary = bt_result.summary
    if summary.empty:
        return '<p class="empty">無回測資料</p>'

    # 1) 策略績效比較柱狀圖
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name='勝率 (%)', x=summary['策略'], y=summary['勝率'],
        marker_color='#42a5f5'
    ))
    fig1.add_trace(go.Bar(
        name='平均報酬 (%)', x=summary['策略'], y=summary['平均報酬'],
        marker_color='#66bb6a'
    ))
    fig1.add_trace(go.Bar(
        name='Sharpe', x=summary['策略'], y=summary['Sharpe'],
        marker_color='#ffa726'
    ))
    fig1.update_layout(
        title='策略績效比較',
        barmode='group', template='plotly_dark',
        paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
        height=400, margin=dict(l=50, r=30, t=50, b=80),
        legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'),
    )
    html_parts.append(fig1.to_html(full_html=False, include_plotlyjs=False))

    # 2) 累積報酬曲線
    curves = bt_result.equity_curves
    if curves:
        fig2 = go.Figure()
        colors = ['#42a5f5', '#ef5350', '#66bb6a', '#ffa726', '#ab47bc', '#78909c']
        for i, (name, curve) in enumerate(curves.items()):
            if curve is not None and not curve.empty:
                fig2.add_trace(go.Scatter(
                    x=curve.index, y=(curve - 1) * 100,
                    name=name, mode='lines',
                    line=dict(width=1.5, color=colors[i % len(colors)])
                ))
        fig2.update_layout(
            title='累積報酬曲線 (%)',
            template='plotly_dark',
            paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
            height=400, margin=dict(l=50, r=30, t=50, b=30),
            yaxis_title='累積報酬 (%)',
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center',
                        font=dict(size=9)),
        )
        html_parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    # 3) 月度報酬熱力圖
    if not bt_result.trades.empty:
        t = bt_result.trades.copy()
        t['month'] = pd.to_datetime(t['exit_date']).dt.strftime('%Y-%m')
        pivot = t.pivot_table(values='return', index='strategy',
                              columns='month', aggfunc='mean') * 100
        if not pivot.empty:
            fig3 = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale='RdYlGn', zmid=0,
                text=pivot.round(1).values,
                texttemplate='%{text}%',
                hovertemplate='%{y}<br>%{x}<br>報酬: %{z:.1f}%<extra></extra>',
            ))
            fig3.update_layout(
                title='月度平均報酬 (%)',
                template='plotly_dark',
                paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
                height=350, margin=dict(l=200, r=30, t=50, b=40),
            )
            html_parts.append(fig3.to_html(full_html=False, include_plotlyjs=False))

    return '\n'.join(html_parts)


def _retry_missing_holdings_data(holdings_df):
    """方案 B：對持股中 _technicals_cache.close is None 的代號做保底單檔重抓。
    yfinance 批次下載偶有漏抓（rate limit / ticker 對應失敗），持股數量小，
    逐檔重試成本低，但能大幅降低『現價遺失』的機率。"""
    if holdings_df is None or holdings_df.empty or '證券代號' not in holdings_df.columns:
        return
    miss = []
    for sid in holdings_df['證券代號'].astype(str).str.strip().unique():
        if not sid:
            continue
        t = _technicals_cache.get(sid)
        if t is None or t.get('close') is None:
            miss.append(sid)
    if not miss:
        return
    print(f"  持股保底補抓：{len(miss)} 檔現價（{', '.join(miss)}）...")
    recovered = 0
    for sid in miss:
        for suffix in ('.TW', '.TWO'):
            try:
                data = yf.download(f"{sid}{suffix}", period='3mo',
                                   progress=False, auto_adjust=False)
                if data is None or data.empty:
                    continue
                r = _compute_technicals(data)
                if r.get('close') is not None:
                    r['rs_vs_taiex'] = _compute_rs_vs_taiex(r)
                    _technicals_cache[sid] = r
                    recovered += 1
                    break
            except Exception:
                continue
    print(f"  保底補抓結果：{recovered}/{len(miss)} 成功恢復現價")


def _build_holdings_section(sell_alerts, initial_holdings_df, all_data,
                            portfolio_risk=None):
    """產生『我的持股』互動式 section（純前端 localStorage）。
    第三階段強化：初始/當前停損、分層狀態、健檢分數、加碼訊號、投組風險卡。"""
    # 方案 B：持股保底補抓，避免 batch 下載漏掉導致現價遺失
    _retry_missing_holdings_data(initial_holdings_df)

    # 蒐集股票資料：代號 → {close, atr, name, rs, ma_status, sell*, health, ...}
    stock_data = {}
    acc_size = _get_account_size()
    for sid, t in _technicals_cache.items():
        close = _as_num(t.get('close'))
        if close is None:
            continue
        atr = _as_num(t.get('atr'))
        stock_data[sid] = {
            'close': round(close, 2),
            'atr': round(atr, 3) if atr else None,
            'rs': round(t['rs_vs_taiex'], 2) if t.get('rs_vs_taiex') is not None else None,
            'ma_status': t.get('ma_status') or None,
            'rsi': round(t['rsi'], 1) if t.get('rsi') is not None else None,
            'sugShares': _suggest_position_size(close, atr, acc_size, None),
        }

    # 附上股票名稱
    if all_data is not None and not all_data.empty and '證券名稱' in all_data.columns:
        name_map = (
            all_data[['證券代號', '證券名稱']]
            .drop_duplicates(subset=['證券代號'])
            .set_index('證券代號')['證券名稱']
            .to_dict()
        )
        for sid, name in name_map.items():
            sid_str = str(sid).strip()
            if sid_str in stock_data:
                stock_data[sid_str]['name'] = str(name)
            else:
                stock_data[sid_str] = {'name': str(name), 'close': None, 'atr': None}

    # 附上賣出訊號 + 新版持股管理欄位
    if sell_alerts is not None and not sell_alerts.empty:
        for _, row in sell_alerts.iterrows():
            sid = str(row.get('證券代號', '')).strip()
            if not sid:
                continue
            stock_data.setdefault(sid, {})
            d = stock_data[sid]
            d['sellSignal'] = str(row.get('賣出訊號', '') or '')
            d['sellAction'] = str(row.get('建議動作', '') or '')
            # 第三階段新欄位
            cs = row.get('當前停損')
            d['currentStop'] = float(cs) if isinstance(cs, (int, float)) and not pd.isna(cs) else None
            d['tierStatus'] = str(row.get('分層狀態', '') or '')
            d['health'] = int(row['健檢分數']) if pd.notna(row.get('健檢分數')) else None
            d['addSignal'] = str(row.get('加碼訊號', '') or '')
            rv = row.get('R值')
            d['Rvalue'] = float(rv) if isinstance(rv, (int, float)) and not pd.isna(rv) else None

    # 將現有 holdings.csv 作為預設持股種子（含第三階段欄位）
    initial_list = []
    if initial_holdings_df is not None and not initial_holdings_df.empty:
        for _, row in initial_holdings_df.iterrows():
            def _num(v):
                try:
                    if pd.isna(v):
                        return None
                    return float(v)
                except Exception:
                    return None
            initial_list.append({
                'stock_id': str(row.get('證券代號', '')).strip(),
                'name': '' if pd.isna(row.get('證券名稱')) else str(row.get('證券名稱', '')),
                'buy_date': '' if pd.isna(row.get('買進日期')) else str(row.get('買進日期', '')),
                'buy_price': _num(row.get('買進價')),
                'target_price': _num(row.get('目標價')),
                'stop_loss': _num(row.get('停損價')),
                'shares': _num(row.get('張數')) or 1,
                'strategy': '' if pd.isna(row.get('策略')) else str(row.get('策略', '')),
                'note': '' if pd.isna(row.get('備註')) else str(row.get('備註', '')),
                'max_high': _num(row.get('入場後最高')),
                'initial_atr': _num(row.get('初始ATR')),
                'initial_stop': _num(row.get('初始停損')),
                'tier_status': '' if pd.isna(row.get('分層狀態')) else str(row.get('分層狀態', '')),
            })

    portfolio_payload = portfolio_risk or {}
    portfolio_json_obj = {
        'total_cost': portfolio_payload.get('total_cost', 0),
        'total_market': portfolio_payload.get('total_market', 0),
        'total_risk': portfolio_payload.get('total_risk', 0),
        'mtm_pct': portfolio_payload.get('mtm_pct', 0),
        'max_concentration': portfolio_payload.get('max_concentration', 0),
        'warnings': portfolio_payload.get('warnings', []),
        'bullish': bool(portfolio_payload.get('bullish', True)),
        'account_size': acc_size,
    }

    stock_json = json.dumps(stock_data, ensure_ascii=False)
    initial_json = json.dumps(initial_list, ensure_ascii=False)
    portfolio_json = json.dumps(portfolio_json_obj, ensure_ascii=False)
    has_initial = 'true' if initial_list else 'false'

    strategy_options = ''.join(
        f'<option>{s}</option>' for s in [
            '策略1 外資連續買超', '策略2 投信連續買超', '策略3 三法人共識',
            '策略4 量價齊揚', '策略5 RSI超賣反彈', '策略6 布林收斂',
            '策略7 營收+法人', '策略8 融資/融券', '策略9 MACD金叉',
            '策略10 KD黃金叉', '策略11 Darvas突破', '策略12 價量背離',
            '策略13 相對強度', '自訂'
        ]
    )

    style = """
<style>
#holdings .holdings-toolbar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }
#holdings .holdings-toolbar button, #holdings .btn-mini {
    padding:8px 14px; border:none; border-radius:6px; cursor:pointer;
    font-size:13px; transition:all .2s;
}
#holdings .btn-primary { background:#42a5f5; color:#fff; }
#holdings .btn-primary:hover { background:#1976d2; }
#holdings .btn-secondary { background:#1a1a2e; color:#e0e0e0; border:1px solid #2a2a4a; }
#holdings .btn-secondary:hover { background:#1e3a5f; }
#holdings .btn-danger { background:#ef5350; color:#fff; }
#holdings .btn-mini { padding:3px 10px; font-size:11px; background:#ef5350; color:#fff; }

#holdings .holdings-summary { margin-bottom:16px; }
#holdings .summary-grid {
    display:grid; grid-template-columns:repeat(auto-fit, minmax(150px,1fr));
    gap:12px;
}
#holdings .summary-item {
    background:#1a1a2e; padding:12px 16px; border-radius:8px;
    border-left:3px solid #42a5f5;
}
#holdings .summary-item .label { font-size:12px; color:#a0a0b0; }
#holdings .summary-item .value {
    font-size:20px; font-weight:600; color:#fff; margin-top:4px;
}

#holdings .holdings-form {
    background:#1a1a2e; padding:16px 20px; border-radius:8px;
    margin-bottom:16px; border:1px solid #2a2a4a;
}
#holdings .form-grid {
    display:grid; grid-template-columns:repeat(auto-fit, minmax(170px,1fr)); gap:12px;
}
#holdings .form-grid label {
    display:block; font-size:12px; color:#a0a0b0; margin-bottom:4px;
}
#holdings .form-grid input, #holdings .form-grid select {
    width:100%; padding:6px 8px; background:#0f0f1a; color:#e0e0e0;
    border:1px solid #2a2a4a; border-radius:4px; font-size:13px;
}
#holdings .form-grid .full-row { grid-column:1 / -1; }
#holdings .form-actions { margin-top:14px; display:flex; gap:8px; }

#holdings .holdings-table { font-size:12.5px; }
#holdings .holdings-table th { padding:10px 8px; }
#holdings .holdings-table td { padding:7px 8px; }
#holdings .holdings-table td.gain { color:#26a69a; font-weight:600; }
#holdings .holdings-table td.loss { color:#ef5350; font-weight:600; }
#holdings .holdings-table td.near-target { color:#ffb74d; font-weight:600; }
#holdings .holdings-table td.near-stop { color:#ef5350; font-weight:600; }
#holdings .offline-price {
    color:#9e9e9e; font-style:italic;
    border-bottom:1px dashed #666; cursor:help;
}
#holdings .offline-badge {
    display:inline-block; margin-left:4px; padding:1px 5px;
    border-radius:3px; background:#4a4a5a; color:#fff;
    font-size:9px; font-weight:600; vertical-align:middle;
}
#holdings .holdings-table td.action-hold { color:#a0a0b0; }
#holdings .holdings-table td.action-alert {
    background:rgba(239,83,80,.18); color:#ef5350; font-weight:600;
}
#holdings .holdings-table td.action-target {
    background:rgba(255,183,77,.18); color:#ffb74d; font-weight:600;
}
#holdings .holdings-table td.action-stoploss {
    background:rgba(239,83,80,.35); color:#fff; font-weight:700;
}
#holdings .rr-good { color:#26a69a; font-weight:600; }
#holdings .rr-bad { color:#ef5350; }

/* 第三階段：風險卡 / 健檢分數 / 分層狀態 */
#holdings .risk-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 4px solid #42a5f5;
    padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;
}
#holdings .risk-card.bear { border-left-color: #ef5350; }
#holdings .risk-card h3 { margin:0 0 10px 0; font-size:15px; color:#e0e0e0; }
#holdings .risk-card .warn-list { margin:6px 0 0 0; padding-left:20px; }
#holdings .risk-card .warn-list li { color:#ffb74d; font-size:12.5px; margin-bottom:3px; }
#holdings .risk-card.bear .warn-list li { color:#ef5350; }

#holdings .tier-badge {
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-size:11px; font-weight:600;
}
#holdings .tier-0 { background:#333; color:#a0a0b0; }
#holdings .tier-1 { background:#1e3a5f; color:#81d4fa; }
#holdings .tier-2 { background:#2e7d32; color:#a5d6a7; }
#holdings .tier-3 { background:#f57c00; color:#fff3e0; }

#holdings .health-bar {
    display:inline-block; width:50px; height:6px; background:#2a2a4a;
    border-radius:3px; vertical-align:middle; margin-right:6px;
    overflow:hidden;
}
#holdings .health-fill { height:100%; transition:width .3s; }
#holdings .health-num { font-size:12px; font-weight:600; }

#holdings .add-signal {
    color:#4caf50; font-weight:600; font-size:12px;
}
#holdings .stop-value { font-family:'Consolas', monospace; font-size:12px; }

#holdings .summary-item.risk-item { border-left-color:#ef5350; }
#holdings .summary-item.profit-item { border-left-color:#26a69a; }
#holdings .summary-item.warn-item { border-left-color:#ffb74d; }
</style>
"""

    html = """
<section id="holdings">
    <h2>💼 我的持股</h2>
    <p style="color:#a0a0b0; font-size:13px; margin-bottom:14px;">
        持股資料儲存於瀏覽器 localStorage，不會上傳至任何伺服器。
        可匯出 CSV 覆蓋 <code>holdings.csv</code> 讓下次回測與賣出警報也使用這份名單。
        <br>🛡️ <b>風險管理（第一至三階段）</b>：ATR 動態停損、保本停損、分層止盈（1R/2R/3R）、
        Chandelier 吊燈停損、策略專屬出場、相對強度轉弱、投組集中度警示、持股健檢分數。
    </p>

    <div id="portfolio-risk-card"></div>

    <div class="holdings-toolbar">
        <button class="btn-primary" onclick="toggleHoldingForm()">➕ 新增持股</button>
        <button class="btn-secondary" onclick="exportHoldingsCSV()">⬇ 匯出 CSV</button>
        <button class="btn-secondary" onclick="document.getElementById('csv-import-input').click()">⬆ 匯入 CSV</button>
        <input type="file" id="csv-import-input" accept=".csv,text/csv"
               style="display:none" onchange="handleCSVImport(event)">
        <button class="btn-secondary" onclick="loadInitialHoldings()" id="btn-load-initial">📂 從專案 holdings.csv 載入</button>
        <button class="btn-danger" onclick="clearHoldings()">🗑 清空全部</button>
    </div>

    <div id="holdings-summary" class="holdings-summary"></div>

    <div id="holdings-form" class="holdings-form" style="display:none">
        <div class="form-grid">
            <div><label>證券代號 *</label><input id="f-stock-id" placeholder="如 2330" oninput="autoFillHoldingName()"></div>
            <div><label>證券名稱</label><input id="f-name" placeholder="自動帶入"></div>
            <div><label>買進日期 *</label><input id="f-buy-date" type="date"></div>
            <div><label>買進價 *</label><input id="f-buy-price" type="number" step="0.01" placeholder="成交均價"></div>
            <div><label>張數</label><input id="f-shares" type="number" step="1" value="1"></div>
            <div><label>目標價</label><input id="f-target" type="number" step="0.01"></div>
            <div><label>停損價</label><input id="f-stop" type="number" step="0.01"></div>
            <div><label>策略</label><select id="f-strategy"><option value="">-- 選擇 --</option>__STRATEGY_OPTIONS__</select></div>
            <div class="full-row"><label>備註</label><input id="f-note"></div>
        </div>
        <div class="form-actions">
            <button class="btn-primary" onclick="addHolding()">加入</button>
            <button class="btn-secondary" onclick="toggleHoldingForm()">取消</button>
            <span id="form-hint" style="color:#a0a0b0; font-size:12px; align-self:center;"></span>
        </div>
    </div>

    <div id="holdings-table-container"></div>
</section>
""".replace('__STRATEGY_OPTIONS__', strategy_options)

    script = """
<script>
const STOCK_DATA = __STOCK_JSON__;
const INITIAL_HOLDINGS = __INITIAL_JSON__;
const PORTFOLIO = __PORTFOLIO_JSON__;
const HAS_INITIAL = __HAS_INITIAL__;
const LS_KEY = 'stockHoldings_v2';  // v2: 含第三階段欄位
const LS_KEY_LEGACY = 'stockHoldings_v1';

function loadHoldings() {
    let raw = localStorage.getItem(LS_KEY);
    if (raw === null) {
        // 相容 v1：若存在舊版就遷移
        const legacy = localStorage.getItem(LS_KEY_LEGACY);
        if (legacy) {
            const arr = JSON.parse(legacy);
            localStorage.setItem(LS_KEY, legacy);
            return arr;
        }
        return [];
    }
    return JSON.parse(raw);
}
function saveHoldings(h) { localStorage.setItem(LS_KEY, JSON.stringify(h)); renderHoldings(); }

function toggleHoldingForm() {
    const el = document.getElementById('holdings-form');
    el.style.display = (el.style.display === 'none' || !el.style.display) ? '' : 'none';
    if (el.style.display !== 'none') {
        const bd = document.getElementById('f-buy-date');
        if (!bd.value) bd.value = new Date().toISOString().slice(0,10);
    }
}

function autoFillHoldingName() {
    const sid = document.getElementById('f-stock-id').value.trim();
    const data = STOCK_DATA[sid];
    const hint = document.getElementById('form-hint');
    if (data) {
        if (data.name && !document.getElementById('f-name').value) {
            document.getElementById('f-name').value = data.name;
        }
        const parts = [];
        if (data.close) parts.push(`現價 ${data.close}`);
        if (data.atr) parts.push(`ATR ${data.atr}`);
        if (data.rs) parts.push(`RS ${data.rs}`);
        if (data.sugShares) parts.push(`建議張數 ${data.sugShares}（風險 1% 資金）`);
        hint.textContent = parts.join(' | ');
        // 自動帶入建議停損價（買進價 - 1.5×ATR，若買進價已填）
        const bp = parseFloat(document.getElementById('f-buy-price').value);
        const stpEl = document.getElementById('f-stop');
        if (!stpEl.value && data.atr && bp && !isNaN(bp)) {
            stpEl.value = (bp - 1.5 * data.atr).toFixed(2);
        }
    } else {
        hint.textContent = sid ? '⚠ 找不到此代號資料' : '';
    }
}

function addHolding() {
    const get = id => document.getElementById(id).value.trim();
    const num = id => { const v = parseFloat(get(id)); return isNaN(v) ? null : v; };
    const h = {
        stock_id: get('f-stock-id'),
        name: get('f-name'),
        buy_date: get('f-buy-date'),
        buy_price: num('f-buy-price'),
        shares: num('f-shares') || 1,
        target_price: num('f-target'),
        stop_loss: num('f-stop'),
        strategy: get('f-strategy'),
        note: get('f-note'),
    };
    if (!h.stock_id || !h.buy_date || !h.buy_price) {
        alert('代號、買進日期、買進價為必填');
        return;
    }
    if (!h.name && STOCK_DATA[h.stock_id] && STOCK_DATA[h.stock_id].name) {
        h.name = STOCK_DATA[h.stock_id].name;
    }
    const all = loadHoldings();
    all.push(h);
    saveHoldings(all);
    ['f-stock-id','f-name','f-buy-price','f-target','f-stop','f-note'].forEach(
        id => document.getElementById(id).value = ''
    );
    document.getElementById('f-shares').value = 1;
    document.getElementById('f-strategy').value = '';
    document.getElementById('form-hint').textContent = '';
    toggleHoldingForm();
}

function deleteHolding(idx) {
    if (!confirm('確定刪除此筆持股？')) return;
    const h = loadHoldings();
    h.splice(idx, 1);
    saveHoldings(h);
}

function clearHoldings() {
    if (!confirm('清空全部持股？此動作無法復原')) return;
    localStorage.removeItem(LS_KEY);
    renderHoldings();
}

function loadInitialHoldings() {
    if (!HAS_INITIAL) { alert('無 holdings.csv 初始資料'); return; }
    if (loadHoldings().length > 0 &&
        !confirm('將覆蓋當前 localStorage 持股，確定？')) return;
    saveHoldings(INITIAL_HOLDINGS);
}

function exportHoldingsCSV() {
    const holdings = loadHoldings();
    if (holdings.length === 0) { alert('無持股可匯出'); return; }
    const header = ['證券代號','證券名稱','買進日期','買進價','目標價','停損價',
                    '張數','策略','備註',
                    '入場後最高','初始ATR','初始停損','分層狀態'];
    const esc = v => `"${String(v ?? '').replace(/"/g,'""')}"`;
    const rows = holdings.map(h => [
        h.stock_id, h.name, h.buy_date, h.buy_price ?? '',
        h.target_price ?? '', h.stop_loss ?? '',
        h.shares ?? '', h.strategy, h.note,
        h.max_high ?? '', h.initial_atr ?? '',
        h.initial_stop ?? '', h.tier_status ?? ''
    ].map(esc).join(','));
    const csv = '\uFEFF' + header.join(',') + '\\n' + rows.join('\\n');
    const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'holdings.csv';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

function parseCSVLine(line) {
    const out = [];
    let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"') {
            if (inQ && line[i+1] === '"') { cur += '"'; i++; }
            else inQ = !inQ;
        } else if (c === ',' && !inQ) { out.push(cur); cur = ''; }
        else cur += c;
    }
    out.push(cur);
    return out;
}

function handleCSVImport(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        try {
            const text = e.target.result.replace(/^\uFEFF/, '').replace(/\\r/g,'');
            const lines = text.split('\\n').filter(l => l.trim());
            if (lines.length < 2) { alert('CSV 內容不足'); return; }
            const header = parseCSVLine(lines[0]).map(s => s.trim());
            const map = {
                '證券代號':'stock_id', '證券名稱':'name', '買進日期':'buy_date',
                '買進價':'buy_price', '目標價':'target_price', '停損價':'stop_loss',
                '張數':'shares', '策略':'strategy', '備註':'note',
                '入場後最高':'max_high', '初始ATR':'initial_atr',
                '初始停損':'initial_stop', '分層狀態':'tier_status'
            };
            const numKeys = ['buy_price','target_price','stop_loss','shares',
                             'max_high','initial_atr','initial_stop'];
            const holdings = [];
            for (let i = 1; i < lines.length; i++) {
                const cells = parseCSVLine(lines[i]);
                const row = { stock_id:'', name:'', buy_date:'', buy_price:null,
                              target_price:null, stop_loss:null, shares:1,
                              strategy:'', note:'',
                              max_high:null, initial_atr:null,
                              initial_stop:null, tier_status:'' };
                header.forEach((h,j) => {
                    const key = map[h];
                    if (!key) return;
                    const v = (cells[j] || '').trim();
                    if (numKeys.includes(key)) {
                        row[key] = v ? parseFloat(v) : null;
                    } else { row[key] = v; }
                });
                if (row.stock_id && row.buy_date) holdings.push(row);
            }
            if (loadHoldings().length > 0 &&
                !confirm(`偵測到 ${holdings.length} 筆，覆蓋現有持股？（取消則合併）`)) {
                saveHoldings(loadHoldings().concat(holdings));
            } else {
                saveHoldings(holdings);
            }
            alert(`已匯入 ${holdings.length} 筆`);
        } catch (err) { alert('CSV 解析失敗: ' + err.message); }
    };
    reader.readAsText(file, 'utf-8');
    event.target.value = '';
}

function tierBadge(status) {
    const m = { '3R已達':'tier-3', '2R已達':'tier-2', '1R已達':'tier-1' };
    const cls = m[status] || 'tier-0';
    return `<span class="tier-badge ${cls}">${status || '未達'}</span>`;
}

function healthBar(score) {
    if (score == null || isNaN(score)) return '-';
    const color = score >= 70 ? '#26a69a' : (score >= 40 ? '#ffb74d' : '#ef5350');
    return `<span class="health-bar"><span class="health-fill" style="width:${score}%;background:${color}"></span></span><span class="health-num" style="color:${color}">${score}</span>`;
}

function renderPortfolioCard(totalCost, trackedMkt, trackedPnL, tracked, totalRisk) {
    const card = document.getElementById('portfolio-risk-card');
    if (!card) return;
    const bearish = !PORTFOLIO.bullish;
    const warnings = (PORTFOLIO.warnings || []).slice();
    const accSize = PORTFOLIO.account_size || 1000000;
    if (totalRisk > accSize * 0.05 && !warnings.some(w => w.includes('總風險'))) {
        warnings.push(`投組總風險 ${Math.round(totalRisk).toLocaleString()} 超過帳戶 5%`);
    }
    if (warnings.length === 0 && PORTFOLIO.bullish) {
        card.innerHTML = `<div class="risk-card">
            <h3>🛡️ 投組風險健檢：✅ 正常</h3>
            <p style="color:#a0a0b0; font-size:12.5px; margin:0;">
                大盤多頭，未偵測到集中度、虧損或風險過大警示。
                帳戶基準 ${accSize.toLocaleString()}，風險上限 ${Math.round(accSize*0.05).toLocaleString()}。
            </p></div>`;
        return;
    }
    card.innerHTML = `<div class="risk-card ${bearish?'bear':''}">
        <h3>🛡️ 投組風險健檢：${bearish?'⚠ 空頭 + ':''}${warnings.length} 項警示</h3>
        <ul class="warn-list">${warnings.map(w=>`<li>${w}</li>`).join('')}</ul>
    </div>`;
}

function renderHoldings() {
    const holdings = loadHoldings();
    const container = document.getElementById('holdings-table-container');
    const summary = document.getElementById('holdings-summary');

    if (holdings.length === 0) {
        container.innerHTML = '<p class="empty">尚無持股。請點「➕ 新增持股」手動輸入，或「⬆ 匯入 CSV」。</p>';
        summary.innerHTML = '';
        renderPortfolioCard(0, 0, 0, [], 0);
        return;
    }

    let totalCost = 0, totalMarket = 0, totalRisk = 0, healthSum = 0, healthCount = 0;
    const rows = holdings.map((h, idx) => {
        const data = STOCK_DATA[h.stock_id] || {};
        const liveClose = (data.close != null) ? Number(data.close) : null;
        const atr = data.atr;
        const shares = h.shares || 1;
        const cost = (h.buy_price || 0) * shares * 1000;
        totalCost += cost;

        // 方案 C：前端 fallback — 現價抓取失敗時用 max_high / buy_price 估算
        const hasLive = liveClose != null && liveClose > 0;
        let effectiveClose = liveClose, priceSrc = 'live';
        if (!hasLive) {
            if (h.max_high != null && Number(h.max_high) > 0) {
                effectiveClose = Number(h.max_high);
                priceSrc = 'max_high';
            } else if (h.buy_price != null && Number(h.buy_price) > 0) {
                effectiveClose = Number(h.buy_price);
                priceSrc = 'buy_price';
            } else {
                effectiveClose = null;
                priceSrc = 'none';
            }
        }

        // 即時計算當前停損（若賣出警報沒提供則前端近似）
        let currentStop = data.currentStop;
        if (currentStop == null && h.initial_stop && h.buy_price) {
            const stops = [h.initial_stop];
            const maxH = h.max_high || effectiveClose;
            if (maxH && maxH >= h.buy_price * 1.05) stops.push(h.buy_price);
            if (h.tier_status === '1R已達') stops.push(h.buy_price);
            if (atr && maxH && maxH > h.buy_price) {
                const mult = PORTFOLIO.bullish ? 3.0 : 2.0;
                stops.push(maxH - mult * atr);
            }
            currentStop = Math.max(...stops);
        }
        // 累計風險：若到停損，損失 = (buy - init_stop) × shares × 1000
        if (h.buy_price && h.initial_stop && h.buy_price > h.initial_stop) {
            totalRisk += (h.buy_price - h.initial_stop) * shares * 1000;
        }

        let pnlCell = '-', pnlCls = '';
        if (effectiveClose != null && h.buy_price) {
            const p = (effectiveClose - h.buy_price) / h.buy_price;
            const pctStr = (p >= 0 ? '+' : '') + (p*100).toFixed(1) + '%';
            if (hasLive) {
                pnlCell = pctStr;
                pnlCls = p > 0 ? 'gain' : (p < 0 ? 'loss' : '');
                totalMarket += liveClose * shares * 1000;
            } else {
                pnlCell = `<span class="offline-price" title="以 ${priceSrc === 'max_high' ? '入場後最高' : '買進價'} 估算">${pctStr}~</span>`;
            }
        }

        let distStpCell = '-', stpCls = '';
        if (currentStop && effectiveClose) {
            const d = (effectiveClose - currentStop) / effectiveClose;
            const dStr = (d*100).toFixed(1) + '%';
            distStpCell = hasLive ? dStr
                : `<span class="offline-price" title="估算">${dStr}~</span>`;
            if (hasLive && d <= 0.03) stpCls = 'near-stop';
        }

        const tierCell = tierBadge(data.tierStatus || h.tier_status || '未達');

        let healthCell = '-';
        if (data.health != null) {
            healthCell = healthBar(data.health);
            healthSum += data.health;
            healthCount++;
        }

        const addSignal = data.addSignal && data.addSignal !== '-'
            ? `<span class="add-signal">${data.addSignal}</span>` : '-';

        let action = '持有', actCls = 'action-hold';
        const sig = data.sellSignal || '';
        if (data.sellAction && data.sellAction !== '持有') {
            action = data.sellAction;
            if (action.includes('全部')) actCls = 'action-stoploss';
            else if (action.includes('分批')) actCls = 'action-alert';
            else if (action.includes('目標')) actCls = 'action-target';
            else actCls = 'action-alert';
        }
        // 若現價已跌破當前停損，強制標記（僅在 live 資料下觸發真警報）
        if (currentStop && hasLive && liveClose < currentStop) {
            action = '🛑 跌破動態停損'; actCls = 'action-stoploss';
        } else if (!hasLive) {
            action = '⚠ 現價取得失敗'; actCls = 'action-alert';
        }

        const fmt = (v, d=2) => (v == null ? '-' : Number(v).toFixed(d));
        const stopDisp = currentStop
            ? `<span class="stop-value">${currentStop.toFixed(2)}</span>` : '-';
        const initStopDisp = h.initial_stop
            ? `<span class="stop-value" title="初始停損 = 買進 - 1.5×ATR">${h.initial_stop.toFixed(2)}</span>` : '-';

        let closeCell = '-';
        if (hasLive) {
            closeCell = liveClose.toFixed(2);
        } else if (effectiveClose != null) {
            const srcLabel = priceSrc === 'max_high' ? '入場後最高' : '買進價';
            const tip = `yfinance 現價抓取失敗，以「${srcLabel}」近似顯示`;
            closeCell = `<span class="offline-price" title="${tip}">${effectiveClose.toFixed(2)}</span><span class="offline-badge" title="${tip}">離線</span>`;
        }

        return `<tr>
            <td>${h.stock_id}</td>
            <td>${h.name || '-'}</td>
            <td>${h.buy_date || '-'}</td>
            <td>${fmt(h.buy_price)}</td>
            <td>${shares}</td>
            <td>${closeCell}</td>
            <td class="${pnlCls}">${pnlCell}</td>
            <td>${tierCell}</td>
            <td>${initStopDisp}</td>
            <td>${stopDisp}</td>
            <td class="${stpCls}">${distStpCell}</td>
            <td>${healthCell}</td>
            <td>${addSignal}</td>
            <td class="${actCls}" title="${sig.replace(/"/g,'&quot;')}">${action}</td>
            <td>${h.strategy || '-'}</td>
            <td>${h.note || ''}</td>
            <td><button class="btn-mini" onclick="deleteHolding(${idx})">刪</button></td>
        </tr>`;
    }).join('');

    const thead = `<tr>
        <th>代號</th><th>名稱</th><th>買進日</th><th>買進價</th><th>張數</th>
        <th>現價</th><th>損益%</th>
        <th>分層</th><th title="初始停損">初始停損</th>
        <th title="動態停損（初始 / 保本 / 分層 / Chandelier 取高）">當前停損</th>
        <th>距停損</th>
        <th title="綜合技術+法人+距停損的健檢分數">健檢</th>
        <th>加碼訊號</th>
        <th>動作</th><th>策略</th><th>備註</th><th></th>
    </tr>`;

    container.innerHTML =
        `<div style="overflow-x:auto;"><table class="data-table holdings-table">
         <thead>${thead}</thead><tbody>${rows}</tbody></table></div>`;

    const tracked = holdings.filter(h => STOCK_DATA[h.stock_id] && STOCK_DATA[h.stock_id].close);
    const trackedCost = tracked.reduce((s,h)=>s+(h.buy_price||0)*(h.shares||1)*1000, 0);
    const trackedMkt = tracked.reduce((s,h)=>s+STOCK_DATA[h.stock_id].close*(h.shares||1)*1000, 0);
    const trackedPnL = trackedMkt - trackedCost;
    const trackedPnLPct = trackedCost > 0 ? (trackedPnL / trackedCost * 100).toFixed(1) : '0.0';
    const winRate = tracked.length > 0 ? (tracked.filter(h => STOCK_DATA[h.stock_id].close > h.buy_price).length / tracked.length * 100).toFixed(0) : '0';
    const avgHealth = healthCount > 0 ? Math.round(healthSum / healthCount) : null;
    const riskPct = totalCost > 0 ? (totalRisk / totalCost * 100).toFixed(1) : '0.0';

    summary.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item"><div class="label">持股檔數</div><div class="value">${holdings.length}</div></div>
            <div class="summary-item"><div class="label">總成本</div><div class="value">${Math.round(totalCost).toLocaleString()}</div></div>
            <div class="summary-item profit-item"><div class="label">可追蹤市值</div><div class="value">${Math.round(trackedMkt).toLocaleString()}</div></div>
            <div class="summary-item ${trackedPnL>=0?'profit-item':'risk-item'}"><div class="label">浮動損益</div><div class="value ${trackedPnL>=0?'gain':'loss'}">${(trackedPnL>=0?'+':'')}${Math.round(trackedPnL).toLocaleString()} (${trackedPnLPct}%)</div></div>
            <div class="summary-item"><div class="label">獲利檔比</div><div class="value">${tracked.filter(h => STOCK_DATA[h.stock_id].close > h.buy_price).length}/${tracked.length} (${winRate}%)</div></div>
            <div class="summary-item risk-item"><div class="label" title="全部持股若碰初始停損的總損失">投組總風險</div><div class="value">${Math.round(totalRisk).toLocaleString()} (${riskPct}%)</div></div>
            <div class="summary-item ${avgHealth==null?'':(avgHealth>=70?'profit-item':(avgHealth>=40?'warn-item':'risk-item'))}"><div class="label">平均健檢分數</div><div class="value">${avgHealth == null ? '-' : avgHealth + '/100'}</div></div>
        </div>`;

    renderPortfolioCard(totalCost, trackedMkt, trackedPnL, tracked, totalRisk);
}

// 初始化：若 localStorage 完全空且有種子資料，自動帶入一次
(function initHoldings() {
    if (HAS_INITIAL && localStorage.getItem(LS_KEY) === null) {
        localStorage.setItem(LS_KEY, JSON.stringify(INITIAL_HOLDINGS));
    }
    if (!HAS_INITIAL) {
        const btn = document.getElementById('btn-load-initial');
        if (btn) btn.style.display = 'none';
    }
    renderHoldings();
})();
</script>
"""
    script = (script
              .replace('__STOCK_JSON__', stock_json)
              .replace('__INITIAL_JSON__', initial_json)
              .replace('__PORTFOLIO_JSON__', portfolio_json)
              .replace('__HAS_INITIAL__', has_initial))

    return style + html + script


def save_html_report(sell_alerts, buy_dfs_map,
                     ranking=None, undervalued=None, early_potential=None,
                     backtest_result=None, all_data=None, revenue_data=None,
                     market_state=None, holdings_df=None,
                     portfolio_risk=None):
    """產生互動式 HTML 報告。
    buy_dfs_map: dict {策略名稱: DataFrame}"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{today_str}_上市上櫃策略報告.html"

    ohlcv_cache = backtest_result.ohlcv_cache if backtest_result else {}

    top_ids = []
    if ranking is not None and not ranking.empty:
        top_ids = ranking['證券代號'].tolist()[:30]

    early_ids = []
    if early_potential is not None and not early_potential.empty:
        early_ids = early_potential['證券代號'].tolist()

    bt_charts = _build_backtest_charts(backtest_result)
    stock_charts = _build_stock_charts(top_ids, all_data, revenue_data, ohlcv_cache)
    early_charts = _build_stock_charts(early_ids, all_data, revenue_data, ohlcv_cache)
    rev_chart = _build_revenue_chart(revenue_data, top_ids)

    sections = [
        ("early", "潛力股提前佈局", early_potential),
        ("ranking", "綜合買入潛力 TOP 30", ranking),
        ("undervalued", "低估股篩選", undervalued),
        ("sell", "賣出警報（強化版）", sell_alerts),
    ]
    strategy_titles = {
        '策略1': '策略1：外資連續買超 + MA',
        '策略2': '策略2：投信連續買超 + MA',
        '策略3': '策略3：三法人共識買超 + MA',
        '策略4': '策略4：量價齊揚突破',
        '策略5': '策略5：RSI 超賣反彈 + 法人進場',
        '策略6': '策略6：布林通道收斂突破',
        '策略7': '策略7：月營收創高 + 法人買超',
        '策略8': '策略8：融資減少 / 融券增加',
        '策略9': '策略9：MACD 金叉 + 紅柱擴大',
        '策略10': '策略10：KD 低檔黃金交叉',
        '策略11': '策略11：Darvas 盒突破（52週高）',
        '策略12': '策略12：底部價量背離',
        '策略13': '策略13：相對強度領漲',
    }
    for idx, (sname, df) in enumerate(buy_dfs_map.items(), 1):
        sec_id = f"s{idx}"
        title = strategy_titles.get(sname, sname)
        sections.append((sec_id, title, df))

    holdings_html = _build_holdings_section(sell_alerts, holdings_df, all_data,
                                            portfolio_risk=portfolio_risk)

    table_sections = ''
    nav_items = '<a href="#market">大盤</a><a href="#holdings">我的持股</a><a href="#backtest">回測績效</a>'
    for sec_id, title, df in sections:
        nav_items += f'<a href="#{sec_id}">{title[:6]}</a>'
        table_sections += f'''
        <section id="{sec_id}">
            <h2>{title}</h2>
            {_df_to_html_table(df)}
        </section>'''

    nav_items += '<a href="#charts">個股圖表</a>'
    nav_items += '<a href="#early-charts">潛力股圖表</a>'
    if rev_chart:
        nav_items += '<a href="#revenue">營收</a>'

    bt_summary_html = ''
    if backtest_result and not backtest_result.summary.empty:
        bt_summary_html = _df_to_html_table(backtest_result.summary)

    # 大盤狀態區
    ms = market_state or get_taiex_state()
    desc = ms.get('desc', '未知')
    bullish = ms.get('bullish', True)
    m_color = '#26a69a' if bullish else '#ef5350'
    m_close = ms.get('close')
    m_ma60 = ms.get('ma_60')
    m_ma20 = ms.get('ma_20')
    m_ret = ms.get('return_20d')
    market_html = f'''
    <section id="market">
        <h2>大盤（TAIEX）狀態</h2>
        <div style="background: var(--bg-card); padding: 16px 20px; border-radius: 8px;
                    border-left: 4px solid {m_color};">
            <div style="display: flex; gap: 30px; flex-wrap: wrap; font-size: 15px;">
                <div><strong>狀態：</strong><span style="color:{m_color}; font-weight:600;">{desc}</span></div>
                <div><strong>加權指數：</strong>{f"{m_close:,.0f}" if m_close else "-"}</div>
                <div><strong>MA20：</strong>{f"{m_ma20:,.0f}" if m_ma20 else "-"}</div>
                <div><strong>MA60：</strong>{f"{m_ma60:,.0f}" if m_ma60 else "-"}</div>
                <div><strong>20日報酬：</strong>{f"{m_ret*100:.2f}%" if m_ret is not None else "-"}</div>
            </div>
            <p style="color: var(--text-secondary); font-size: 13px; margin-top: 10px;">
                {'多頭環境，策略進場積極度可正常。' if bullish else '空頭/震盪環境，綜合分數已自動打折，建議只選最高分標的並縮小部位。'}
            </p>
        </div>
    </section>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>策略報告 - {today_str}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
    --bg-primary: #0f0f1a;
    --bg-card: #1a1a2e;
    --bg-table-row: #16213e;
    --text-primary: #e0e0e0;
    --text-secondary: #a0a0b0;
    --accent: #42a5f5;
    --accent-green: #26a69a;
    --accent-red: #ef5350;
    --border: #2a2a4a;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: 'Segoe UI', -apple-system, sans-serif;
    background: var(--bg-primary); color: var(--text-primary);
    line-height: 1.6;
}}
nav {{
    position: sticky; top: 0; z-index: 100;
    background: #0d0d1a; border-bottom: 1px solid var(--border);
    padding: 8px 16px; display: flex; flex-wrap: wrap;
    gap: 4px; overflow-x: auto;
}}
nav a {{
    color: var(--text-secondary); text-decoration: none;
    padding: 6px 14px; border-radius: 6px; font-size: 13px;
    white-space: nowrap; transition: all 0.2s;
}}
nav a:hover {{ background: var(--bg-card); color: var(--accent); }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
header {{
    text-align: center; padding: 30px 0 10px;
    border-bottom: 1px solid var(--border); margin-bottom: 24px;
}}
header h1 {{ font-size: 28px; color: #fff; margin-bottom: 6px; }}
header .subtitle {{ color: var(--text-secondary); font-size: 14px; }}
section {{ margin-bottom: 36px; }}
h2 {{
    font-size: 20px; color: #fff; padding: 12px 0;
    border-bottom: 2px solid var(--accent); margin-bottom: 16px;
}}
.data-table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
    background: var(--bg-card); border-radius: 8px; overflow: hidden;
}}
.data-table th {{
    background: #0e1628; color: var(--accent); padding: 10px 12px;
    text-align: left; font-weight: 600; position: sticky; top: 0;
    cursor: pointer; user-select: none; white-space: nowrap;
}}
.data-table th:hover {{ color: #fff; }}
.data-table td {{
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    white-space: nowrap;
}}
.data-table tr:nth-child(even) {{ background: var(--bg-table-row); }}
.data-table tr:hover {{ background: #1e3a5f; }}
.empty {{ color: var(--text-secondary); font-style: italic; padding: 20px; }}
.stock-card {{ margin-bottom: 8px; }}
.accordion {{
    background: var(--bg-card); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 12px 20px; width: 100%; text-align: left;
    cursor: pointer; font-size: 15px; font-weight: 600;
    transition: background 0.2s;
}}
.accordion:hover {{ background: #1e3a5f; }}
.accordion::after {{ content: '\\25BC'; float: right; transition: transform 0.3s; }}
.accordion.active::after {{ transform: rotate(180deg); }}
.accordion-content {{
    max-height: 0; overflow: hidden;
    transition: max-height 0.4s ease-out;
    background: var(--bg-card); border-radius: 0 0 6px 6px;
}}
.accordion-content.open {{ max-height: 5000px; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
footer {{
    text-align: center; padding: 30px 0; color: var(--text-secondary);
    font-size: 12px; border-top: 1px solid var(--border); margin-top: 40px;
}}
</style>
</head>
<body>
<nav>{nav_items}</nav>
<div class="container">

<header>
    <h1>上市 + 上櫃策略報告</h1>
    <p class="subtitle">{today_str} &nbsp;|&nbsp; 執行時間 {datetime.now().strftime('%H:%M')}</p>
</header>

{market_html}

{holdings_html}

<section id="backtest">
    <h2>回測績效總覽</h2>
    {bt_summary_html}
    <div style="margin-top:20px;">{bt_charts}</div>
</section>

{table_sections}

<section id="revenue">
    <h2>營收年增率</h2>
    {rev_chart if rev_chart else '<p class="empty">無營收資料</p>'}
</section>

<section id="charts">
    <h2>個股技術分析圖表（TOP 30）</h2>
    {stock_charts if stock_charts else '<p class="empty">無圖表資料</p>'}
</section>

<section id="early-charts">
    <h2>潛力股提前佈局圖表</h2>
    {early_charts if early_charts else '<p class="empty">無圖表資料</p>'}
</section>

<footer>
    <p>投資有風險，以上資訊僅供參考，請自行判斷。</p>
    <p>資料來源：TWSE、TPEx、MOPS、Yahoo Finance</p>
</footer>

</div>

<script>
function toggleAccordion(btn) {{
    btn.classList.toggle('active');
    var content = btn.nextElementSibling;
    content.classList.toggle('open');
}}

document.querySelectorAll('.data-table th').forEach(function(th) {{
    th.addEventListener('click', function() {{
        var table = th.closest('table');
        var idx = Array.from(th.parentNode.children).indexOf(th);
        var rows = Array.from(table.querySelectorAll('tbody tr'));
        var asc = th.dataset.asc !== 'true';
        th.dataset.asc = asc;
        rows.sort(function(a, b) {{
            var av = a.children[idx].textContent.trim();
            var bv = b.children[idx].textContent.trim();
            var an = parseFloat(av.replace(/,/g, ''));
            var bn = parseFloat(bv.replace(/,/g, ''));
            if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
            return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        }});
        var tbody = table.querySelector('tbody');
        rows.forEach(function(r) {{ tbody.appendChild(r); }});
    }});
}});
</script>
</body>
</html>'''

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML 報告儲存：{filename}")
    return filename


# ===================== 主程式 =====================

if __name__ == "__main__":
    from backtest import run_backtest

    try:
        # 每日啟動時先跑排程器：依週期自動執行 GPU 網格/ML 重訓/深度模型重訓
        try:
            from scheduler import run_scheduled_tasks
            run_scheduled_tasks()
        except Exception as _sched_err:
            print(f"⚠ 排程器執行失敗（不影響主流程）：{_sched_err}")

        _technicals_cache.clear()

        print("抓取大盤 TAIEX 狀態...")
        market_state = get_taiex_state()
        print(f"  大盤：{market_state['desc']} (TAIEX={market_state.get('close')}, 20日報酬={market_state.get('return_20d', 0)*100:.2f}%)")

        all_data = get_recent_institutional()

        all_stock_ids = all_data['證券代號'].unique().tolist()
        holdings = load_holdings()
        holding_ids = holdings['證券代號'].tolist() if not holdings.empty else []
        prefetch_technicals(all_stock_ids + holding_ids)

        # 第一～三階段：先更新持股狀態（入場後最高、初始ATR、初始停損、分層狀態）
        if not holdings.empty:
            holdings = _update_holdings_state(holdings)
            # 寫回 CSV，保留狀態供下次使用
            try:
                save_holdings(holdings)
            except Exception as e:
                print(f"⚠ 無法寫回 holdings.csv：{e}")

        sell_alerts = check_sell_signals(all_data, holdings)

        # 投組風險總覽
        portfolio_risk = check_portfolio_risk(holdings)
        if portfolio_risk.get('warnings'):
            print("\n===== 投組風險警示 =====")
            for w in portfolio_risk['warnings']:
                print(f"  • {w}")

        buy1 = strategy1(all_data)
        buy2 = strategy2(all_data)
        buy3 = strategy3(all_data)
        buy4 = strategy4(all_data)
        buy5 = strategy5(all_data)
        buy6 = strategy6(all_data)

        print("\n抓取月營收資料...")
        revenue_data = get_latest_revenue()
        buy7 = strategy7(all_data, revenue_data)

        print("\n抓取融資資料...")
        margin_data = get_recent_margin()
        buy8 = strategy8(all_data, margin_data)

        print("\n執行新策略 9-13...")
        buy9 = strategy9(all_data)
        buy10 = strategy10(all_data)
        buy11 = strategy11(all_data)
        buy12 = strategy12(all_data)
        buy13 = strategy13(all_data)

        buy_dfs_map = {
            '策略1': buy1, '策略2': buy2, '策略3': buy3, '策略4': buy4,
            '策略5': buy5, '策略6': buy6, '策略7': buy7, '策略8': buy8,
            '策略9': buy9, '策略10': buy10, '策略11': buy11,
            '策略12': buy12, '策略13': buy13,
        }

        # 回測（使用所有策略候選股）
        bt_stock_ids = set()
        for df in buy_dfs_map.values():
            if df is not None and not df.empty and '證券代號' in df.columns:
                bt_stock_ids.update(df['證券代號'].tolist())
        bt_stock_ids.update(all_stock_ids[:100])

        print("\n執行回測...")
        bt_result = run_backtest(list(bt_stock_ids), period='1y')

        # 以回測 Sharpe 動態調整權重
        backtest_scores = {}
        if bt_result and not bt_result.summary.empty:
            max_sharpe = max(float(bt_result.summary['Sharpe'].max()), 0.01)
            for _, row in bt_result.summary.iterrows():
                raw_name = str(row['策略'])
                # 從 '策略1-3 (法人+MA近似)' 解析出可識別鍵
                for key in buy_dfs_map.keys():
                    if raw_name.startswith(key) or raw_name.startswith(key.replace('策略', '策略1-3')):
                        sharpe = float(row['Sharpe'])
                        backtest_scores[key] = max(min(sharpe / max_sharpe, 1.0), 0.0)
                        break

        print("\n計算綜合買入潛力排名...")
        ranking = compute_composite_ranking(buy_dfs_map, all_data, backtest_scores=backtest_scores)

        print("\n嘗試套用 ML ranker（路線 2）...")
        try:
            all_candidates_df = pd.concat(
                [df for df in buy_dfs_map.values() if df is not None and not df.empty],
                ignore_index=True
            ) if any((df is not None and not df.empty) for df in buy_dfs_map.values()) else None
            ranking_ml, used_ml = apply_ml_ranking(ranking, candidate_pool_df=all_candidates_df)
            if used_ml:
                print("  ✓ 已採用 ML 融合排名（rule 40% + ml 60%）")
                ranking = ranking_ml
            else:
                print("  · 未啟用 ML（找不到 ranker_model.json 或無資料），沿用規則排名")
        except Exception as _ml_err:
            import traceback; traceback.print_exc()
            print(f"  ⚠ ML 融合失敗，退回規則排名：{_ml_err}")

        print("\n嘗試套用 Deep ranker（路線 3）...")
        try:
            ranking_deep, used_deep = apply_deep_ranking(ranking, deep_weight=0.35)
            if used_deep:
                print("  ✓ 已採用 三層融合排名（rule + ml + deep）")
                ranking = ranking_deep
            else:
                print("  · 未啟用 Deep（找不到 deep_model.pt 或無資料），沿用現有排名")
        except Exception as _dp_err:
            import traceback; traceback.print_exc()
            print(f"  ⚠ Deep 融合失敗，沿用現有排名：{_dp_err}")

        print("\n篩選低估股...")
        undervalued = find_undervalued(all_data)

        print("\n偵測潛力股提前佈局...")
        early_potential = find_early_potential(all_data)

        save_html_report(
            sell_alerts, buy_dfs_map,
            ranking=ranking, undervalued=undervalued,
            early_potential=early_potential,
            backtest_result=bt_result,
            all_data=all_data, revenue_data=revenue_data,
            market_state=market_state,
            holdings_df=holdings,
            portfolio_risk=portfolio_risk,
        )

        if not early_potential.empty:
            print(f"\n===== 潛力股提前佈局 ({len(early_potential)} 檔) =====")
            print(early_potential.head(20).to_string(index=False))
        else:
            print("\n潛力股提前佈局：無符合")

        if not ranking.empty:
            print("\n===== 綜合買入潛力 TOP 30 =====")
            print(ranking.to_string(index=False))
        else:
            print("\n綜合買入潛力排名：無符合")

        if not undervalued.empty:
            print(f"\n===== 低估股篩選 ({len(undervalued)} 檔) =====")
            print(undervalued.head(20).to_string(index=False))
        else:
            print("\n低估股篩選：無符合")

        for name, df in buy_dfs_map.items():
            print(f"\n{name} 前10")
            print(df.head(10) if df is not None and not df.empty else "無符合")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"錯誤: {e}")

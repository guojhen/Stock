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
CHANDELIER_ATR_MULT_SHOCK = 1.5   # D4：大盤跌破 MA60 > 5% 時加倍緊縮
MARKET_DRAWDOWN_THRESHOLD = -0.05  # D4：TAIEX 距 MA60 跌幅閾值
BREAKEVEN_PROFIT_TRIGGER = 0.05   # 獲利 N% 後啟動保本停損
ACCOUNT_SIZE_DEFAULT = 1_000_000  # 帳戶總資金（可由 HOLDINGS_ACCOUNT_SIZE 環境變數覆蓋）
POSITION_RISK_PCT = 0.01          # 單筆最多損失比例（凱利簡化版）

_technicals_cache = {}
# Tier B6：共用原始 OHLCV 快取（避免 prefetch_technicals 與 _batch_download 重複下載）
_raw_ohlcv_cache: dict = {}
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
        'ma_20': None, 'ma_60': None,                     # 中長期 MA（A2/S20 用）
        'ma_20_slope': None, 'ma_60_slope': None,          # 5 日斜率（趨勢方向）
        'ma20_60_golden_cross': False,                     # S40: MA20/MA60 黃金交叉
        'ma20_60_cross_days_ago': None,
        'prev_close_above_ma60': None,                     # S20：前一日是否站上 MA60
        'days_above_ma60': None,                           # S20：連續站上 MA60 天數
        'rsi': None, 'rsi_history': [],
        'bb_upper': None, 'bb_lower': None, 'bb_width': None, 'bb_width_pctl': None,
        'bb_squeeze_days': 0,
        'volume_ratio': None, 'avg_volume_20d': None,
        'volume_ratio_5d': None,                           # S18：近 5 日量比
        'prev_high_20d': None, 'prev_close': None,
        'low_10d': None,                                   # D2：Donchian 10 日低
        'low_20d': None,
        'high_30d': None, 'low_30d': None,                # S18：30 日盤整振幅
        'range_30d_pct': None,                             # S18：(high-low)/low
        'atr_expansion_ratio': None,                       # S18：近 5 日 ATR / 20 日 ATR
        'volume_history_15': [],                           # S19：近 15 日成交量
        'price_change_history_15': [],                     # S19：近 15 日漲跌（用來判斷上漲/下跌日）
        'today_open': None,                                 # S24：今日開盤
        'today_high': None,                                 # S23/S24：今日最高
        'today_low': None,                                  # S23：今日最低
        'prev_high': None,                                  # S23/S24：昨日最高
        'prev_low': None,                                   # S23：昨日最低
        'inside_days_count': 0,                             # S23：連續內包日
        'atr': None, 'atr_pct': None,
        'macd': None, 'macd_signal': None, 'macd_hist': None, 'macd_hist_prev': None,
        'kd_k': None, 'kd_d': None, 'k_history': [],
        'high_52w': None, 'low_52w': None, 'dist_from_52w_high': None,
        'return_20d': None, 'rs_vs_taiex': None,
        'rs_line_60d_high': None,                          # S16：RS line 是否創 60 日新高
        'days_close_below_52w_high_pct': None,             # S16 用：距 52 週高 N%
        'upper_shadow_pct': None,
        'close_history_30': [],
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
        ma_20_series = c.rolling(20).mean()
        ma_60_series = c.rolling(60).mean()
        close_v = c.iloc[-1].item()
        ma_s = _safe_float(ma_s_series.iloc[-1])
        ma_l = _safe_float(ma_l_series.iloc[-1])
        ma_20 = _safe_float(ma_20_series.iloc[-1])
        ma_60 = _safe_float(ma_60_series.iloc[-1])

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
        t['ma_20'] = ma_20
        t['ma_60'] = ma_60
        t['ma_status'] = ma_status

        # MA 斜率（5 日變化率）— 用於 A2 / S18 / S20 過濾下跌通道
        if ma_20 is not None and len(ma_20_series.dropna()) >= 6:
            prev_ma20 = _safe_float(ma_20_series.dropna().iloc[-6])
            if prev_ma20 and prev_ma20 != 0:
                t['ma_20_slope'] = (ma_20 - prev_ma20) / prev_ma20
        if ma_60 is not None and len(ma_60_series.dropna()) >= 6:
            prev_ma60 = _safe_float(ma_60_series.dropna().iloc[-6])
            if prev_ma60 and prev_ma60 != 0:
                t['ma_60_slope'] = (ma_60 - prev_ma60) / prev_ma60

        # S40：MA20 / MA60 黃金交叉偵測（昨日 MA20 ≤ MA60、今日 MA20 > MA60）
        if (ma_20 is not None and ma_60 is not None
                and len(ma_20_series.dropna()) >= 2
                and len(ma_60_series.dropna()) >= 2):
            try:
                prev_ma20_y = _safe_float(ma_20_series.dropna().iloc[-2])
                prev_ma60_y = _safe_float(ma_60_series.dropna().iloc[-2])
                if (prev_ma20_y is not None and prev_ma60_y is not None
                        and prev_ma20_y <= prev_ma60_y and ma_20 > ma_60):
                    t['ma20_60_golden_cross'] = True
                else:
                    t['ma20_60_golden_cross'] = False
                # 距離黃金交叉天數（最近 N 日是否曾發生）
                ma20_arr = ma_20_series.values
                ma60_arr = ma_60_series.values
                days_since = None
                for i in range(len(ma20_arr) - 1, 0, -1):
                    a, b = ma20_arr[i], ma60_arr[i]
                    pa, pb = ma20_arr[i-1], ma60_arr[i-1]
                    if (pd.notna(a) and pd.notna(b) and pd.notna(pa) and pd.notna(pb)
                            and pa <= pb and a > b):
                        days_since = (len(ma20_arr) - 1 - i)
                        break
                t['ma20_60_cross_days_ago'] = days_since
            except Exception:
                pass

        # S20：是否首次站上 MA60（前一日仍在下方、今日站上）
        if ma_60 is not None and len(c) >= 61:
            prev_close = _safe_float(c.iloc[-2])
            prev_ma60 = _safe_float(ma_60_series.iloc[-2])
            if prev_close is not None and prev_ma60 is not None:
                t['prev_close_above_ma60'] = prev_close > prev_ma60
            # 連續站上 MA60 天數（從最近往回算）
            try:
                close_arr = c.values
                ma60_arr = ma_60_series.values
                cnt = 0
                for i in range(len(close_arr) - 1, -1, -1):
                    cv = close_arr[i]
                    mv = ma60_arr[i]
                    if cv is None or mv is None:
                        break
                    cv = float(cv); mv = float(mv)
                    if np.isnan(cv) or np.isnan(mv) or cv <= mv:
                        break
                    cnt += 1
                t['days_above_ma60'] = cnt
            except Exception:
                pass

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
            vol_avg5 = data['Volume'].rolling(5).mean()
            vl = _safe_float(data['Volume'].iloc[-1])
            va = _safe_float(vol_avg.iloc[-1])
            va5 = _safe_float(vol_avg5.iloc[-1])
            if va and va > 0:
                t['avg_volume_20d'] = va / 1000
                if vl:
                    t['volume_ratio'] = vl / va
            if va5 and va5 > 0 and va and va > 0:
                t['volume_ratio_5d'] = va5 / va  # >1 表示近期量增；<1 表示量縮
            # S19：保留近 15 日成交量 + 漲跌方向，給 Pocket Pivot 判斷用
            if len(data) >= 16:
                vol_tail = data['Volume'].tail(15).tolist()
                close_tail = data['Close'].tail(16).tolist()
                t['volume_history_15'] = [
                    float(v) for v in vol_tail
                    if v is not None and pd.notna(v)
                ]
                pc = []
                for i in range(1, len(close_tail)):
                    a, b = close_tail[i - 1], close_tail[i]
                    if a is None or b is None or pd.isna(a) or pd.isna(b) or a <= 0:
                        pc.append(0.0)
                    else:
                        pc.append((b - a) / a)
                t['price_change_history_15'] = pc

        # --- Previous 20-day high & prev close ---
        if 'High' in data.columns and len(data) > 1:
            t['prev_high_20d'] = _safe_float(data['High'].iloc[:-1].tail(20).max())
        t['prev_close'] = _safe_float(c.iloc[-2]) if len(data) > 1 else None

        # D2：近 10/20 日低點（Donchian channel 出場用，排除今日避免自我參考）
        if 'Low' in data.columns and len(data) > 10:
            t['low_10d'] = _safe_float(data['Low'].iloc[:-1].tail(10).min())
        if 'Low' in data.columns and len(data) > 20:
            t['low_20d'] = _safe_float(data['Low'].iloc[:-1].tail(20).min())

        # S18：30 日 high-low 振幅（盤整偵測）
        if 'High' in data.columns and 'Low' in data.columns and len(data) > 30:
            tail30 = data.tail(30)
            high30 = _safe_float(tail30['High'].max())
            low30 = _safe_float(tail30['Low'].min())
            if high30 and low30 and low30 > 0:
                t['high_30d'] = high30
                t['low_30d'] = low30
                t['range_30d_pct'] = (high30 - low30) / low30

        # S23/S24：今日 OHLC + 昨日 high/low + 連續內包日數
        if 'Open' in data.columns and len(data) >= 1:
            t['today_open'] = _safe_float(data['Open'].iloc[-1])
        if 'High' in data.columns and len(data) >= 2:
            t['today_high'] = _safe_float(data['High'].iloc[-1])
            t['prev_high'] = _safe_float(data['High'].iloc[-2])
        if 'Low' in data.columns and len(data) >= 2:
            t['today_low'] = _safe_float(data['Low'].iloc[-1])
            t['prev_low'] = _safe_float(data['Low'].iloc[-2])
        # S23：連續內包日（today high < prev high AND today low > prev low）
        if 'High' in data.columns and 'Low' in data.columns and len(data) >= 6:
            highs = data['High'].tail(6).tolist()
            lows = data['Low'].tail(6).tolist()
            ic = 0
            for i in range(len(highs) - 1, 0, -1):
                if (highs[i] is None or highs[i - 1] is None
                        or lows[i] is None or lows[i - 1] is None):
                    break
                if highs[i] < highs[i - 1] and lows[i] > lows[i - 1]:
                    ic += 1
                else:
                    break
            t['inside_days_count'] = ic

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
            # S18：ATR 擴張比 — 近 5 日 ATR / 過去 20 日平均 ATR
            if len(atr_series.dropna()) >= 25:
                atr5 = _safe_float(atr_series.tail(5).mean())
                atr20_prev = _safe_float(atr_series.tail(25).head(20).mean())
                if atr5 and atr20_prev and atr20_prev > 0:
                    t['atr_expansion_ratio'] = atr5 / atr20_prev

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

        # --- Tier A1 + S16：保留近 60 日收盤序列
        # （A1 RS 加速度只需 12 點，S16 RS line 60d 新高需要 60 點；欄名保留向下相容）
        try:
            tail60 = c.tail(60).dropna().tolist()
            t['close_history_30'] = [float(x) for x in tail60]
        except Exception:
            t['close_history_30'] = []

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
    """B6：在計算技術指標的同時，把原始 OHLCV 存入 _raw_ohlcv_cache 共用。"""
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
                        try:
                            _raw_ohlcv_cache[batch_ids[0]] = _normalize_ohlcv(raw)
                        except Exception:
                            pass
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
                                try:
                                    _raw_ohlcv_cache[sid] = _normalize_ohlcv(stock_data)
                                except Exception:
                                    pass
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            print(f"  批次下載錯誤: {e}")
        done = min(i + batch_size, len(stock_ids))
        print(f"  進度: {done}/{len(stock_ids)}")


def _normalize_ohlcv(df):
    """整平 yfinance 可能回傳的 MultiIndex columns，統一為單層欄位。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    if hasattr(out.columns, 'nlevels') and out.columns.nlevels > 1:
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    return out


def get_shared_ohlcv(stock_ids, period='9mo'):
    """B6：共用 OHLCV 取得器。
    優先從 _raw_ohlcv_cache 拿（prefetch_technicals 已下載過 1y），
    缺的部分再呼叫 backtest._batch_download 補齊。
    """
    out = {}
    missing = []
    for sid in stock_ids:
        sid = str(sid).strip()
        if sid in _raw_ohlcv_cache and not _raw_ohlcv_cache[sid].empty:
            out[sid] = _raw_ohlcv_cache[sid]
        else:
            missing.append(sid)
    if missing:
        try:
            from backtest import _batch_download
            extra = _batch_download(missing, period=period)
            out.update(extra)
            for sid, df in extra.items():
                _raw_ohlcv_cache[sid] = df
        except Exception as e:
            print(f"  ⚠ 共用 OHLCV 補抓失敗：{e}")
    return out


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
            'close_history_30': [float(x) for x in close.tail(60).tolist()],
        })
    except Exception as e:
        print(f"大盤資料擷取失敗: {e}")
        _taiex_returns_cache.update({
            'bullish': True, 'close': None, 'ma_60': None, 'ma_20': None,
            'return_20d': 0.0, 'desc': '未知（預設多頭）',
            'close_history_30': [],
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
    """RSI 超賣反彈 + MACD/KD 任一底部確認 + 法人轉買。
    A2 強化：過濾下跌通道（要求 close > MA60 或 MA20 斜率 ≥ 0），避免低 RSI 持續破底。
    """
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
        if t['ma_long'] is None:
            continue
        if t['close'] < t['ma_long'] * 0.97:
            continue
        # ── A2 過濾下跌通道：MA20 斜率不為負 或 收盤站上 MA60 ──
        ma20_slope = t.get('ma_20_slope')
        ma60 = t.get('ma_60')
        in_downtrend = True
        trend_tag = []
        if ma60 is not None and t['close'] >= ma60 * 0.98:
            in_downtrend = False
            trend_tag.append('站上MA60')
        if ma20_slope is not None and ma20_slope >= -0.005:  # MA20 斜率 ≥ -0.5%
            in_downtrend = False
            if ma20_slope > 0:
                trend_tag.append(f'MA20上揚{ma20_slope*100:.1f}%')
        if in_downtrend:
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
            '趨勢過濾': '+'.join(trend_tag) if trend_tag else '弱多',
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('RSI', ascending=True, inplace=True)
    return df


# ===================== 策略 6：布林通道收斂突破 =====================

def strategy6(all_df):
    """布林通道收斂（>=5 日）+ 突破上軌 + 法人買超 + 流動性。
    A3 強化：方向確認 — MACD_hist > 0 且 量比 ≥ 1.8（過濾假突破）。
    """
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
        if t.get('bb_squeeze_days', 0) < 5:
            continue
        if t['close'] < t['bb_upper']:
            continue
        if t.get('upper_shadow_pct') is not None and t['upper_shadow_pct'] > 0.03:
            continue
        # ── A3 方向確認：MACD_hist > 0 且 量比 ≥ 1.8 ──
        macd_h = t.get('macd_hist')
        vol_r = t.get('volume_ratio')
        confirm = []
        if macd_h is not None and macd_h > 0:
            confirm.append(f'MACD柱>0({macd_h:.3f})')
        else:
            continue
        if vol_r is not None and vol_r >= 1.8:
            confirm.append(f'量比{vol_r:.1f}x')
        else:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略6',
            '帶寬百分位': round(t['bb_width_pctl'] * 100, 1),
            '收斂天數': t.get('bb_squeeze_days', 0),
            '最新收盤': t['close'],
            '布林上軌': round(t['bb_upper'], 2),
            'MA 狀態': t['ma_status'],
            '方向確認': '+'.join(confirm),
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
    """A6 + A9 強化版：用變化率，並把訊號分成 ⭐ 弱 / ⭐⭐ 中 / ⭐⭐⭐ 強三級。
    入選條件：
      - 融資減幅 ≥ 5%（過濾大型股小幅波動）→ 籌碼結構轉強
      - **OR** 融券增加且 融資未增 → 軋空潛力
      - 法人買超 + MA 多頭
    A9 軋空評分：
      base：融券增加% × 0.6 + 融資減幅% × 0.4
      bonus：融券餘額 / 融資餘額 ≥ 0.10（券資比偏高 → 軋空火藥）→ +0.5×base
      bonus：法人 5 日累計買超 ≥ 5000 張 → +20%
      強度：score < 5 → ⭐；5 ≤ score < 12 → ⭐⭐；≥ 12 → ⭐⭐⭐
    """
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
        if not balances or balances[-1] <= 0:
            continue
        signals = []
        margin_drop_pct = 0.0
        short_rise_pct = 0.0
        margin_drop_lots = 0
        short_rise_lots = 0
        short_to_margin_ratio = 0.0
        if balances[0] < balances[-1]:
            margin_drop_pct = (balances[-1] - balances[0]) / balances[-1]
            margin_drop_lots = int(balances[-1] - balances[0])
            if margin_drop_pct >= 0.05:
                signals.append(f'融資減幅 {margin_drop_pct*100:.1f}%')
        # 融券增加且 融資未增 → 軋空潛力
        margin_not_up = balances[0] <= balances[-1] * 1.005
        if '融券餘額' in mg.columns:
            shorts = mg['融券餘額'].tolist()
            if len(shorts) >= 2 and shorts[-1] > 0:
                if shorts[0] > shorts[-1] * 1.1 and margin_not_up:
                    short_rise_pct = (shorts[0] - shorts[-1]) / max(shorts[-1], 1)
                    short_rise_lots = int(shorts[0] - shorts[-1])
                    signals.append(f'融券+{short_rise_pct*100:.0f}% (軋空)')
                # 券資比（軋空火藥度）
                if balances[-1] > 0:
                    short_to_margin_ratio = shorts[-1] / balances[-1]
        if not signals:
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t['ma_long'] is None or t['close'] < t['ma_long']:
            continue
        if not _is_liquid(t):
            continue

        # A9：訊號強度評分
        base = short_rise_pct * 100 * 0.6 + margin_drop_pct * 100 * 0.4
        score = base
        bonus_factors = []
        if short_to_margin_ratio >= 0.10:
            score += base * 0.5
            bonus_factors.append(f'券資比{short_to_margin_ratio*100:.1f}%')
        net5 = float(recent['三大法人買賣超'].sum())
        if net5 >= 5000:
            score *= 1.2
            bonus_factors.append(f'法人累買{int(net5)}張')
        if score >= 12:
            grade = '⭐⭐⭐ 強'
        elif score >= 5:
            grade = '⭐⭐ 中'
        else:
            grade = '⭐ 弱'

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略8',
            '訊號強度': grade,
            '評分': round(score, 1),
            '融資減幅%': round(margin_drop_pct * 100, 1),
            '融券增加%': round(short_rise_pct * 100, 1),
            '券資比%': round(short_to_margin_ratio * 100, 1),
            '加分因子': ','.join(bonus_factors) if bonus_factors else '—',
            '融資減少張數': margin_drop_lots,
            '融券增加張數': short_rise_lots,
            '訊號': '+'.join(signals),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
            'RSI': round(t['rsi'], 1) if t['rsi'] else None,
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('評分', ascending=False, inplace=True)
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


# ===================== 策略 14：RS Line 領漲（先創新高） =====================

def _compute_rs_line_60d_high_flag(t, market_hist):
    """S14：相對強度線 RS_line[t] = 個股_close / TAIEX_close。
    判斷今日 RS_line 是否創 60 日新高（個股相對強勢領漲於股價）。"""
    s_hist = t.get('close_history_30') or []
    if not market_hist or len(s_hist) < 30 or len(market_hist) < 30:
        return False, None
    n = min(len(s_hist), len(market_hist))
    s = np.array(s_hist[-n:], dtype=float)
    m = np.array(market_hist[-n:], dtype=float)
    if (s <= 0).any() or (m <= 0).any():
        return False, None
    rs_line = s / m
    if not np.isfinite(rs_line).all():
        return False, None
    cur = float(rs_line[-1])
    high60 = float(rs_line.max())
    return cur >= high60 * 0.999, cur


def strategy14(all_df):
    """S14：RS Line 創 60 日新高，但股價還沒到 52 週高 5~15%（領先機會）。
    多頭排列 + 法人買超 + 流動性。"""
    market = get_taiex_state()
    market_hist = market.get('close_history_30') or []
    if len(market_hist) < 30:
        return pd.DataFrame()
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if not (recent['三大法人買賣超'] > 0).any():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_long') is None:
            continue
        if not _is_liquid(t):
            continue
        # MA 多頭排列且 MA50/MA60 上揚
        if t['ma_short'] <= t['ma_long']:
            continue
        if t.get('ma_60_slope') is not None and t['ma_60_slope'] < -0.005:
            continue
        # 距 52 週高 5~15%（還沒衝太高）
        dist = t.get('dist_from_52w_high')
        if dist is None or dist < 0.05 or dist > 0.20:
            continue
        # RS line 創 60 日新高
        is_rs_high, rs_val = _compute_rs_line_60d_high_flag(t, market_hist)
        if not is_rs_high:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略14',
            'RS Line': round(rs_val, 4) if rs_val else '-',
            '距52週高%': f"{dist*100:.1f}%",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
            'MA60斜率': f"{(t.get('ma_60_slope') or 0)*100:.1f}%",
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('距52週高%', ascending=True, inplace=True)
    return df


# ===================== 策略 15：量縮回測 MA20/MA50 =====================

def strategy15(all_df):
    """S15：多頭排列 + 30 日內曾突破 + 回測 MA20/MA50 ±2%（量縮）+ 紅 K 站回。
    抓「整理回測後再啟動」的高勝率切點。"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        # 法人不要求嚴格買超（量縮回測常見法人持平），但要求未連續賣超
        if (recent['三大法人買賣超'] < 0).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        # 多頭排列：MA5 > MA10、MA20 上揚
        if t.get('ma_short') is None or t.get('ma_long') is None:
            continue
        if t['ma_short'] <= t['ma_long']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] < 0:
            continue
        # 回測 MA20 ±2% 或 MA60 ±2%（任一）
        ma20 = t['ma_20']; ma60 = t.get('ma_60')
        close = t['close']
        near_ma20 = abs(close - ma20) / ma20 <= 0.02 if ma20 else False
        near_ma60 = (ma60 is not None and abs(close - ma60) / ma60 <= 0.02)
        if not (near_ma20 or near_ma60):
            continue
        # 量縮：5 日均量 < 20 日均量 × 0.85
        vr5 = t.get('volume_ratio_5d')
        if vr5 is None or vr5 > 0.85:
            continue
        # 今日紅 K（close > prev_close）— 視為站回訊號
        prev = t.get('prev_close')
        if prev is None or close <= prev:
            continue
        which = 'MA20' if near_ma20 else 'MA60'
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略15',
            '回測位置': which,
            '量縮比': f"{vr5:.2f}",
            '最新收盤': close, 'MA 狀態': t['ma_status'],
            'MA20斜率': f"{(t.get('ma_20_slope') or 0)*100:.1f}%",
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('量縮比', ascending=True, inplace=True)
    return df


# ===================== 策略 16：季線（MA60）首次站上 + 量增 =====================

def strategy16(all_df):
    """S16：close 首次站上 MA60（前 60 日多在下方）+ MA60 由下彎轉平/上揚 + 量增 1.8x。
    抓「中期翻多」起點，常見大行情前兆。"""
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        # 至少今日法人未連續大幅賣超
        if (recent['三大法人買賣超'] < -1000).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue
        # 站上 MA60
        if t['close'] < t['ma_60'] * 1.005:  # 至少站上 0.5% 以上
            continue
        # 連續站上天數 1~5（首次站上才算）
        days_above = t.get('days_above_ma60')
        if days_above is None or days_above > 5:
            continue
        # MA60 斜率：不再加速下彎（≥ -1%）
        ma60_slope = t.get('ma_60_slope')
        if ma60_slope is None or ma60_slope < -0.01:
            continue
        # 量增 ≥ 1.8x
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.8:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略16',
            '站上MA60天': days_above,
            'MA60斜率': f"{ma60_slope*100:.1f}%",
            '量比': f"{vr:.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('量比', ascending=False, inplace=True)
    return df


# ===================== 策略 17：主力建倉代理（法人連買 + 量能 + 占比）=====================

def strategy17(all_df):
    """S17：主力建倉訊號（無分點資料情境下的代理）。
    入選條件：
      - 三大法人連買 ≥ 5 日（持續吸籌）
      - 今日成交量 > 20 日均量 × 1.5（量能放大代表非單純散戶）
      - 今日法人淨買金額 / 20日均量金額 ≥ 8%（法人占當日成交占比高 → 主力同向）
      - 收盤站上 MA20、MA20 上揚（不在下跌段建倉）
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(5)
        if len(recent) < 5:
            continue
        if not (recent['三大法人買賣超'] > 0).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] < 0:
            continue
        # 量能放大
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.5:
            continue
        # 法人淨買量 vs 20日均量
        avg_vol = t.get('avg_volume_20d')  # 單位：張
        net_inst = float(recent.iloc[0].get('三大法人買賣超', 0))  # 單位：張
        if avg_vol is None or avg_vol <= 0:
            continue
        inst_ratio = net_inst / avg_vol
        if inst_ratio < 0.08:
            continue
        # 5 日累計法人淨買 / 5×日均量
        net5 = float(recent['三大法人買賣超'].sum())
        cum_ratio = net5 / (avg_vol * 5)
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略17',
            '法人連買日': 5,
            '單日法人占比': f"{inst_ratio*100:.1f}%",
            '5日累計占比': f"{cum_ratio*100:.1f}%",
            '量比': f"{vr:.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        # 排序：5日累計占比 → 單日占比
        df['_cum'] = df['5日累計占比'].str.rstrip('%').astype(float)
        df['_day'] = df['單日法人占比'].str.rstrip('%').astype(float)
        df.sort_values(['_cum', '_day'], ascending=[False, False], inplace=True)
        df.drop(columns=['_cum', '_day'], inplace=True)
    return df


# ===================== 策略 18：中期盤整突破（Tight Range Breakout）=====================

def strategy18(all_df):
    """S18：30 日盤整 + 突破上沿 + ATR 擴張。
    入選條件：
      - 過去 30 日 (high-low)/low ≤ 12%（中期盤整）
      - 今日 close 突破過去 30 日 high × 1.005
      - 量比 ≥ 1.5
      - ATR 擴張比 (近 5d ATR / 前 20d ATR) ≥ 1.2（波動度開始擴張）
      - MA60 上揚 + 法人不連續賣超
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -1500).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None:
            continue
        if not _is_liquid(t):
            continue
        rng = t.get('range_30d_pct')
        if rng is None or rng > 0.12:
            continue
        h30 = t.get('high_30d')
        if h30 is None or t['close'] < h30 * 1.005:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.5:
            continue
        atr_exp = t.get('atr_expansion_ratio')
        if atr_exp is None or atr_exp < 1.2:
            continue
        ma60_sl = t.get('ma_60_slope')
        if ma60_sl is not None and ma60_sl < -0.005:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略18',
            '30日振幅': f"{rng*100:.1f}%",
            '突破點': round(h30, 2),
            '量比': f"{vr:.1f}x",
            'ATR擴張': f"{atr_exp:.2f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_atr'] = df['ATR擴張'].str.rstrip('x').astype(float)
        df.sort_values('_atr', ascending=False, inplace=True)
        df.drop(columns=['_atr'], inplace=True)
    return df


# ===================== 策略 19：Pocket Pivot =====================

def strategy19(all_df):
    """S19：Pocket Pivot（O'Neil/Morales 經典）。
    在上升趨勢中（close > MA50）且當日為上漲日，
    若當日成交量 > 過去 10 個交易日中所有「下跌日」的最大量，視為機構建倉訊號。
    比一般 BO 更早識別出趨勢中的「sneaky 進場」。

    入選條件：
      - close > MA20 且 MA20 > MA60（多頭排列）
      - 今日為上漲日
      - 今日量 > 過去 10 日中所有下跌日的最大量
      - RSI < 75（避免過熱追高）
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -2000).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20'] or t['ma_20'] < t['ma_60']:
            continue
        if t.get('rsi') is not None and t['rsi'] >= 75:
            continue

        vols = t.get('volume_history_15') or []
        pcs = t.get('price_change_history_15') or []
        if len(vols) < 11 or len(pcs) < 11:
            continue
        # 取最後 11 筆 → 前 10 筆為「過去 10 日」、最後一筆為今日
        v10 = vols[-11:-1]
        pc10 = pcs[-11:-1]
        today_vol = vols[-1]
        today_pc = pcs[-1]
        if today_pc <= 0:
            continue
        down_vols = [v for v, p in zip(v10, pc10) if p < 0]
        if not down_vols:
            # 過去 10 日無下跌日 → 條件改為「>= 過去 10 日量平均 1.2 倍」
            avg10 = sum(v10) / len(v10) if v10 else 0
            if today_vol < avg10 * 1.2:
                continue
            multiple = today_vol / avg10 if avg10 > 0 else 0
            tag = '無下跌日（量增 1.2x）'
        else:
            max_down_vol = max(down_vols)
            if today_vol <= max_down_vol:
                continue
            multiple = today_vol / max_down_vol if max_down_vol > 0 else 0
            tag = f'{len(down_vols)} 日下跌量低點'

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略19',
            '今日漲幅': f"{today_pc*100:+.1f}%",
            '量能倍數': f"{multiple:.2f}x",
            '參考': tag,
            '量比': f"{(t.get('volume_ratio') or 0):.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_m'] = df['量能倍數'].str.rstrip('x').astype(float)
        df.sort_values('_m', ascending=False, inplace=True)
        df.drop(columns=['_m'], inplace=True)
    return df


# ===================== 策略 20：營收驚喜後進（PEAD Drift） =====================

def strategy20(all_df, revenue_data=None):
    """S20：Post-Earnings/Revenue Announcement Drift。
    台股月營收約 10 號公布。S20 抓「公布後仍處低位但動能持續」的標的：
      - 營收年增率 ≥ 30%（強驚喜）且月增率 > 0
      - 過去 5 日累計漲幅 < 12%（尚未追高）
      - 收盤站上 MA20、5 日高附近（≥ 5 日高 × 0.97，動能還在）
      - 法人不連續賣超 + 量比 > 1.0
      - RSI < 70（避免追高過熱）
    """
    if revenue_data is None or revenue_data.empty:
        return pd.DataFrame()
    if '營收年增率' not in revenue_data.columns:
        return pd.DataFrame()

    grouped = all_df.groupby('證券代號')
    rev_lookup = revenue_data.set_index('證券代號')
    results = []
    for stock_id, group in grouped:
        if stock_id not in rev_lookup.index:
            continue
        rev_row = rev_lookup.loc[stock_id]
        if isinstance(rev_row, pd.DataFrame):
            rev_row = rev_row.iloc[0]
        yoy = rev_row.get('營收年增率')
        mom = rev_row.get('營收月增率')
        if yoy is None or pd.isna(yoy) or yoy < 0.30:
            continue
        if mom is None or pd.isna(mom) or mom <= 0:
            continue

        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -1500).all():
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue

        # 過去 5 日累計漲幅 < 12%（尚未追高）
        hist = t.get('close_history_30') or []
        if len(hist) < 6:
            continue
        try:
            ret_5d = (hist[-1] - hist[-6]) / hist[-6]
        except Exception:
            continue
        if ret_5d >= 0.12 or ret_5d < -0.05:
            continue  # 太高（追高）或太低（趨勢已破）

        # 接近 5 日高
        recent_high = max(hist[-5:])
        if t['close'] < recent_high * 0.97:
            continue

        vr = t.get('volume_ratio')
        if vr is None or vr < 1.0:
            continue
        rsi = t.get('rsi')
        if rsi is not None and rsi >= 70:
            continue

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略20',
            '營收YoY': f"{yoy*100:+.1f}%",
            '營收MoM': f"{mom*100:+.1f}%",
            '5日漲幅': f"{ret_5d*100:+.1f}%",
            '量比': f"{vr:.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_yoy'] = df['營收YoY'].str.rstrip('%').astype(float)
        df.sort_values('_yoy', ascending=False, inplace=True)
        df.drop(columns=['_yoy'], inplace=True)
    return df


# ===================== 策略 28：Cup with Handle（杯柄型態） =====================

def strategy28(all_df):
    """S28：經典 Cup with Handle 型態（O'Neil CANSLIM 核心型態）。
    定義：
      - 杯：60 日內形成 U 型 — 左緣高(LH)、底部低(LO)、右緣高(RH) ≈ LH（誤差 ≤ 6%）
      - 杯深 (LH - LO) / LH ∈ [0.12, 0.35]（合理深度，太淺像橫盤，太深像下跌）
      - 杯時長 ≥ 30 日
      - 把手：右緣後 5~15 日的小回測，回檔 ≤ 杯深 × 1/3，未跌破 50% 杯位
      - 今日 close > 把手最高 × 1.005（突破把手）
      - 量比 ≥ 1.5（突破量能確認）
      - close > MA20、MA60 上揚
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -2000).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_60_slope') is None or t['ma_60_slope'] <= -0.005:
            continue
        hist = t.get('close_history_30') or []
        if len(hist) < 50:
            continue
        try:
            arr = np.array(hist[-50:], dtype=float)
            if (arr <= 0).any():
                continue
        except Exception:
            continue
        # arr[-1] = today；handle 為「今日之前」的最後 handle_len 日
        # 動態尋找 handle 長度（5~12 日）
        best = None
        today = float(arr[-1])
        for handle_len in range(5, 13):
            # cup = arr[:-handle_len-1]；handle = arr[-handle_len-1:-1]
            cup = arr[: -(handle_len + 1)]
            handle = arr[-(handle_len + 1): -1]
            if len(cup) < 25:
                continue
            lh = float(cup[:5].max())     # 左緣 5 日內最高
            rh = float(cup[-5:].max())    # 右緣 5 日內最高
            lo = float(cup.min())
            if lh <= 0 or rh <= 0 or lo <= 0:
                continue
            if abs(rh - lh) / lh > 0.06:
                continue
            cup_depth = (max(lh, rh) - lo) / max(lh, rh)
            if not (0.12 <= cup_depth <= 0.35):
                continue
            handle_high = float(handle.max())
            handle_low = float(handle.min())
            handle_drop = (handle_high - handle_low) / handle_high if handle_high > 0 else 1
            if handle_drop > cup_depth / 3:
                continue
            mid_cup = (max(lh, rh) + lo) / 2
            if handle_low < mid_cup:
                continue
            if today < handle_high * 1.005:
                continue                  # 今日必須突破把手
            score = cup_depth * 100 - handle_drop * 100
            best = {
                '杯深': cup_depth,
                '杯時長': len(cup),
                '把手長': handle_len,
                '把手高': round(handle_high, 2),
                'score': score,
            }
            break

        if best is None:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.5:
            continue

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略28',
            '杯深': f"{best['杯深']*100:.1f}%",
            '杯時長': best['杯時長'],
            '把手長': best['把手長'],
            '把手高(突破點)': best['把手高'],
            '量比': f"{vr:.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_d'] = df['杯深'].str.rstrip('%').astype(float)
        df.sort_values('_d', ascending=False, inplace=True)
        df.drop(columns=['_d'], inplace=True)
    return df


# ===================== 策略 29：Pivot Breakout =====================

def strategy29(all_df):
    """S29：Pivot Breakout（樞紐高點 + 量價突破）。
    定義：以最近 60 日找出 pivot high（左右 5 日皆低於該點），形成壓力線；
    今日 close 突破最近最高 pivot × 1.005，並符合：
      - 至少 2 個有效 pivot
      - close > MA20、MA20 上揚
      - 量比 ≥ 1.6、紅 K
      - RSI < 78
      - 距 60 日低 ≥ 12%
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -2000).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] <= 0:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi >= 78:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.6:
            continue
        topen = t.get('today_open')
        if topen is not None and t['close'] <= topen:
            continue
        low_60 = t.get('low_60d')
        if low_60 is not None and low_60 > 0:
            if (t['close'] - low_60) / low_60 < 0.12:
                continue

        hist = t.get('close_history_30') or []
        if len(hist) < 25:
            continue
        try:
            arr = np.array(hist[-60:], dtype=float)
        except Exception:
            continue
        if len(arr) < 15:
            continue
        pivots = []
        for i in range(5, len(arr) - 6):
            wl = arr[i - 5:i]
            wr = arr[i + 1:i + 6]
            if len(wl) < 5 or len(wr) < 5:
                continue
            if arr[i] > wl.max() and arr[i] > wr.max():
                pivots.append((i, float(arr[i])))
        if len(pivots) < 2:
            continue
        pivots.sort(key=lambda x: x[0])
        recent_pivots = pivots[-3:]
        resistance = max(p[1] for p in recent_pivots)
        today = float(arr[-1])
        if today < resistance * 1.005:
            continue
        if len(arr) >= 7 and today < float(arr[-6:-1].max()) * 1.002:
            continue

        breakout_pct = (today - resistance) / resistance
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略29',
            '壓力線': round(resistance, 2),
            '突破幅度': f"{breakout_pct * 100:.2f}%",
            'Pivot 數': len(recent_pivots),
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_b'] = df['突破幅度'].str.rstrip('%').astype(float)
        df.sort_values(['_b', '量比'], ascending=[False, False], inplace=True)
        df.drop(columns=['_b'], inplace=True)
    return df


# ===================== 策略 40：MA20/MA60 黃金交叉 + 量能共振 =====================

def strategy40(all_df):
    """S40：經典中長線高勝率訊號 — MA20 上穿 MA60 黃金交叉。
    定義：
      - MA20 / MA60 黃金交叉發生於最近 5 個交易日內
      - close > MA20 > MA60（多頭排列確認）
      - MA60 斜率 > 0（中長期趨勢轉多）
      - 量比 ≥ 1.3（量能放大確認）
      - RSI 介於 40 ~ 75（避免極端）
      - 法人最近 3 日不全為負
      - 不在 60 日高 + 20% 之內（避免追頂）
    觸發後：典型趨勢起漲第一棒。
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent3 = group.head(3)
        if len(recent3) < 3:
            continue
        if (recent3['三大法人買賣超'] < 0).all():
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue

        days_ago = t.get('ma20_60_cross_days_ago')
        if days_ago is None or days_ago > 5:
            continue
        if not (t['close'] > t['ma_20'] > t['ma_60']):
            continue
        if t.get('ma_60_slope') is None or t['ma_60_slope'] <= 0:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi < 40 or rsi > 75:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.3:
            continue
        # 避免追頂：close 不能距 52 週高過近（10% 內視為已過熱）
        h52 = t.get('high_52w')
        if h52 and t['close'] >= h52 * 0.95:
            continue

        cross_label = '今日黃金交叉' if days_ago == 0 else f'{days_ago}日前金叉'

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略40',
            'MA20': round(t['ma_20'], 2),
            'MA60': round(t['ma_60'], 2),
            '黃金交叉': cross_label,
            'MA60斜率': f"{t['ma_60_slope']*100:+.1f}%",
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        # 越靠近今日交叉的越優先
        df['_d'] = df['黃金交叉'].apply(
            lambda s: 0 if s == '今日黃金交叉' else int(s.replace('日前金叉', '')))
        df.sort_values(['_d', 'MA60斜率'], ascending=[True, False], inplace=True)
        df.drop(columns=['_d'], inplace=True)
    return df


# ===================== 策略 38：跌破 MA20 後反吃（短打反轉） =====================

def strategy38(all_df):
    """S38：跌破 MA20 後迅速反吃 — 假跌破 + 強勢反彈。
    定義：
      - 過去 5 日內曾出現「close < MA20」（假跌破）
      - 今日 close ≥ MA20 × 1.005（站回 MA20 之上）
      - 今日紅 K + 量比 ≥ 1.5
      - MA20 仍上揚或與 5 日前相比未明顯走弱（slope > -0.005）
      - close > MA60（中期未失守）
      - RSI 介於 35~70（避免極端）
      - 法人最近 2 日不全為負
    觸發後：洗盤完成後續攻擊。
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent2 = group.head(2)
        if len(recent2) < 2:
            continue
        if (recent2['三大法人買賣超'] < 0).all():
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_60']:
            continue
        if t['close'] < t['ma_20'] * 1.005:
            continue
        if t.get('ma_20_slope') is not None and t['ma_20_slope'] < -0.005:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi < 35 or rsi > 70:
            continue
        topen = t.get('today_open')
        if topen is not None and t['close'] <= topen:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.5:
            continue

        # 必須過去 5 日內曾跌破 MA20（假跌破）
        # 用 close_history_30 的最後 6 個（含今日）+ ma_20 比較
        hist = t.get('close_history_30') or []
        if len(hist) < 6:
            continue
        try:
            last5 = np.array(hist[-6:-1], dtype=float)   # 不含今日
        except Exception:
            continue
        if (last5 <= 0).any():
            continue
        # 比較對象：MA20（用今天的 ma_20 估算前幾日大致水平 — 用 today close 不對）
        # 簡化：若最低 5 日 close < ma_20 × 0.99 → 假跌破
        if float(last5.min()) >= t['ma_20'] * 0.99:
            continue

        recover_pct = (t['close'] - float(last5.min())) / float(last5.min()) * 100

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略38',
            'MA20': round(t['ma_20'], 2),
            '近5日低': round(float(last5.min()), 2),
            '反彈幅度': f"+{recover_pct:.1f}%",
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_r'] = df['反彈幅度'].str.lstrip('+').str.rstrip('%').astype(float)
        df.sort_values('_r', ascending=False, inplace=True)
        df.drop(columns=['_r'], inplace=True)
    return df


# ===================== 策略 39：外資 + 投信「皆」連 2 日買超（強共識） =====================

def strategy39(all_df):
    """S39：強共識 — 外資、投信兩大法人「皆」最近 2 日連續買超。
    比 S3（三大法人共識）更嚴格（自營商不參與計算）：
      - 外資最近 2 日皆為買超（每日 ≥ 100 張）
      - 投信最近 2 日皆為買超（每日 ≥ 100 張）
      - 2 日累計：外資 ≥ 1000 張、投信 ≥ 300 張
      - close > MA20 + MA20 上揚
      - RSI < 78
      - 量比 ≥ 1.0
    觸發後：強共識，後勢看好。
    """
    if not all(c in all_df.columns for c in ('外資買賣超', '投信買賣超')):
        return pd.DataFrame()
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent2 = group.head(2)
        if len(recent2) < 2:
            continue
        if not (recent2['外資買賣超'] >= 100).all():
            continue
        if not (recent2['投信買賣超'] >= 100).all():
            continue
        foreign_2 = float(recent2['外資買賣超'].sum())
        invest_2 = float(recent2['投信買賣超'].sum())
        if foreign_2 < 1000 or invest_2 < 300:
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] <= 0:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi >= 78:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.0:
            continue

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略39',
            '外資2日累計': int(foreign_2),
            '投信2日累計': int(invest_2),
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_total'] = df['外資2日累計'] + df['投信2日累計']
        df.sort_values('_total', ascending=False, inplace=True)
        df.drop(columns=['_total'], inplace=True)
    return df


# ===================== 策略 36：連跌量縮 + 法人偷買（吸籌型） =====================

def strategy36(all_df):
    """S36：吸籌型 — 散戶賣壓出盡 + 法人逆勢偷買。
    定義：
      - 過去 10 日股價累積 −5% 至 −20%（已下跌但不到崩跌）
      - 近 5 日均量 < 過去 20 日均量的 80%（量縮，散戶停損完成）
      - 法人最近 5 日累計買超 ≥ 0（沒砍）且最近 3 日內至少 2 日買超
      - close > MA60 × 0.95（中期未失守）
      - close > MA5（短期止跌）
      - RSI 介於 30~55（超賣後回升中）
    觸發後：底部吸籌，未來上漲時可獲取大波段。
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent5 = group.head(5)
        recent3 = group.head(3)
        if len(recent5) < 5:
            continue

        net5 = float(recent5['三大法人買賣超'].sum())
        if net5 < 0:
            continue
        buy3_count = int((recent3['三大法人買賣超'] > 0).sum())
        if buy3_count < 2:
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_5') is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_5']:
            continue
        if t['close'] < t['ma_60'] * 0.95:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi < 30 or rsi > 55:
            continue

        # 10 日累積跌幅
        hist = t.get('close_history_30') or []
        if len(hist) < 12:
            continue
        try:
            arr10 = np.array(hist[-11:], dtype=float)
        except Exception:
            continue
        if (arr10 <= 0).any():
            continue
        cum = (arr10[-1] - arr10[0]) / arr10[0]
        if cum > -0.05 or cum < -0.20:
            continue

        # 量縮：5 日均量 / 20 日均量
        vol_hist = t.get('volume_history_15') or []
        if len(vol_hist) < 5:
            continue
        try:
            vol_recent5 = float(np.mean(vol_hist[-5:]))
        except Exception:
            continue
        avg20 = t.get('20d_avg_volume') or 0
        if avg20 <= 0 or vol_recent5 / avg20 > 0.8:
            continue

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略36',
            '10日跌幅': f"{cum * 100:.1f}%",
            '5日量比': f"{vol_recent5 / avg20 * 100:.0f}%",
            '法人5日累計': int(net5),
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('法人5日累計', ascending=False, inplace=True)
    return df


# ===================== 策略 37：大盤盤整 + 個股 RS 領頭 =====================

def strategy37(all_df):
    """S37：rotation leader — 大盤橫盤但個股相對強度突出。
    定義：
      - 大盤過去 20 日報酬絕對值 < 3%（盤整）
      - 個股 rs_vs_taiex ≥ 8%（明顯領先）
      - 個股 20 日報酬 ≥ 5%
      - close > MA20、MA60，MA20 上揚
      - 量比 ≥ 1.2（仍有買盤）
      - RSI < 75（有空間）
      - 法人未連賣（最近 3 日不全為負）
    觸發後：強者恆強，盤整中往往是 rotation 領頭羊。
    """
    market = get_taiex_state()
    m_ret = market.get('return_20d')
    if m_ret is None:
        return pd.DataFrame()
    if abs(m_ret) >= 0.03:
        return pd.DataFrame()    # 大盤非盤整，不啟動

    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent3 = group.head(3)
        if len(recent3) < 3:
            continue
        if (recent3['三大法人買賣超'] < 0).all():
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20'] or t['close'] < t['ma_60']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] <= 0:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi >= 75:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.2:
            continue

        rs = t.get('rs_vs_taiex')
        if rs is None or rs < 0.08:
            continue
        s_ret = t.get('return_20d')
        if s_ret is None or s_ret < 0.05:
            continue

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略37',
            'RS vs 大盤': f"+{rs * 100:.1f}%",
            '20日報酬': f"+{s_ret * 100:.1f}%",
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_rs'] = df['RS vs 大盤'].str.lstrip('+').str.rstrip('%').astype(float)
        df.sort_values('_rs', ascending=False, inplace=True)
        df.drop(columns=['_rs'], inplace=True)
    return df


# ===================== 策略 34：軋空動能（突破型） =====================

def strategy34(all_df, margin_df):
    """S34：軋空動能 — 法人連買 + 融券居高 + 量價突破。
    定義：
      - 三大法人最近 3 日皆為買超（共識）
      - 券資比 ≥ 8%（軋空火藥充足）
      - 融券餘額 5 日內未明顯減少（≥ 90% 維持，未提前回補）
      - 今日 close 創 20 日新高 + 紅 K + 量比 ≥ 1.8
      - close > MA20、MA20 上揚
      - RSI < 80（避免追高）
    觸發後：上漲彈性極強（被軋者被迫回補 → 短期暴衝）。
    """
    if margin_df is None or margin_df.empty:
        return pd.DataFrame()
    grouped_inst = all_df.groupby('證券代號')
    grouped_margin = margin_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped_inst:
        recent = group.head(3)
        if len(recent) < 3 or not (recent['三大法人買賣超'] > 0).all():
            continue
        if stock_id not in grouped_margin.groups:
            continue
        mg = grouped_margin.get_group(stock_id).head(5)
        if len(mg) < 2 or '融券餘額' not in mg.columns or '融資餘額' not in mg.columns:
            continue
        shorts = mg['融券餘額'].tolist()
        margins = mg['融資餘額'].tolist()
        if not shorts or shorts[-1] <= 0 or margins[-1] <= 0:
            continue
        # 券資比（最新一日）
        short_margin_ratio = shorts[0] / margins[0] if margins[0] > 0 else 0
        if short_margin_ratio < 0.08:
            continue
        # 融券未明顯減少（軋空者尚未撤出）
        # 注意：head(5) 是降序，shorts[0] 最新，shorts[-1] 最舊
        if shorts[0] < shorts[-1] * 0.9:
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] <= 0:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi >= 80:
            continue
        topen = t.get('today_open')
        if topen is not None and t['close'] <= topen:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.8:
            continue
        # 20 日新高（用 high_30d 近似，更嚴格用 close > high_30d）
        high30 = t.get('high_30d')
        if high30 is None or t['close'] < high30 * 1.001:
            continue

        net3 = int(recent['三大法人買賣超'].sum())
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略34',
            '券資比': f"{short_margin_ratio * 100:.1f}%",
            '融券餘額': int(shorts[0]),
            '法人3日累買': net3,
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_r'] = df['券資比'].str.rstrip('%').astype(float)
        df.sort_values(['_r', '法人3日累買'], ascending=[False, False], inplace=True)
        df.drop(columns=['_r'], inplace=True)
    return df


# ===================== 策略 33：V-shape 反轉 =====================

def strategy33(all_df):
    """S33：V-shape 反轉 — 連續下跌後出現紅 K + 量爆。
    定義：
      - 近 5 日內至少 3 日為下跌（close[i] < close[i-1]）
      - 5 日累積跌幅 ≥ 5%（避免小波動誤觸）
      - 今日紅 K（close > open），且 close > 昨收 ≥ 1.5%
      - 量比 ≥ 1.8（成交量明顯放大）
      - 收回 MA5 之上（突破短期趨勢）
      - close > MA60（中期仍多頭）— 過濾下降趨勢中的死貓彈
      - RSI 在 25~55 區間（不過度超賣亦非偏多頭）
      - 近 60 日低不在最低（避免抄真底失敗）
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -3000).all():
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_5') is None or t.get('ma_60') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_60']:
            continue
        if t['close'] < t['ma_5']:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi < 25 or rsi > 55:
            continue
        topen = t.get('today_open')
        if topen is not None and t['close'] <= topen:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.8:
            continue

        close_hist = t.get('close_history_30') or []
        if len(close_hist) < 7:
            continue
        try:
            arr = np.array(close_hist[-7:], dtype=float)
        except Exception:
            continue
        if (arr <= 0).any():
            continue

        # 近 5 日（不含今日）累積跌幅
        prev5 = arr[-6:-1]   # 5 個 close（前 5 日）
        # 連跌天數：count i where prev[i]>prev[i+1]
        down_days = sum(1 for i in range(len(prev5) - 1) if prev5[i + 1] < prev5[i])
        if down_days < 3:
            continue
        cum_drop = (prev5[-1] - prev5[0]) / prev5[0] if prev5[0] > 0 else 0
        if cum_drop > -0.05:
            continue

        # 今日漲幅 vs 昨收 ≥ 1.5%
        prev_close = float(arr[-2])
        today = float(arr[-1])
        if prev_close <= 0 or (today - prev_close) / prev_close < 0.015:
            continue

        # 不能在 60 日絕對最低
        low60 = t.get('low_60d')
        if low60 is not None and low60 > 0:
            if today / low60 < 1.02:
                continue

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略33',
            '5日跌幅': f"{cum_drop * 100:.1f}%",
            '今日漲幅': f"{(today - prev_close) / prev_close * 100:.2f}%",
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_v'] = df['量比'].str.rstrip('x').astype(float)
        df.sort_values('_v', ascending=False, inplace=True)
        df.drop(columns=['_v'], inplace=True)
    return df


# ===================== 策略 31：法人三連買 + 量能整理突破 =====================

def strategy31(all_df):
    """S31：短打型 — 法人連 3 日買超（持續性 OK）+ 近 5 日量能整理 + 今日突破。
    定義：
      - 三大法人最近 3 日皆為買超（每日淨買 ≥ 100 張）
      - 近 5 日 volume 標準差/均值 ≤ 0.35（量能收斂、整理）
      - 今日 close 創 5 日新高（突破整理）+ 紅 K
      - 今日 volume 比 5 日均量 ≥ 1.4
      - close > MA20，MA20 上揚（短期趨勢仍向上）
      - RSI < 75（避免追高）
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(3)
        if len(recent) < 3:
            continue
        if not (recent['三大法人買賣超'] >= 100).all():
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] <= 0:
            continue
        rsi = t.get('rsi')
        if rsi is None or rsi >= 75:
            continue
        topen = t.get('today_open')
        if topen is not None and t['close'] <= topen:
            continue

        vol_hist = t.get('volume_history_15') or []
        if len(vol_hist) < 5:
            continue
        try:
            v_recent = np.array(vol_hist[-5:], dtype=float)
        except Exception:
            continue
        if (v_recent <= 0).any():
            continue
        v_mean = float(v_recent.mean())
        v_std = float(v_recent.std())
        if v_mean <= 0:
            continue
        cv = v_std / v_mean
        if cv > 0.35:
            continue

        today_vol = float(v_recent[-1])
        prior_mean = float(v_recent[:-1].mean()) if len(v_recent) > 1 else v_mean
        if prior_mean <= 0 or today_vol / prior_mean < 1.4:
            continue

        # 今日創 5 日新高（突破整理區）
        close_hist = t.get('close_history_30') or []
        if len(close_hist) < 5:
            continue
        try:
            c5 = np.array(close_hist[-5:], dtype=float)
        except Exception:
            continue
        if t['close'] < float(c5[:-1].max()) * 1.002:
            continue

        # 法人 3 日累積
        net_3d = int(recent['三大法人買賣超'].sum())
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略31',
            '法人3日累積': net_3d,
            '量能CV': f"{cv * 100:.1f}%",
            '今日量比': f"{today_vol / prior_mean:.1f}x",
            'RSI': round(rsi, 1),
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('法人3日累積', ascending=False, inplace=True)
    return df


# ===================== 策略 27：融資斷頭反彈（A10） =====================

def strategy27(all_df, margin_df):
    """A10：融資快速減少（散戶被洗）+ 當天止跌反彈 → 籌碼浪費後的反彈進場機會。
    入選條件：
      - 5 日融資減幅 ≥ 8%（明顯斷頭）
      - 今日 close > 今日 open（紅 K 反彈）
      - 今日 close > MA5（短期止跌）
      - 量比 ≥ 1.2（有量承接）
      - 法人不連續賣超
      - close > 60 日低 × 1.02（不在絕對底，避免接落下刀）
      - RSI < 65（剛從底部反彈）
    經典「籌碼洗清後反彈」型訊號，常見於上升趨勢中的回測。
    """
    if margin_df is None or margin_df.empty:
        return pd.DataFrame()
    grouped_inst = all_df.groupby('證券代號')
    grouped_margin = margin_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped_inst:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -1500).all():
            continue
        if stock_id not in grouped_margin.groups:
            continue
        mg = grouped_margin.get_group(stock_id).head(5)
        if len(mg) < 5:
            continue
        balances = mg['融資餘額'].tolist()
        if not balances or balances[-1] <= 0:
            continue
        # 5 日融資減幅
        if balances[0] >= balances[-1]:
            continue
        margin_drop_pct = (balances[-1] - balances[0]) / balances[-1]
        if margin_drop_pct < 0.08:
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_short') is None:
            continue
        if not _is_liquid(t):
            continue
        op = t.get('today_open')
        if op is None or t['close'] <= op:
            continue   # 今天必須是紅 K
        if t['close'] < t['ma_short']:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.2:
            continue
        rsi = t.get('rsi')
        if rsi is not None and rsi >= 65:
            continue
        # 不在絕對底
        hist = t.get('close_history_30') or []
        if len(hist) >= 60:
            try:
                low60 = float(min(hist[-60:]))
                if low60 > 0 and t['close'] < low60 * 1.02:
                    continue
            except Exception:
                pass
        rebound_pct = (t['close'] - op) / op

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略27',
            '5日融資減幅%': round(margin_drop_pct * 100, 1),
            '今日反彈%': f"{rebound_pct*100:+.2f}%",
            '量比': f"{vr:.1f}x",
            'RSI': round(rsi, 1) if rsi else None,
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('5日融資減幅%', ascending=False, inplace=True)
    return df


# ===================== 策略 25：籌碼集中度（10 日法人結構分） =====================

def strategy25(all_df):
    """S25：用 10 日的法人連續性 + 佔量比，產生「籌碼結構分」(0~100)。
    與 S17（嚴格法人連買 5 日）不同，這個更彈性 — 抓的是「持續溫和買進」型的長線族群。

    分數構成（最高 100 分）：
      - 連續性：(買超日 - 賣超日) / 10 × 30  →  10 天全買 = 30 分
      - 強度：平均日法人淨買 / 20日均量 × 200（0.05 → 10 分）→ 上限 30 分
      - 累積方向：10 日累計法人淨買為正 +20 分
      - 量能溫和擴張：5 日均量 / 20 日均量 ∈ [1.05, 1.5] +20 分（不暴量也不縮）

    入選條件：
      - 結構分 ≥ 60
      - close > MA20、MA20 上揚
      - 法人累積 10 日淨買為正
      - RSI < 75
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(10)
        if len(recent) < 10:
            continue
        nets = recent['三大法人買賣超'].astype(float).tolist()
        n_buy = sum(1 for v in nets if v > 0)
        n_sell = sum(1 for v in nets if v < 0)
        cum10 = sum(nets)
        if cum10 <= 0:
            continue   # 累計賣超直接淘汰

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] <= 0:
            continue
        rsi = t.get('rsi')
        if rsi is not None and rsi >= 75:
            continue
        avg_vol = t.get('avg_volume_20d')
        vol_ratio_5d = t.get('volume_ratio_5d')
        if not avg_vol or avg_vol <= 0:
            continue

        # 評分
        consistency_score = (n_buy - n_sell) / 10 * 30
        avg_net = sum(nets) / 10
        strength_score = min(30, max(0, avg_net / avg_vol * 200))
        cumulative_score = 20 if cum10 > 0 else 0
        vol_score = 0
        if vol_ratio_5d is not None and 1.05 <= vol_ratio_5d <= 1.5:
            vol_score = 20
        total = round(consistency_score + strength_score + cumulative_score + vol_score, 1)
        if total < 60:
            continue

        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略25',
            '籌碼結構分': total,
            '10日買超日': n_buy,
            '10日賣超日': n_sell,
            '10日累計': int(cum10),
            '日均淨買占比': f"{avg_net/avg_vol*100:.2f}%",
            '5日量比': f"{vol_ratio_5d:.2f}x" if vol_ratio_5d else '—',
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values('籌碼結構分', ascending=False, inplace=True)
    return df


# ===================== 策略 26：三重底反轉 =====================

def strategy26(all_df):
    """S26：60 日內三次以上測低，且頸線突破。
    入選條件：
      - 過去 60 日有 ≥ 3 個 close 落在 (low_60d, low_60d × 1.04) 內 → 反覆測底
      - 今日 close > 過去 30 日最高 × 1.005（突破頸線）
      - 量比 ≥ 1.5
      - close > MA20（趨勢轉強）
      - 法人不連續賣超
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -2000).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        hist = t.get('close_history_30') or []
        if len(hist) < 60:
            continue
        try:
            arr = np.array(hist[-60:], dtype=float)
            if (arr <= 0).any():
                continue
        except Exception:
            continue
        low60 = float(arr.min())
        if low60 <= 0:
            continue
        # 過去 60 日有幾天落在低點 4% 範圍內
        low_band_count = int((arr <= low60 * 1.04).sum())
        if low_band_count < 3:
            continue
        # 突破：close > 30 日 high
        high_30d = t.get('high_30d')
        if high_30d is None or t['close'] < high_30d * 1.005:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.5:
            continue
        # 反彈幅度（從 60 日低算起）
        rebound_pct = (t['close'] - low60) / low60
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略26',
            '60日低': round(low60, 2),
            '測底次數': low_band_count,
            '頸線突破點': round(high_30d, 2),
            '自低點反彈': f"{rebound_pct*100:+.1f}%",
            '量比': f"{vr:.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_r'] = df['自低點反彈'].str.rstrip('%').astype(float)
        df.sort_values(['測底次數', '_r'], ascending=[False, False], inplace=True)
        df.drop(columns=['_r'], inplace=True)
    return df


# ===================== 策略 23：Inside Day Breakout =====================

def strategy23(all_df):
    """S23：連續內包日收斂 → 突破。
    入選條件：
      - 連續 ≥ 2 日內包日（today high < prev high AND today low > prev low）
      - 今日 close 突破內包期間最高（即今日 close > prev_high of inside-day chain start）
      - 量比 ≥ 1.5（突破量能確認）
      - close > MA20
      - 法人不連續賣超
    比 BB squeeze 抓更短期的盤整突破，常見於「整理 2-3 日就噴」的個股。
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -1500).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        ic = t.get('inside_days_count') or 0
        if ic < 2:
            continue
        # 突破：今日 close > 內包鏈最早一天的 prev_high（也就是 ic 天前的 high）
        # 簡化：用 high_30d 的近期高估計，但更精確用 prev_high 倒推
        # 這裡用 prev_high 作為「最近一次未被打破的高點」近似
        prev_high = t.get('prev_high')
        if prev_high is None or t['close'] < prev_high:
            continue
        vr = t.get('volume_ratio')
        if vr is None or vr < 1.5:
            continue
        rsi = t.get('rsi')
        if rsi is not None and rsi >= 80:
            continue
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略23',
            '內包天數': ic,
            '突破點': round(prev_high, 2),
            '量比': f"{vr:.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_v'] = df['量比'].str.rstrip('x').astype(float)
        df.sort_values(['內包天數', '_v'], ascending=[False, False], inplace=True)
        df.drop(columns=['_v'], inplace=True)
    return df


# ===================== 策略 24：Breakaway Gap（跳空缺口突破） =====================

def strategy24(all_df):
    """S24：跳空缺口 + 量爆 + 突破近期高點。
    入選條件：
      - 今日 open > 昨日 high × 1.005（明顯跳空）
      - 今日 close > 今日 open（沒被回補，跳空有效）
      - 量比 ≥ 2.0（跳空伴隨爆量）
      - 今日 close > 30 日 high（兼為新高突破）
      - close > MA20、法人不連續賣超
    通常代表「重大利多」進場，是非常強的趨勢起點。
    """
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(2)
        if len(recent) < 2:
            continue
        if (recent['三大法人買賣超'] < -2000).all():
            continue
        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        op = t.get('today_open'); ph = t.get('prev_high')
        if not op or not ph or op < ph * 1.005:
            continue
        if t['close'] <= op:
            continue   # 缺口被回補
        vr = t.get('volume_ratio')
        if vr is None or vr < 2.0:
            continue
        h30 = t.get('high_30d')
        if h30 is None or t['close'] <= h30:
            continue
        gap_pct = (op - ph) / ph
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略24',
            '跳空幅度': f"{gap_pct*100:+.1f}%",
            '突破30日高': round(h30, 2),
            '量比': f"{vr:.1f}x",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df['_g'] = df['跳空幅度'].str.rstrip('%').astype(float)
        df['_v'] = df['量比'].str.rstrip('x').astype(float)
        df.sort_values(['_g', '_v'], ascending=[False, False], inplace=True)
        df.drop(columns=['_g', '_v'], inplace=True)
    return df


# ===================== 策略 21：法人 + 營收 + RS 三聯 =====================

def strategy21(all_df, revenue_data=None):
    """S21：高品質「三聯共振」進場 — 籌碼 + 基本面 + 相對強度 同時亮燈。
    入選條件：
      - 三大法人連買 ≥ 5 日（籌碼）
      - 月營收 YoY ≥ 25% 且 累計 YoY ≥ 15%（基本面持續加速）
      - 個股 vs TAIEX 相對強度 ≥ 5%（領先大盤）
      - 收盤站上 MA20 且 MA20 上揚（趨勢正向）
      - RSI < 75（避免過熱追高）
    勝率與賠率俱佳的「核心持股級」訊號。
    """
    if revenue_data is None or revenue_data.empty:
        return pd.DataFrame()
    if '營收年增率' not in revenue_data.columns:
        return pd.DataFrame()

    rev_lookup = revenue_data.set_index('證券代號')
    grouped = all_df.groupby('證券代號')
    results = []
    for stock_id, group in grouped:
        recent = group.head(5)
        if len(recent) < 5:
            continue
        if not (recent['三大法人買賣超'] > 0).all():
            continue
        if stock_id not in rev_lookup.index:
            continue
        rev_row = rev_lookup.loc[stock_id]
        if isinstance(rev_row, pd.DataFrame):
            rev_row = rev_row.iloc[0]
        yoy = rev_row.get('營收年增率')
        cum_yoy = rev_row.get('累計年增率')
        if yoy is None or pd.isna(yoy) or yoy < 0.25:
            continue
        if cum_yoy is None or pd.isna(cum_yoy) or cum_yoy < 0.15:
            continue

        t = get_stock_technicals(stock_id)
        if t['close'] is None or t.get('ma_20') is None:
            continue
        if not _is_liquid(t):
            continue
        if t['close'] < t['ma_20']:
            continue
        if t.get('ma_20_slope') is None or t['ma_20_slope'] <= 0:
            continue
        rs = t.get('rs_vs_taiex')
        if rs is None or rs < 0.05:
            continue
        rsi = t.get('rsi')
        if rsi is not None and rsi >= 75:
            continue

        net5 = float(recent['三大法人買賣超'].sum())
        results.append({
            '證券代號': stock_id, '證券名稱': group['證券名稱'].iloc[0],
            '市場': group['市場'].iloc[0], '策略': '策略21',
            '法人連買日': 5,
            '5日法人累計': int(net5),
            '營收YoY': f"{yoy*100:+.1f}%",
            '累計YoY': f"{cum_yoy*100:+.1f}%",
            '相對強度': f"{rs*100:+.1f}%",
            '最新收盤': t['close'], 'MA 狀態': t['ma_status'],
        })
    df = pd.DataFrame(results)
    if not df.empty:
        # 三聯綜合分數：YoY + RS + 法人累計
        df['_yoy'] = df['營收YoY'].str.rstrip('%').astype(float)
        df['_rs'] = df['相對強度'].str.rstrip('%').astype(float)
        df['_score'] = df['_yoy'] * 0.5 + df['_rs'] * 1.0 + df['5日法人累計'] / 1000 * 0.3
        df.sort_values('_score', ascending=False, inplace=True)
        df.drop(columns=['_yoy', '_rs', '_score'], inplace=True)
    return df


def _estimate_target_price(t, deep_pred=None):
    """
    Tier A1：融合版目標價預估。

    - 若有 deep_pred（來自 GRU/Transformer 模型，dict 含 pred_20d / target_price / up_prob）
      採 **60% 模型 + 40% 技術面** 融合；up_prob > 0.6 時提高模型權重到 70%
    - 若無模型資料，退回原本的技術面加權（布林中軌 / 20日高點 / MA / RSI）

    回傳 (目標價, '漲幅%字串')。
    """
    close = t['close']
    if close is None or close <= 0:
        return None, None

    # ── 技術面 baseline ──
    tech_targets = []
    if t['bb_upper'] is not None and t['bb_lower'] is not None:
        bb_mid = (t['bb_upper'] + t['bb_lower']) / 2
        tech_targets.append(bb_mid)
    if t['prev_high_20d'] is not None and t['prev_high_20d'] > close:
        tech_targets.append(t['prev_high_20d'] * 0.85)
    if t['ma_long'] is not None and t['ma_long'] > close:
        tech_targets.append(t['ma_long'])
    if t['rsi'] is not None and t['rsi'] < 40:
        rsi_boost = 1 + (40 - t['rsi']) / 200
        tech_targets = [tgt * rsi_boost for tgt in tech_targets] if tech_targets else []

    tech_target = (sum(tech_targets) / len(tech_targets)) if tech_targets else None

    # ── 模型融合 ──
    if deep_pred and deep_pred.get('target_price') is not None:
        model_target = float(deep_pred['target_price'])
        up_prob = deep_pred.get('up_prob') or 0.5
        # 機率越高、越信模型
        w_model = 0.7 if up_prob > 0.6 else 0.6
        if tech_target is not None:
            target = round(model_target * w_model + tech_target * (1 - w_model), 2)
        else:
            target = round(model_target, 2)
    else:
        if tech_target is None:
            return None, None
        target = round(tech_target, 2)

    if target <= close:
        return None, None

    upside = round((target - close) / close * 100, 1)
    return target, f'+{upside}%'


# ===================== 潛力股提前佈局 =====================

def find_early_potential(all_df, deep_preds=None, up_prob_min=0.50,
                          breakout_preds=None):
    """
    找出「尚未發動但正在蓄勢」的潛力股。

    偵測五大提前訊號，每滿足一項 +1 分，至少 3 分列入：
    1. 量能蓄積：近 3 天量比逐日攀升但股價波動 < 2%（主力悄悄吃貨）
    2. 法人試探性買超：最近 1-2 天法人小量買超（尚未達連 3 天門檻）
    3. 布林極度收斂：帶寬百分位 < 20%，即將噴發
    4. 均線糾結即將突破：MA5 與 MA10 差距 < 1% 且 MA5 趨勢向上
    5. RSI 底部回升：RSI 從 < 35 回升中，尚未到 50（動能正在累積）

    Tier A 增強：
    - deep_preds: {sid: {pred_20d, up_prob, target_price, lower_price, upper_price}}
      若提供，加入「上漲機率 / 模型預測漲幅 / 信賴區間」欄位
      且預設以 up_prob ≥ 0.50 做**第二道篩網**（rule≥3 AND model 不看空）
    - breakout_preds: {sid: p_breakout} 來自 breakout_classifier 的 T+10 ≥ +10% 機率
    """
    grouped = all_df.groupby('證券代號')
    results = []
    deep_preds = deep_preds or {}
    breakout_preds = breakout_preds or {}

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
            sid_key = str(stock_id).strip()
            dp = deep_preds.get(sid_key)
            bp = breakout_preds.get(sid_key)

            # Tier A3 雙門檻：若模型看空（up_prob < up_prob_min）則剔除
            # 若沒有模型預測則單靠規則分數通過（避免流失）
            if dp is not None:
                up_v = dp.get('up_prob')
                # B9：同時處理 None / NaN
                if up_v is not None and not pd.isna(up_v) and float(up_v) < up_prob_min:
                    continue

            target_price, upside = _estimate_target_price(t, deep_pred=dp)
            row = {
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
            }
            # Tier A2: 模型加值欄位
            if dp is not None:
                if dp.get('up_prob') is not None:
                    row['上漲機率(20d)'] = f"{dp['up_prob']*100:.0f}%"
                if dp.get('pred_20d') is not None:
                    row['模型預測漲幅'] = f"{dp['pred_20d']*100:+.1f}%"
                # 信賴區間
                lo = dp.get('lower_price'); hi = dp.get('upper_price')
                if lo is not None and hi is not None:
                    row['95%區間'] = f"{lo:.2f} ~ {hi:.2f}"
            if bp is not None:
                row['突破機率(10d)'] = f"{bp*100:.0f}%"
            # 複合排名分：潛力分數 × 上漲機率 × (1 + 突破機率)，沒有則退回潛力分數
            comp = float(score) * 20
            if dp and dp.get('up_prob') is not None:
                comp *= (0.5 + dp['up_prob'])
            if bp is not None:
                comp *= (1 + bp)
            row['綜合評分'] = round(comp, 1)
            results.append(row)

    df = pd.DataFrame(results)
    if not df.empty:
        sort_col = '綜合評分' if '綜合評分' in df.columns else '潛力分數'
        df.sort_values(sort_col, ascending=False, inplace=True)
    return df


# ===================== 綜合買入潛力排名 =====================

def compute_composite_ranking(buy_dfs_map, all_df, top_n=30, backtest_scores=None):
    """跨策略加權評分，輸出最具買入潛力 TOP N。
    buy_dfs_map: dict {策略名稱: DataFrame}
    backtest_scores: 可選 dict {策略名稱: 綜效分數 0~1}，用於動態加權
    """
    # 策略類別分組（用於相關性懲罰：同類多命中不加分過多）
    strategy_groups = {
        '法人': {'策略1', '策略2', '策略3', '策略31', '策略39'},
        '量價': {'策略4', '策略6', '策略11'},
        '反彈': {'策略5', '策略10', '策略12', '策略33', '策略36', '策略38'},
        '基本面': {'策略7'},
        '籌碼': {'策略8', '策略34'},
        '技術面': {'策略9', '策略13', '策略37'},
        '中期突破': {'策略14', '策略15', '策略16', '策略18', '策略28', '策略40'},
        '主力': {'策略17', '策略19', '策略25'},
        '基本面': {'策略20', '策略21'},
        '短期突破': {'策略23', '策略24', '策略29'},
        '反轉': {'策略26', '策略27'},
    }

    # 策略權重（可由回測動態調整）
    default_weights = {
        '策略1': 15, '策略2': 15, '策略3': 18,
        '策略4': 14, '策略5': 12, '策略6': 13,
        '策略7': 18, '策略8': 12,
        '策略9': 12, '策略10': 10, '策略11': 14,
        '策略12': 10, '策略13': 14,
        '策略14': 16, '策略15': 13, '策略16': 17,
        '策略17': 18, '策略18': 16,
        '策略19': 17, '策略20': 18,
        '策略21': 22,    # 三聯共振 → 高權重
        '策略23': 16, '策略24': 19,
        '策略25': 17, '策略26': 17, '策略27': 14,
        '策略28': 21,    # CANSLIM 經典 → 高權重
        '策略29': 18,
        '策略31': 16,    # 短打型法人連買 + 整理突破
        '策略33': 14,    # 短打 V-shape 反轉
        '策略34': 19,    # 軋空動能（高彈性）
        '策略36': 17,    # 吸籌型（中長線）
        '策略37': 16,    # rotation leader（盤整中強者）
        '策略38': 13,    # 假跌破反吃（短打）
        '策略39': 20,    # 外資+投信「皆」連2日買超（強共識）
        '策略40': 20,    # MA20/60 黃金交叉（經典中長線高勝率）
    }
    weights = default_weights.copy()
    if backtest_scores:
        for sname, adj in backtest_scores.items():
            if sname in weights:
                weights[sname] = int(weights[sname] * (0.6 + 0.8 * float(adj)))

    # C13：依 OOS summary 動態加成（過去 6 個月 forward 10 日）
    oos_mults = _get_strategy_oos_multipliers()
    if oos_mults:
        adjusted = []
        for sname in list(weights.keys()):
            m = oos_mults.get(sname, 1.0)
            if abs(m - 1.0) > 0.01:
                weights[sname] = max(3, int(round(weights[sname] * m)))
                adjusted.append(f"{sname}×{m:.2f}")
        if adjusted:
            print(f"  · C13 策略 OOS 加成：{', '.join(adjusted[:6])}"
                  + (' ...' if len(adjusted) > 6 else ''))

    # C18：策略衰退偵測（最近 60D vs 180D 惡化）→ 再乘 0.5
    decay_mults = _get_strategy_decay_multipliers()
    if decay_mults:
        decayed_list = []
        for sname, m in decay_mults.items():
            if sname in weights:
                weights[sname] = max(2, int(round(weights[sname] * m)))
                decayed_list.append(sname)
        if decayed_list:
            print(f"  · C18 策略衰退降權 0.5x：{', '.join(decayed_list)}")

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


# ===================== Tier D3: IG 解釋性 =====================

def _compute_top_ig_explanations(ranking_df, shared_ohlcv_cache, top_n=5):
    """對 TOP N ranking 股票計算 Integrated Gradients，
    回傳 {sid: [(feature, contribution), ...]} 方便 HTML 展示。"""
    if ranking_df is None or ranking_df.empty or not shared_ohlcv_cache:
        return {}
    try:
        import deep_ranker as dr
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, ckpt = dr.load_model(device=device, return_ckpt=True)
        if model is None:
            return {}
        features = ckpt.get('features', dr.SEQ_FEATURES)
        sids = ranking_df['證券代號'].astype(str).str.strip().head(top_n).tolist()
        X, ok_sids, _d, _c = dr.build_live_windows(
            sids, ohlcv_cache=shared_ohlcv_cache,
            feature_cols=features,
        )
        if len(X) == 0:
            return {}
        out = {}
        for i, sid in enumerate(ok_sids):
            try:
                pairs = dr.explain_prediction(
                    model, X[i], features,
                    target='cls', device=device, top_k=6,
                )
                out[sid] = pairs
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"  ⚠ IG 解釋失敗：{e}")
        return {}


# ===================== Tier C: 情境特徵加成 =====================

def _compute_institutional_streak(all_df):
    """計算每檔「連續買超天數」— 從最新日往前找連續買超，遇到賣超就中斷。"""
    streak_map = {}
    if all_df is None or all_df.empty or '三大法人買賣超' not in all_df.columns:
        return streak_map
    for sid, g in all_df.groupby('證券代號'):
        g_sorted = g.sort_values('日期', ascending=False) if '日期' in g.columns else g
        streak = 0
        for v in g_sorted['三大法人買賣超']:
            if v is None or pd.isna(v) or v <= 0:
                break
            streak += 1
        if streak > 0:
            streak_map[str(sid).strip()] = streak
    return streak_map


def _compute_rs_acceleration(stock_ids):
    """Tier A1 修正版：用 close_history_30 與 TAIEX close_history 即時計算 RS 加速度。

    定義：rs[t] = (S_t / S_0) / (M_t / M_0)，其中 S 為個股、M 為大盤
        → rs > 1 表示個股相對大盤強
    加速度 = 最近 5 天 rs 斜率 − 前 5 天 rs 斜率（>0 表動能向上加速）。
    """
    accel_map: dict = {}
    market = get_taiex_state()
    m_hist = market.get('close_history_30') or []
    if len(m_hist) < 12:
        return accel_map
    m = np.array(m_hist, dtype=float)
    if m[0] <= 0 or not np.isfinite(m).all():
        return accel_map
    m_norm = m / m[0]

    for sid in stock_ids:
        sid_key = str(sid).strip()
        t = _technicals_cache.get(sid_key)
        if t is None:
            continue
        s_hist = t.get('close_history_30') or []
        # 對齊長度：取兩者尾端共同長度
        n = min(len(s_hist), len(m_hist))
        if n < 12:
            continue
        try:
            s = np.array(s_hist[-n:], dtype=float)
            mm = np.array(m_hist[-n:], dtype=float)
            if s[0] <= 0 or mm[0] <= 0:
                continue
            rs = (s / s[0]) / (mm / mm[0])
            if not np.isfinite(rs).all():
                continue
            recent = rs[-5:]
            baseline = rs[-10:-5]
            if len(recent) < 3 or len(baseline) < 3:
                continue
            slope_recent = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
            slope_base = (baseline[-1] - baseline[0]) / max(1, len(baseline) - 1)
            accel = slope_recent - slope_base
            accel_map[sid_key] = float(accel)
        except Exception:
            continue
    return accel_map


def _apply_context_boost(ranking_df, all_data, revenue_data,
                         score_col='綜合分數'):
    """Tier C: 以 法人連買天數 / 營收YoY / RS加速度 / 融資變化 調整分數。
    每項因子 → 歸一化 [0, 1] → boost 0~10 分；同時新增欄位便於 HTML 顯示。"""
    if ranking_df is None or ranking_df.empty or score_col not in ranking_df.columns:
        return ranking_df

    enhanced = ranking_df.copy()
    sid_key = enhanced['證券代號'].astype(str).str.strip()

    # 1. 法人連買天數（0-5+ 天）
    streaks = _compute_institutional_streak(all_data)
    enhanced['法人連買天'] = sid_key.map(lambda s: streaks.get(s, 0))

    # 2. 營收 YoY（%）
    rev_yoy_map = {}
    if revenue_data is not None and not revenue_data.empty:
        rev_col_candidates = ['去年同月增減(%)', 'YoY', '年增率(%)', '營收年增率']
        rev_col = next((c for c in rev_col_candidates if c in revenue_data.columns), None)
        if rev_col and '公司代號' in revenue_data.columns:
            for _, r in revenue_data.iterrows():
                sid = str(r.get('公司代號', '')).strip()
                v = r.get(rev_col)
                try:
                    v = float(str(v).replace('%', '').replace(',', ''))
                    rev_yoy_map[sid] = v
                except Exception:
                    continue
    enhanced['營收YoY'] = sid_key.map(
        lambda s: f"{rev_yoy_map.get(s):+.1f}%" if s in rev_yoy_map else '—')

    # 3. RS 加速度
    accel_map = _compute_rs_acceleration(sid_key.tolist())
    enhanced['RS加速'] = sid_key.map(
        lambda s: f"{accel_map[s]:+.2f}" if s in accel_map else '—')

    # 4. 計算 boost（加到分數）
    def _boost(row):
        sid = str(row['證券代號']).strip()
        b = 0.0
        # 法人連買：每天 +1，最多 +5
        b += min(streaks.get(sid, 0), 5) * 1.0
        # 營收 YoY > 20% +3, > 50% +6
        yoy = rev_yoy_map.get(sid)
        if yoy is not None:
            if yoy >= 50:
                b += 6
            elif yoy >= 20:
                b += 3
            elif yoy >= 0:
                b += 1
            elif yoy < -20:
                b -= 3
        # RS 加速度 > 0 代表 RS 由負轉正/加速領漲
        ac = accel_map.get(sid)
        if ac is not None:
            b += min(max(ac * 20, -3), 5)
        return round(b, 1)

    enhanced['情境加成'] = enhanced.apply(_boost, axis=1)
    # 套用到規則分數 / 綜合分數（不超過原分數的 ±20%）
    base = enhanced[score_col].astype(float)
    max_boost = base * 0.20
    adj = enhanced['情境加成'].clip(lower=-max_boost, upper=max_boost)
    enhanced[score_col] = (base + adj).round(1)
    return enhanced


# ===================== S18sec: 產業輪動（族群相對強度） =====================
# 沒有外部產業 mapping → 用 stock_id 前 2 碼作為族群代理（與 TWSE 大類一致）
# 1xxx 傳產 / 2xxx 電子 / 28xx 金融 / 47xx ETF / 5xxx-9xxx 雜項

_INDUSTRY_PREFIXES = {
    '11': '水泥', '12': '食品', '13': '塑化', '14': '紡織',
    '15': '電機機械', '16': '電器電纜', '17': '化學生技', '18': '玻璃陶瓷',
    '19': '造紙', '20': '鋼鐵', '21': '橡膠', '22': '汽車',
    '23': '半導體', '24': '電腦周邊', '25': '光電', '26': '通信網路',
    '27': '電子零組件', '28': '金融', '29': '貿易百貨',
    '30': '電子通路', '31': '資訊服務', '32': '其他電子',
    '33': '半導體', '34': '電腦周邊', '35': '光電', '36': '通信網路',
    '38': '電子通路', '46': '其他', '47': 'ETF', '49': '其他',
    '52': '建材營造', '55': '航運', '57': '觀光餐旅',
    '60': '金融', '88': 'ETF', '91': '其他',
}


def _infer_industry(stock_id: str) -> str:
    sid = str(stock_id).strip()
    if len(sid) < 2:
        return '其他'
    return _INDUSTRY_PREFIXES.get(sid[:2], '其他')


def _stock_20d_return(t):
    hist = (t or {}).get('close_history_30') or []
    if len(hist) < 21:
        return None
    try:
        a, b = float(hist[-21]), float(hist[-1])
        if a <= 0:
            return None
        return (b - a) / a
    except Exception:
        return None


def _apply_sector_rotation(ranking_df, taiex_state=None):
    """S18sec：依產業族群 20 日報酬輪動，加分龍頭族群成員，扣分落後族群。
    - 計算每一檔 20d return
    - 依推估產業 + 市場分組
    - 同產業內 ≥ 5 檔者：產業平均報酬 vs 大盤 → 龍頭 / 跟隨 / 落後
    - 龍頭族群成員 +4、跟隨 +1、落後 -3
    """
    if ranking_df is None or ranking_df.empty:
        return ranking_df
    score_col = None
    for c in ('三層融合', '混合分數', '綜合分數', '規則分數'):
        if c in ranking_df.columns:
            score_col = c; break
    if score_col is None:
        return ranking_df

    out = ranking_df.copy()
    out['_industry'] = out['證券代號'].astype(str).map(_infer_industry)

    # 個股 20d return
    rets = []
    for sid in out['證券代號'].astype(str):
        t = _technicals_cache.get(sid.strip())
        rets.append(_stock_20d_return(t))
    out['_ret20'] = rets

    taiex_ret20 = None
    if taiex_state:
        hist = taiex_state.get('close_history_30') or []
        if len(hist) >= 21:
            try:
                a, b = float(hist[-21]), float(hist[-1])
                if a > 0:
                    taiex_ret20 = (b - a) / a
            except Exception:
                pass
    if taiex_ret20 is None:
        taiex_ret20 = 0.0

    # 產業統計
    industry_stats = {}
    for ind, sub in out.groupby('_industry'):
        valid = sub['_ret20'].dropna()
        if len(valid) < 5:
            continue
        avg = float(valid.mean())
        industry_stats[ind] = {
            'avg_ret': avg,
            'lead': avg - taiex_ret20,    # 領先大盤多少
            'count': len(valid),
        }

    # 將 lead 排序給予分級
    if industry_stats:
        sorted_inds = sorted(industry_stats.items(),
                             key=lambda kv: kv[1]['lead'], reverse=True)
        n = len(sorted_inds)
        top_n = max(1, n // 3)
        bot_n = max(1, n // 3)
        leaders = {ind for ind, _ in sorted_inds[:top_n] if _['lead'] > 0.02}
        laggards = {ind for ind, _ in sorted_inds[-bot_n:] if _['lead'] < -0.02}
    else:
        leaders, laggards = set(), set()

    bonuses = []
    tags = []
    for ind, ret in zip(out['_industry'], out['_ret20']):
        st = industry_stats.get(ind)
        if st is None:
            tags.append(f'{ind}（樣本不足）'); bonuses.append(0.0); continue
        if ind in leaders:
            tags.append(f'{ind} 龍頭族群（{st["lead"]*100:+.1f}%）')
            bonuses.append(4.0)
        elif ind in laggards:
            tags.append(f'{ind} 落後族群（{st["lead"]*100:+.1f}%）')
            bonuses.append(-3.0)
        else:
            tags.append(f'{ind} 跟隨（{st["lead"]*100:+.1f}%）')
            bonuses.append(1.0)

    out['族群強度'] = tags
    out[score_col] = (out[score_col].astype(float) + pd.Series(bonuses)).round(1)
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    if '排名' in out.columns:
        out['排名'] = range(1, len(out) + 1)
    out.drop(columns=['_industry', '_ret20'], inplace=True)
    return out


# ===================== C4: 反轉守門員（Anti-trap risk score） =====================

def _reversal_risk_score(t):
    """C4：用既有技術指標，估算「未來短期下跌風險」 0~1 分（越高越危險）。
    無需訓練：完全規則組合，因子來自學界常見的 mean-reversion / 過熱訊號。

    高風險特徵（+0.10 ~ +0.25 each, 累加後 clip）：
      ① RSI > 80 且本日近 5 日已 +15%（過熱反轉）
      ② 收盤距 20 日布林上軌 > 2σ（極度乖離）
      ③ KD 雙鈍化（K > 80 且 K < D，已轉折）
      ④ 收盤距 52 週高 < 1%（追高停利潮）
      ⑤ ATR/Close > 8%（高波動，dump 風險大）
      ⑥ 量比 < 0.6（高位量縮無人接）
      ⑦ MA20 斜率轉負 + 收盤跌破 MA5
      ⑧ 上影線占比 > 4%（壓力線測試失敗）
    """
    risk = 0.0
    factors = []

    rsi = t.get('rsi')
    hist = t.get('close_history_30') or []
    if rsi is not None and rsi > 80 and len(hist) >= 6:
        try:
            arr = np.array(hist[-6:], dtype=float)
            if (arr > 0).all() and (arr[-1] - arr[0]) / arr[0] > 0.15:
                risk += 0.20
                factors.append('RSI>80+5日漲>15%')
        except Exception:
            pass

    bb_up = t.get('bb_upper'); close = t.get('close')
    bb_lo = t.get('bb_lower')
    if bb_up and close and bb_lo and bb_up > bb_lo:
        bb_mid = (bb_up + bb_lo) / 2
        sigma = (bb_up - bb_mid) / 2  # BB_STD=2
        if sigma > 0 and (close - bb_mid) / sigma > 2.0:
            risk += 0.15
            factors.append('BB乖離>2σ')

    k = t.get('kd_k'); d = t.get('kd_d')
    if k is not None and d is not None and k > 80 and k < d:
        risk += 0.15
        factors.append('KD高檔死叉')

    dist = t.get('dist_from_52w_high')
    if dist is not None and dist < 0.01:
        risk += 0.10
        factors.append('貼52週高')

    atr_pct = t.get('atr_pct')
    if atr_pct is not None and atr_pct > 0.08:
        risk += 0.10
        factors.append(f'ATR/Close {atr_pct*100:.1f}%')

    vr = t.get('volume_ratio')
    rsi_high = (rsi or 0) > 65
    if vr is not None and vr < 0.6 and rsi_high:
        risk += 0.10
        factors.append(f'高位量縮 {vr:.1f}x')

    ma20_sl = t.get('ma_20_slope')
    ma5 = t.get('ma_short')
    if (ma20_sl is not None and ma20_sl < -0.005
            and close and ma5 and close < ma5):
        risk += 0.15
        factors.append('MA20翻負+破MA5')

    upper_shadow = t.get('upper_shadow_pct')
    if upper_shadow is not None and upper_shadow > 0.04:
        risk += 0.10
        factors.append(f'長上影 {upper_shadow*100:.1f}%')

    return min(risk, 1.0), factors


def _apply_anti_trap_filter(ranking_df, score_col=None,
                            high_risk_threshold: float = 0.45):
    """C4：對排名套用反轉風險扣分。
    - 風險 ≥ 0.45 → 扣 5 分 + 標記 ⚠
    - 風險 ≥ 0.65 → 扣 10 分 + 標記 🛑（排名快速下沉）
    - 排名前 10 但風險 ≥ 0.55 → 額外標 '高位陷阱警示' 供使用者注意
    """
    if ranking_df is None or ranking_df.empty:
        return ranking_df
    if score_col is None:
        for cand in ('三層融合', '混合分數', '綜合分數', '規則分數'):
            if cand in ranking_df.columns:
                score_col = cand
                break
    if score_col is None or score_col not in ranking_df.columns:
        return ranking_df

    out = ranking_df.copy()
    sid_key = out['證券代號'].astype(str).str.strip()

    risks = []
    factor_strs = []
    deductions = []
    for sid in sid_key:
        t = _technicals_cache.get(sid)
        if t is None:
            risks.append(None); factor_strs.append('—'); deductions.append(0.0)
            continue
        r, fs = _reversal_risk_score(t)
        risks.append(r)
        if r >= 0.65:
            tag = '🛑 高反轉風險'; deduct = 10.0
        elif r >= high_risk_threshold:
            tag = '⚠ 反轉風險'; deduct = 5.0
        else:
            tag = '✓ 安全'; deduct = 0.0
        if fs:
            tag += f'：{",".join(fs[:3])}'
        factor_strs.append(tag)
        deductions.append(deduct)

    out['反轉風險'] = [
        f"{r*100:.0f}%" if r is not None else '—' for r in risks]
    out['風險訊號'] = factor_strs
    out[score_col] = (out[score_col].astype(float) - pd.Series(deductions)).round(1)
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    if '排名' in out.columns:
        out['排名'] = range(1, len(out) + 1)
    return out


# ===================== C1: 多時框共振（短期 + 中期動能驗證） =====================

def _short_term_momentum_score(t):
    """C1：5/10 日短期動能 0~1 分。
    成分：5日報酬正、MA5>MA10、KD 短期向上、收盤站上 MA5。"""
    score = 0.0
    weight = 0.0

    close = t.get('close')
    ma5 = t.get('ma_short')
    ma10 = t.get('ma_long')
    if close and ma5:
        weight += 0.25
        if close > ma5:
            score += 0.25
    if ma5 and ma10:
        weight += 0.25
        if ma5 > ma10:
            score += 0.25

    # 5 日斜率（從 close_history_30 末段近似）
    hist = t.get('close_history_30') or []
    if len(hist) >= 6:
        try:
            arr = np.array(hist[-6:], dtype=float)
            if (arr > 0).all():
                ret_5d = (arr[-1] - arr[0]) / arr[0]
                weight += 0.30
                # ret_5d ≥ 3% → 滿分；0~3% 線性；負值 0
                if ret_5d >= 0.03:
                    score += 0.30
                elif ret_5d > 0:
                    score += 0.30 * (ret_5d / 0.03)
        except Exception:
            pass

    k = t.get('kd_k'); d = t.get('kd_d')
    if k is not None and d is not None:
        weight += 0.20
        if k > d and k < 80:  # 避免過熱
            score += 0.20
    return (score / weight) if weight > 0 else 0.0


def _mid_term_momentum_score(t):
    """C1：20/60 日中期動能 0~1 分。
    成分：close>MA20、MA20>MA60、MA60 斜率≥0、20日報酬>0。"""
    score = 0.0
    weight = 0.0

    close = t.get('close')
    ma20 = t.get('ma_20'); ma60 = t.get('ma_60')
    if close and ma20:
        weight += 0.25
        if close > ma20:
            score += 0.25
    if ma20 and ma60:
        weight += 0.25
        if ma20 > ma60:
            score += 0.25
    sl60 = t.get('ma_60_slope')
    if sl60 is not None:
        weight += 0.30
        if sl60 >= 0:
            score += 0.30
        elif sl60 > -0.01:
            score += 0.15
    ret20 = t.get('return_20d')
    if ret20 is not None:
        weight += 0.20
        if ret20 >= 0.05:
            score += 0.20
        elif ret20 > 0:
            score += 0.20 * (ret20 / 0.05)
    return (score / weight) if weight > 0 else 0.0


def _apply_multi_timeframe(ranking_df, score_col=None):
    """C1：對排名加入多時框共振欄位與分數加成。
    - 短期 ≥ 0.6 + 中期 ≥ 0.6 → 共振強，+5 分
    - 短期 < 0.4 + 中期 ≥ 0.7 → 中期強短期弱（耐心等回測）, 0 分
    - 短期 ≥ 0.7 + 中期 < 0.3 → 短強中弱（容易誘多）, -3 分
    - 兩者皆 < 0.4 → 雙弱, -2 分
    """
    if ranking_df is None or ranking_df.empty:
        return ranking_df
    if score_col is None:
        for cand in ('三層融合', '混合分數', '綜合分數', '規則分數'):
            if cand in ranking_df.columns:
                score_col = cand
                break
    if score_col is None or score_col not in ranking_df.columns:
        return ranking_df

    out = ranking_df.copy()
    sid_key = out['證券代號'].astype(str).str.strip()

    short_scores = {}
    mid_scores = {}
    for sid in sid_key:
        t = _technicals_cache.get(sid)
        if t is None:
            continue
        short_scores[sid] = _short_term_momentum_score(t)
        mid_scores[sid] = _mid_term_momentum_score(t)

    def _label(s, m):
        if s is None or m is None:
            return '—', 0.0
        if s >= 0.6 and m >= 0.6:
            return '✅ 共振', 5.0
        if s >= 0.7 and m < 0.3:
            return '⚠ 短強中弱', -3.0
        if s < 0.4 and m >= 0.7:
            return '⏳ 中強短弱', 0.0
        if s < 0.4 and m < 0.4:
            return '❌ 雙弱', -2.0
        return '🟡 中性', 0.0

    labels = []
    bonuses = []
    for sid in sid_key:
        s = short_scores.get(sid)
        m = mid_scores.get(sid)
        lab, bonus = _label(s, m)
        labels.append(lab)
        bonuses.append(bonus)

    out['短期動能'] = [round(short_scores.get(s, 0)*100) if short_scores.get(s) is not None else '—' for s in sid_key]
    out['中期動能'] = [round(mid_scores.get(s, 0)*100) if mid_scores.get(s) is not None else '—' for s in sid_key]
    out['多時框'] = labels
    out[score_col] = (out[score_col].astype(float) + pd.Series(bonuses)).round(1)
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    if '排名' in out.columns:
        out['排名'] = range(1, len(out) + 1)
    return out


def _apply_cohort_relative(ranking_df, deep_preds=None, score_col=None):
    """Tier B3：cohort-relative percentile。

    看全市場 up_prob 百分位，前 10% = +3 分、前 25% = +1.5、後 10% = -3、後 25% = -1。

    score_col 修正：若未指定，自動挑選當下實際被排序的「分數欄位」
        優先順序：三層融合 > 混合分數 > 綜合分數 > 規則分數
    """
    if not deep_preds or ranking_df is None or ranking_df.empty:
        return ranking_df

    if score_col is None:
        for cand in ('三層融合', '混合分數', '綜合分數', '規則分數'):
            if cand in ranking_df.columns:
                score_col = cand
                break
    if score_col is None or score_col not in ranking_df.columns:
        return ranking_df

    all_up = pd.Series({s: d.get('up_prob', 0) for s, d in deep_preds.items()})
    if all_up.empty:
        return ranking_df
    pct_rank = all_up.rank(pct=True)
    up_pct_map = pct_rank.to_dict()

    def _adj(row):
        sid = str(row['證券代號']).strip()
        p = up_pct_map.get(sid)
        if p is None:
            return 0.0
        if p >= 0.90:
            return 3.0
        if p >= 0.75:
            return 1.5
        if p <= 0.10:
            return -3.0
        if p <= 0.25:
            return -1.0
        return 0.0

    out = ranking_df.copy()
    out['cohort_adj'] = out.apply(_adj, axis=1)
    out['相對強度百分位'] = out['證券代號'].astype(str).str.strip().map(
        lambda s: f"{up_pct_map.get(s, 0.5)*100:.0f}%" if s in up_pct_map else '—')
    out[score_col] = (out[score_col].astype(float) + out['cohort_adj']).round(1)
    # 重新依此分數排序（保持一致性）
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    if '排名' in out.columns:
        out['排名'] = range(1, len(out) + 1)
    out.drop(columns=['cohort_adj'], inplace=True, errors='ignore')
    return out


# ===================== ML Ranker 融合（路線 2） =====================

def apply_ml_ranking(ranking_df, candidate_pool_df=None, top_n=30, ml_weight=0.6):
    """若 ranker_model.json 存在，將 XGBoost ML 分數與規則分數加權融合。

    ranking_df: compute_composite_ranking 產出（已有 排名/綜合分數 等欄位）
    candidate_pool_df: 可選的更大候選池（如全部命中任一策略的股票），
                      用於讓 ML 對更多標的做分數，避免只對 TOP 30 排名
    ml_weight: ML 分數在混合公式中的權重（0~1）
                C8：實際使用權重會依 OOS IC 自動微調（最近 60 日 IC < -0.05 → 降權）

    回傳 (enhanced_df, used_ml: bool)
    """
    # C8：依 OOS IC 動態調整 ml_weight
    multipliers = _get_oos_weight_multipliers()
    if multipliers['ml'] != 1.0 or multipliers['rule'] != 1.0:
        ml_share = ml_weight * multipliers['ml']
        rule_share = (1 - ml_weight) * multipliers['rule']
        total = ml_share + rule_share
        if total > 0:
            ml_weight = ml_share / total
            print(f"  · C8 OOS 反饋：ML 權重微調 → {ml_weight:.2f} "
                  f"(ml ic mult={multipliers['ml']:.2f}, rule mult={multipliers['rule']:.2f})")
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

    # 欄位順序：把 ML/混合 放在分數後面（保留綜合分數，下游 cohort_relative / stacking 會用到）
    cols = enhanced.columns.tolist()
    desired_front = ['排名', '證券代號', '證券名稱', '市場',
                     '混合分數', 'ML 分數', '規則分數', '綜合分數',
                     '命中策略數', '命中策略',
                     'MA 狀態', 'RSI', '量比', '相對強度', '最新收盤']
    ordered = [c for c in desired_front if c in cols] + \
              [c for c in cols if c not in desired_front]
    enhanced = enhanced[ordered]

    matched = sum(1 for s in sid_key if s in ml_map)
    print(f"  ML 融合完成：{matched}/{len(enhanced)} 檔有 ML 分數 "
          f"(權重 rule:{1-ml_weight:.1f} / ml:{ml_weight:.1f})")
    return enhanced, True


# ===================== Deep Ranker 融合（路線 3） =====================

def apply_deep_ranking(ranking_df, top_n=30, deep_weight=0.35,
                       precomputed_preds=None):
    """若 deep_model.pt 存在，把 GRU 深度模型預測分數加入三層融合。

    ranking_df 需至少含 '證券代號' 與（若存在）'混合分數' 或 '綜合分數'。
    precomputed_preds: 來自 deep_ranker.predict_all_for_stocks 的 dict，避免重抓資料。
    回傳 (new_df, used_deep: bool)

    C8：deep_weight 會依 OOS IC 自動調整（最近 60 日 Deep 分數 IC 越高 → 權重越大）。
    """
    try:
        import deep_ranker as dr
    except Exception as e:
        print(f"  ⚠ deep_ranker 模組無法載入: {e}")
        return ranking_df, False

    if ranking_df is None or ranking_df.empty:
        return ranking_df, False

    # C8：依 OOS IC 動態調整 deep_weight
    multipliers = _get_oos_weight_multipliers()
    if multipliers['deep'] != 1.0:
        new_dw = min(0.7, max(0.05, deep_weight * multipliers['deep']))
        if abs(new_dw - deep_weight) > 0.01:
            print(f"  · C8 OOS 反饋：Deep 權重 {deep_weight:.2f} → {new_dw:.2f} "
                  f"(deep ic mult={multipliers['deep']:.2f})")
            deep_weight = new_dw

    # Tier A 快速路徑：重用已計算的預測
    if precomputed_preds:
        deep_map = {s: d['deep_score'] for s, d in precomputed_preds.items()}
        pred20_map = {s: d['pred_20d'] for s, d in precomputed_preds.items()}
        upprob_map = {s: d['up_prob'] for s, d in precomputed_preds.items()}
        lower_map = {s: d.get('lower_price') for s, d in precomputed_preds.items()}
        upper_map = {s: d.get('upper_price') for s, d in precomputed_preds.items()}
    else:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'

        model, ckpt = dr.load_model(device=device, return_ckpt=True)
        if model is None:
            return ranking_df, False

        sids = ranking_df['證券代號'].astype(str).str.strip().tolist()
        print(f"  載入 Deep 模型 ({dr.MODEL_FILE})，計算 {len(sids)} 檔序列分數...")
        try:
            X, ok_sids, _dates, _closes = dr.build_live_windows(
                sids, period='9mo', feature_cols=ckpt.get('features'))
            if len(X) == 0:
                print("  ⚠ 無可用序列視窗")
                return ranking_df, False
            pred_r, up_p = dr.predict_scores(model, X, device=device)
        except Exception as e:
            print(f"  ⚠ Deep 推論失敗: {e}")
            return ranking_df, False

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
        lower_map = {}
        upper_map = {}

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
    enhanced['上漲機率(20d)'] = sid_key.map(
        lambda s: f"{upprob_map.get(s, 0.0) * 100:.0f}%" if s in upprob_map else '—')

    # ── C7：Conformal 區間納入排名（風險報酬比加分） ──
    # 加分定義：上行%(close→upper) 與 下行%(close→lower) 的不對稱比
    #   bonus = clip( (upside_pct - 0.6 × downside_pct) × 100, -5, +8 )
    # 上行空間越大、下行越小 → 加分；反之扣分
    rr_bonus = {}
    if lower_map and upper_map:
        def _range(s):
            lo = lower_map.get(s); hi = upper_map.get(s)
            if lo is not None and hi is not None:
                return f"{lo:.1f}~{hi:.1f}"
            return '—'
        enhanced['Deep 95%區間'] = sid_key.map(_range)

        def _rr_score(s):
            lo = lower_map.get(s); hi = upper_map.get(s)
            t = _technicals_cache.get(s) or {}
            close = t.get('close')
            if lo is None or hi is None or close is None or close <= 0:
                return 0.0
            up = max(0.0, (hi - close) / close)
            down = max(0.0, (close - lo) / close)
            raw = (up - 0.6 * down) * 100
            return float(np.clip(raw, -5.0, 8.0))
        rr_bonus = {s: _rr_score(s) for s in lower_map.keys()}
        enhanced['區間RR分數'] = sid_key.map(lambda s: round(rr_bonus.get(s, 0.0), 2))

        # C16：個股不確定度警示（Conformal band 過寬 → 模型對該股低信心）
        def _band_tag(s):
            lo = lower_map.get(s); hi = upper_map.get(s)
            t = _technicals_cache.get(s) or {}
            close = t.get('close')
            if lo is None or hi is None or close is None or close <= 0:
                return ''
            band = (hi - lo) / close
            if band > 0.30:
                return f'⚠ 高不確定 {band*100:.0f}%'
            if band > 0.20:
                return f'△ 偏寬 {band*100:.0f}%'
            return ''
        enhanced['不確定度'] = sid_key.map(_band_tag)

    # C14：confidence-weighted blending
    # 信心 = 1.0 - clip(uncertainty, 0, 0.5)
    # 不確定度來源：
    #   ① 區間寬度 (upper-lower)/close > 0.18 → +0.20
    #   ② up_prob 與 breakout_prob 反向（差距 > 0.30）→ +0.15
    #   ③ 多時框「短強中弱」→ +0.10
    def _confidence(sid, t):
        unc = 0.0
        lo = lower_map.get(sid); hi = upper_map.get(sid)
        if lo is not None and hi is not None and t.get('close'):
            band = (hi - lo) / t['close']
            if band > 0.18:
                unc += min(0.20, (band - 0.18) * 1.0)
        up = upprob_map.get(sid)
        # breakout_prob 透過 _technicals_cache 暫無接入，這裡預留
        # 可由 row 取得（若 enhanced 已有 '突破機率(10d)'）
        return 1.0 - min(0.5, unc)

    def _fuse(row):
        sid = str(row['證券代號']).strip()
        base_norm = float(row[base_col]) / base_max * 100
        if sid in deep_map:
            t = _technicals_cache.get(sid, {})
            conf = _confidence(sid, t)
            # C14：信心係數降低 deep 權重
            eff_dw = deep_weight * conf
            blended = base_norm * (1 - eff_dw) + deep_map[sid] * eff_dw
        else:
            conf = 1.0
            blended = base_norm
        # C7：加上 RR bonus（最多 ±8 分）
        return blended + rr_bonus.get(sid, 0.0) * conf  # RR bonus 也按信心打折

    enhanced['三層融合'] = enhanced.apply(_fuse, axis=1).round(1)
    # C14：保留信心度欄位，方便 HTML 排序
    enhanced['模型信心'] = sid_key.map(
        lambda s: round(_confidence(s, _technicals_cache.get(s, {})), 2))

    # C24：三層融合來源歸因（rule / ml / deep 各貢獻 %）
    def _attribution(row):
        sid = str(row['證券代號']).strip()
        rule_norm = float(row[base_col]) / base_max * 100
        ml_norm = None
        if 'ML 分數' in row.index and pd.notna(row.get('ML 分數')):
            try:
                ml_norm = float(row['ML 分數']) / max(1.0, base_max) * 100
            except Exception:
                ml_norm = None
        deep_norm = float(deep_map.get(sid, 0.0)) if sid in deep_map else None

        # 計算各 layer 貢獻 = norm * 權重
        contrib = {}
        # rule (原 base_col 已含 ml；分離邏輯：base = (1-ml_w)*rule + ml_w*ml)
        if base_col == '混合分數' and ml_norm is not None:
            # 反推 rule 貢獻
            ml_w = 0.6  # 與 apply_ml_ranking 預設一致
            try:
                ml_w_used = 0.6
            except Exception:
                ml_w_used = 0.6
            rule_part = rule_norm * (1 - ml_w_used)
            ml_part = ml_norm * ml_w_used
            base_part = rule_part + ml_part
        else:
            rule_part = rule_norm
            ml_part = 0.0
            base_part = rule_norm

        if sid in deep_map:
            t = _technicals_cache.get(sid, {})
            eff_dw = deep_weight * _confidence(sid, t)
            contrib['rule'] = rule_part * (1 - eff_dw)
            contrib['ml'] = ml_part * (1 - eff_dw)
            contrib['deep'] = deep_norm * eff_dw
        else:
            contrib['rule'] = rule_part
            contrib['ml'] = ml_part
            contrib['deep'] = 0.0

        total = sum(max(0.0, v) for v in contrib.values())
        if total <= 0:
            return '—'
        pct = {k: (max(0.0, v) / total * 100) for k, v in contrib.items()}
        # 視覺化：▰ 比例 + 數字
        def _bar(p):
            n = int(round(p / 10))
            return '▰' * n + '▱' * (10 - n)
        parts = []
        if pct['rule'] >= 5:
            parts.append(f"R{pct['rule']:.0f}")
        if pct['ml'] >= 5:
            parts.append(f"M{pct['ml']:.0f}")
        if pct['deep'] >= 5:
            parts.append(f"D{pct['deep']:.0f}")
        return ' / '.join(parts) if parts else '—'

    enhanced['來源歸因'] = enhanced.apply(_attribution, axis=1)

    # D25：智能進場區間建議（避免裸追高）
    def _entry_zone(row):
        sid = str(row['證券代號']).strip()
        t = _technicals_cache.get(sid, {})
        close = t.get('close')
        ma5 = t.get('ma_5')
        ma20 = t.get('ma_20')
        rsi = t.get('rsi')
        atr = t.get('atr')
        if close is None or close <= 0:
            return '—'
        if rsi is not None and rsi >= 78:
            return '🔥 超買 等休息'
        if ma5 and close > ma5 * 1.06:
            target_lo = ma5 * 0.99
            target_hi = ma5 * 1.02
            return f'⏳ 漲多 等回測 {target_lo:.1f}~{target_hi:.1f}'
        if ma5 and close > ma5 * 1.03:
            target_lo = ma5
            target_hi = close * 0.99
            return f'⚠ 偏高 分批 {target_lo:.1f}~{target_hi:.1f}'
        # 正常區間：以 close ± 0.5×ATR 為建議區
        if atr and atr > 0:
            lo = close - 0.5 * atr
            hi = close + 0.3 * atr
            return f'✅ 可進 {lo:.1f}~{hi:.1f}'
        # 退而求其次：以 MA5 ~ close
        if ma5:
            return f'✅ 可進 {min(ma5, close):.1f}~{max(ma5, close):.1f}'
        return f'✅ 可進 ~{close:.1f}'

    enhanced['進場建議'] = enhanced.apply(_entry_zone, axis=1)
    enhanced = enhanced.sort_values('三層融合', ascending=False).reset_index(drop=True)
    enhanced = enhanced.head(top_n).copy()
    enhanced['排名'] = range(1, len(enhanced) + 1)

    # 欄位順序
    cols = enhanced.columns.tolist()
    desired = ['排名', '證券代號', '證券名稱', '市場',
               '元模型分數', '三層融合',
               '混合分數', 'ML 分數', 'Deep 分數',
               'Deep Pred20d', '上漲機率(20d)', 'Deep 95%區間', '區間RR分數', '不確定度',
               '分位', '歷史同分位OOS', 'OOS信心',
               'MC P(+10%)', 'MC P(+15%)', '突破機率(10d)', '模型信心', '來源歸因',
               '進場建議',
               '規則分數', '綜合分數', '命中策略數', '命中策略',
               '短期動能', '中期動能', '多時框',
               '反轉風險', '風險訊號', '族群強度',
               'MA 狀態', 'RSI', '量比', '相對強度', '最新收盤']
    ordered = [c for c in desired if c in cols] + \
              [c for c in cols if c not in desired]
    enhanced = enhanced[ordered]

    matched = sum(1 for s in sid_key if s in deep_map)
    print(f"  Deep 融合完成：{matched}/{len(enhanced)} 檔有 Deep 分數 "
          f"(Deep weight={deep_weight:.2f})")
    return enhanced, True


# ===================== 低估股篩選 =====================

def find_undervalued(all_df, min_score=3, deep_preds=None, up_prob_min=0.45):
    """篩選股價可能被低估、且法人開始進場的標的。
    Tier A 增強：同 find_early_potential，融合 deep_preds 目標價與上漲機率。"""
    grouped = all_df.groupby('證券代號')
    results = []
    deep_preds = deep_preds or {}

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
            sid_key = str(stock_id).strip()
            dp = deep_preds.get(sid_key)

            # A3：低估股的 up_prob 門檻放寬（0.45），因低估股本就偏弱
            if dp is not None:
                up_v = dp.get('up_prob')
                # B9：同時處理 None / NaN
                if up_v is not None and not pd.isna(up_v) and float(up_v) < up_prob_min:
                    continue

            target_price, upside = _estimate_target_price(t, deep_pred=dp)
            row = {
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
            }
            if dp is not None:
                if dp.get('up_prob') is not None:
                    row['上漲機率(20d)'] = f"{dp['up_prob']*100:.0f}%"
                if dp.get('pred_20d') is not None:
                    row['模型預測漲幅'] = f"{dp['pred_20d']*100:+.1f}%"
                lo = dp.get('lower_price'); hi = dp.get('upper_price')
                if lo is not None and hi is not None:
                    row['95%區間'] = f"{lo:.2f} ~ {hi:.2f}"
            results.append(row)

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
    market_state = get_taiex_state()
    bullish = bool(market_state.get('bullish', True))
    stress = _market_stress_level(market_state)
    if stress == 'shock':
        print(f"  ⚠ D4：大盤跌破 MA60 > {abs(MARKET_DRAWDOWN_THRESHOLD)*100:.0f}% → "
              f"所有 Chandelier 停損縮緊至 {CHANDELIER_ATR_MULT_SHOCK}×ATR、禁止加碼")

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

        # ── 1) 動態停損（含 Break-even、分層、Chandelier；D4: 大盤壓力縮緊）──
        current_stop = None
        if buy_price and initial_stop:
            current_stop = _calc_current_stop(
                buy_price, initial_stop, max_high, initial_atr,
                tier_status, bullish=bullish, market_state=market_state,
                close=close,
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

        # ── 7.5) D2: Donchian 10 日低出場（趨勢結束的客觀訊號） ──
        # 條件：已獲利（profit_pct ≥ 5%）+ 收盤跌破近 10 日低 → 半倉
        # 大盤 shock 時 → 全部出場（更積極保護獲利）
        low_10d = t.get('low_10d')
        if (low_10d is not None and close < low_10d
                and profit_pct is not None and profit_pct >= 0.05):
            if stress == 'shock':
                sell_reasons.append(f'📉 D2 跌破 10 日低 {low_10d:.2f}（大盤 shock，全出）')
                action, action_priority = '全部出場', max(action_priority, 3)
            elif action_priority < 2:
                sell_reasons.append(f'📉 D2 跌破 10 日低 {low_10d:.2f}，趨勢結束，建議半倉')
                action, action_priority = '分批出一半', 2

        # ── 7.7) D3: RS Line 轉弱（個股相對大盤走弱）──
        # 條件：大盤多頭 + 個股 RS_5MA 跌破 RS_20MA + 已獲利 ≥ 3% → 先減半
        if (bullish and profit_pct is not None and profit_pct >= 0.03):
            rs_trend = _compute_rs_line_trend(
                t.get('close_history_30') or [],
                market_state.get('close_history_30') or []
            )
            if rs_trend.get('weak'):
                gap_pct = (rs_trend['rs_20ma'] - rs_trend['rs_5ma']) / max(rs_trend['rs_20ma'], 1e-9) * 100
                if gap_pct >= 0.5 and action_priority < 1:
                    sell_reasons.append(
                        f'📉 D3 RS Line 5MA 跌破 20MA（{gap_pct:.1f}% 落後大盤），個股相對轉弱'
                    )
                    action, action_priority = '分批減碼', 1

        # ── 8) 加碼訊號（僅趨勢策略、保本以上、多頭；D4：shock 直接禁止）──
        add_signal = ''
        ids = _extract_strategy_ids(strategy)
        if stress == 'shock':
            pass  # D4：大盤跌破 MA60 > 5% 全面禁止加碼
        elif (bullish and buy_price and profit_pct is not None
                and profit_pct >= 0.05
                and tier and max_high and max_high >= tier['1R']
                and any(i in ids for i in ['4', '11', '13', '14', '16'])
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


def _compute_rs_line_trend(stock_close_hist, taiex_close_hist):
    """D3：計算 RS line 的 5MA vs 20MA 趨勢狀態。
    回傳 dict: {'rs_5ma': float, 'rs_20ma': float, 'weak': bool, 'rs_value': float}
    weak=True 表示 RS_5MA 跌破 RS_20MA（個股相對大盤轉弱）。
    """
    out = {'rs_5ma': None, 'rs_20ma': None, 'weak': None, 'rs_value': None}
    if not stock_close_hist or not taiex_close_hist:
        return out
    n = min(len(stock_close_hist), len(taiex_close_hist))
    if n < 25:
        return out
    try:
        s = np.array(stock_close_hist[-n:], dtype=float)
        m = np.array(taiex_close_hist[-n:], dtype=float)
        if (s <= 0).any() or (m <= 0).any():
            return out
        rs_line = s / m
        if not np.isfinite(rs_line).all():
            return out
        rs_5 = float(rs_line[-5:].mean())
        rs_20 = float(rs_line[-20:].mean())
        out.update({
            'rs_5ma': rs_5,
            'rs_20ma': rs_20,
            'weak': rs_5 < rs_20,
            'rs_value': float(rs_line[-1]),
        })
    except Exception:
        pass
    return out


def _market_stress_level(market_state=None):
    """D4：市場壓力等級。
    回傳 'shock'（TAIEX 跌破 MA60 > 5%）/ 'bear'（一般空頭）/ 'normal'。"""
    ms = market_state or get_taiex_state()
    bullish = ms.get('bullish', True)
    if bullish:
        return 'normal'
    close = ms.get('close'); ma60 = ms.get('ma_60')
    if close and ma60 and ma60 > 0:
        dd = (close - ma60) / ma60
        if dd <= MARKET_DRAWDOWN_THRESHOLD:
            return 'shock'
    return 'bear'


def _calc_current_stop(buy_price, initial_stop, max_high, atr, tier_status,
                       bullish=True, market_state=None, close=None):
    """動態停損 = 所有適用停損規則取最大值（只上升不下降）。
    D4：market_state 傳入時，依大盤壓力等級加倍縮緊（shock 用 1.5x ATR）。
    D15：依「max_high vs buy_price 漲幅」階梯式縮緊 Chandelier ATR 倍數
         （+10% → ×0.9、+20% → ×0.75、+30% → ×0.6）。
    D16：close 與 max_high 同時提供時，當 max_high ≥ buy_price × 1.15
         觸發「盈利保護」緊縮停損 = max(stop, close - 1×ATR)。"""
    if buy_price is None or initial_stop is None:
        return None
    stops = [initial_stop]
    if max_high is not None and max_high >= buy_price * (1 + BREAKEVEN_PROFIT_TRIGGER):
        stops.append(buy_price)
    tier = _calc_tier_levels(buy_price, initial_stop)
    if tier:
        if tier_status == '1R已達':
            stops.append(buy_price)
        elif tier_status == '2R已達':
            stops.append(tier['1R'])
        elif tier_status == '3R已達':
            stops.append(tier['2R'])
    if max_high is not None and max_high > buy_price and atr:
        if market_state is not None:
            stress = _market_stress_level(market_state)
            mult = {
                'shock': CHANDELIER_ATR_MULT_SHOCK,
                'bear': CHANDELIER_ATR_MULT_BEAR,
                'normal': CHANDELIER_ATR_MULT,
            }.get(stress, CHANDELIER_ATR_MULT)
        else:
            mult = CHANDELIER_ATR_MULT_BEAR if not bullish else CHANDELIER_ATR_MULT
        # D15：依累積漲幅縮緊
        gain = (max_high - buy_price) / buy_price if buy_price > 0 else 0.0
        if gain >= 0.30:
            mult = max(0.9, mult * 0.6)
        elif gain >= 0.20:
            mult = max(1.2, mult * 0.75)
        elif gain >= 0.10:
            mult = max(1.5, mult * 0.9)
        ch = _calc_chandelier_stop(max_high, atr, mult=mult)
        if ch is not None:
            stops.append(ch)

        # D16：盈利保護（max_high 漲幅 ≥ 15%）
        if close is not None and gain >= 0.15:
            tight = close - 1.0 * atr
            stops.append(tight)

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


# C5：Kelly fraction 緩存（避免每次 _suggest_position_size 重讀 CSV）
_KELLY_CACHE: dict = {}


def _kelly_fraction_from_history(min_trades: int = 20) -> dict:
    """C5：從 trade_history.csv 計算 Kelly fraction 動態調整單筆風險。

    Kelly% = p - (1 - p) / b
        p = 勝率
        b = 平均獲利 / |平均虧損|
    為避免高估，採用 0.25× Kelly（Kelly Quarter，業界常見）。
    回傳 {'kelly_q': float, 'win_rate': float, 'payoff': float, 'n': int, 'source': str}
    """
    if 'cached' in _KELLY_CACHE:
        return _KELLY_CACHE['cached']

    default = {'kelly_q': POSITION_RISK_PCT, 'win_rate': None,
               'payoff': None, 'n': 0, 'source': 'default'}
    if not os.path.exists(TRADE_HISTORY_FILE):
        _KELLY_CACHE['cached'] = default
        return default
    try:
        df = pd.read_csv(TRADE_HISTORY_FILE)
    except Exception:
        _KELLY_CACHE['cached'] = default
        return default
    if df.empty or '損益%' not in df.columns:
        _KELLY_CACHE['cached'] = default
        return default

    rets = pd.to_numeric(df['損益%'], errors='coerce').dropna() / 100.0
    rets = rets[rets.between(-0.5, 1.0)]   # clip 極端值
    if len(rets) < min_trades:
        out = dict(default, n=len(rets), source=f'history_too_few({len(rets)})')
        _KELLY_CACHE['cached'] = out
        return out

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    if len(wins) == 0 or len(losses) == 0:
        _KELLY_CACHE['cached'] = default
        return default

    p = len(wins) / len(rets)
    avg_win = float(wins.mean())
    avg_loss = float(abs(losses.mean()))
    if avg_loss <= 0:
        _KELLY_CACHE['cached'] = default
        return default

    b = avg_win / avg_loss
    kelly = p - (1 - p) / b
    kelly_q = max(0.0, kelly * 0.25)               # quarter kelly
    # 安全上限：單筆風險不超過帳戶 2.5% (POSITION_RISK_PCT × 2.5)
    kelly_q = min(kelly_q, POSITION_RISK_PCT * 2.5)
    # 下限：不低於 POSITION_RISK_PCT × 0.5（避免 kelly < 0 完全停手）
    kelly_q = max(kelly_q, POSITION_RISK_PCT * 0.5)

    out = {
        'kelly_q': float(kelly_q),
        'win_rate': round(p, 3),
        'payoff': round(b, 2),
        'n': int(len(rets)),
        'source': 'trade_history',
    }
    _KELLY_CACHE['cached'] = out
    return out


def _market_risk_budget(stress: str = 'normal'):
    """D6：依市場壓力等級回傳風險預算。
    回傳 dict:
      - position_risk_mult：單筆風險倍率（套用在 POSITION_RISK_PCT）
      - max_total_exposure：總曝險上限（占帳戶比例）
      - max_per_position：單一檔上限（占帳戶比例）
      - desc：說明文字
    """
    table = {
        'normal': {
            'position_risk_mult': 1.0,
            'max_total_exposure': 1.00,
            'max_per_position': 0.15,
            'desc': '常態：滿倉允許，單筆 ≤15%',
        },
        'bear': {
            'position_risk_mult': 0.6,
            'max_total_exposure': 0.60,
            'max_per_position': 0.10,
            'desc': '空頭：總曝險 ≤60%、單筆 ≤10%、單筆風險 0.6×',
        },
        'shock': {
            'position_risk_mult': 0.4,
            'max_total_exposure': 0.30,
            'max_per_position': 0.05,
            'desc': '崩跌（shock）：總曝險 ≤30%、單筆 ≤5%、單筆風險 0.4×',
        },
    }
    return table.get(stress, table['normal'])


def _vol_scale_factor(atr_pct, target_vol: float = 0.03):
    """D8：波動度 Scaling — 個股 ATR/Close 大幅高於 target_vol 時，按比例縮減倉位。
    回傳乘數 ≤ 1.0：
      atr_pct ≤ target_vol → 1.0（不變）
      atr_pct = 2×target → 0.5
      atr_pct = 3×target → 0.33
      下限：0.25（不會壓到完全進不了場）
    """
    if atr_pct is None or atr_pct <= 0:
        return 1.0
    if atr_pct <= target_vol:
        return 1.0
    return max(0.25, target_vol / atr_pct)


def _suggest_position_size(buy_price, atr, account_size=None, strategy=None,
                           use_kelly: bool = True, market_state=None,
                           vol_scaling: bool = True):
    """按 ATR 反向縮放建議張數。
    C5：use_kelly=True 時，單筆風險百分比改用 0.25× Kelly（從歷史 trade_history 動態計算）。
    D6：market_state 提供時，依大盤壓力等級對單筆風險再縮放。
    D8：vol_scaling=True 時，個股波動度（ATR/Close）大時自動降張數。
    """
    acc = account_size or _get_account_size()
    if not buy_price or not atr or atr <= 0:
        return 1
    ids = _extract_strategy_ids(strategy)
    mult = 2.0 if any(i in ids for i in ['4', '11', '13', '14', '16']) else ATR_INITIAL_STOP_MULT
    risk_per_share = mult * atr
    if use_kelly:
        risk_pct = _kelly_fraction_from_history()['kelly_q']
    else:
        risk_pct = POSITION_RISK_PCT
    # D6：依大盤壓力縮放
    if market_state is not None:
        stress = _market_stress_level(market_state)
        budget = _market_risk_budget(stress)
        risk_pct *= budget['position_risk_mult']
    # D8：依個股波動度縮放
    if vol_scaling and buy_price > 0:
        atr_pct = atr / buy_price
        risk_pct *= _vol_scale_factor(atr_pct)
    max_risk_total = acc * risk_pct
    shares = max_risk_total / (risk_per_share * 1000)
    # D6：總部位金額上限（不超過帳戶 max_per_position）
    if market_state is not None:
        budget = _market_risk_budget(_market_stress_level(market_state))
        max_position_value = acc * budget['max_per_position']
        max_shares_by_position = max_position_value / (buy_price * 1000)
        shares = min(shares, max_shares_by_position)
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


# ===================== D5: 停損後黑名單（避免反覆被洗） =====================

BLACKLIST_DAYS = 30        # 停損後幾日內列入黑名單
BLACKLIST_LOSS_THRESHOLD = -0.03  # 損益 < -3% 才視為「真停損」（小幅出場不列）


def get_recent_stoploss_blacklist(days: int = BLACKLIST_DAYS) -> dict:
    """從 trade_history.csv 找出最近 N 天內因停損出場的股票。
    回傳 {sid: {'days_ago': int, 'loss_pct': float, 'reason': str, 'exit_date': str}}。
    判定為停損的條件：
      - 出場原因含「停損 / 跌破 / Chandelier」其中之一
      - 損益% ≤ -3%（過濾分批/分層落袋）
    """
    blacklist: dict = {}
    if not os.path.exists(TRADE_HISTORY_FILE):
        return blacklist
    try:
        df = pd.read_csv(TRADE_HISTORY_FILE)
    except Exception:
        return blacklist
    if df.empty or '日期' not in df.columns:
        return blacklist
    today = datetime.now().date()
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df[df['日期'].notna()]
    cutoff = pd.Timestamp(today - pd.Timedelta(days=days).to_pytimedelta())
    df = df[df['日期'] >= cutoff]
    if df.empty:
        return blacklist

    stoploss_keywords = ['停損', '跌破', 'Chandelier', 'ATR']
    for _, row in df.iterrows():
        reason = str(row.get('出場原因', '') or '')
        if not any(k in reason for k in stoploss_keywords):
            continue
        loss = row.get('損益%')
        if pd.notna(loss):
            try:
                loss_v = float(loss) / 100.0
                if loss_v > BLACKLIST_LOSS_THRESHOLD:
                    continue  # 不夠虧損，視為主動減碼非停損
            except (ValueError, TypeError):
                continue
        else:
            loss_v = None

        sid = str(row.get('證券代號', '')).strip()
        if not sid:
            continue
        days_ago = (pd.Timestamp(today) - row['日期']).days
        cur = blacklist.get(sid)
        if cur is None or days_ago < cur['days_ago']:
            blacklist[sid] = {
                'days_ago': int(days_ago),
                'loss_pct': loss_v,
                'reason': reason[:60],
                'exit_date': row['日期'].strftime('%Y-%m-%d'),
            }
    return blacklist


def _apply_blacklist(df: pd.DataFrame, blacklist: dict, name: str = '排名') -> pd.DataFrame:
    """過濾掉黑名單股票，並印出剔除清單。
    對於 ranking 等大表，回傳剔除後 DataFrame；保留剔除統計做主流程印出。"""
    if df is None or df.empty or not blacklist or '證券代號' not in df.columns:
        return df
    sid_str = df['證券代號'].astype(str).str.strip()
    mask_kick = sid_str.isin(blacklist.keys())
    kicked = sid_str[mask_kick].tolist()
    if not kicked:
        return df
    out = df[~mask_kick].reset_index(drop=True)
    if '排名' in out.columns:
        out['排名'] = range(1, len(out) + 1)
    print(f"  · D5 黑名單剔除自{name}：{len(kicked)} 檔 → {kicked[:8]}{'...' if len(kicked) > 8 else ''}")
    return out


# ===================== D19: Ranking snapshot + stacking_train_log 整合 =====================

RANKING_SNAPSHOT_FILE = 'ranking_snapshots.csv'
STACKING_TRAIN_LOG_FILE = 'stacking_train_log.csv'


OOS_MONITOR_FILE = 'oos_monitor.csv'
OOS_IC_FILE = 'oos_ic.json'
STRATEGY_SNAPSHOT_FILE = 'strategy_snapshots.csv'   # C11：各策略每日 hits
STRATEGY_OOS_FILE = 'strategy_oos.csv'              # C11：策略 OOS 歷史報酬
STRATEGY_OOS_SUMMARY_FILE = 'strategy_oos_summary.json'  # C11：每策略命中率彙總


def _save_ranking_snapshot(ranking_df, market_state):
    """每天執行時把當日 ranking 的 (代號, 規則分數, ML 分數, Deep 分數, 上漲機率, 市場狀態) 快照下來。
    供日後與 trade_history.csv 配對，產生 stacking 訓練資料。
    C6：同時保存當日收盤價，供 OOS 監控計算 forward return。"""
    if ranking_df is None or ranking_df.empty:
        return
    cols_pref = ['證券代號', '規則分數', 'ML 分數', 'Deep 分數', '混合分數',
                 '綜合分數', '元模型分數', '三層融合', '上漲機率(20d)']
    keep = [c for c in cols_pref if c in ranking_df.columns]
    if not keep:
        return
    snap = ranking_df[keep].copy()
    snap.insert(0, '日期', datetime.now().strftime('%Y-%m-%d'))
    # C6：保存快照當日收盤價（供 forward return 計算）
    snap['快照收盤'] = snap['證券代號'].astype(str).str.strip().map(
        lambda s: (_technicals_cache.get(s, {}) or {}).get('close')
    )
    bullish = bool(market_state.get('bullish', True)) if isinstance(market_state, dict) else True
    ret20 = (market_state or {}).get('return_20d', 0.0) or 0.0
    if ret20 > 0.02:
        regime = 'bull'
    elif ret20 < -0.02:
        regime = 'bear'
    else:
        regime = 'neutral'
    snap['市場狀態'] = regime
    file_exists = os.path.exists(RANKING_SNAPSHOT_FILE)
    snap.to_csv(RANKING_SNAPSHOT_FILE, mode='a', header=not file_exists,
                index=False, encoding='utf-8-sig')


def compute_oos_monitor(window_days: int = 10, top_n: int = 10):
    """C6 lite：滾動 OOS 精準監控。
    對於 ≥ window_days 個交易日前的 ranking 快照，計算 Top-N 的 forward 期間報酬，
    寫入 OOS_MONITOR_FILE 供報告引用。

    回傳 dict（最近 90 天的統計）：
      {'mean_ret': float, 'hit_rate': float, 'n_dates': int, 'window_days': int}
    """
    if not os.path.exists(RANKING_SNAPSHOT_FILE):
        return None
    try:
        snaps = pd.read_csv(RANKING_SNAPSHOT_FILE)
    except Exception as e:
        print(f"  · C6 OOS 讀取失敗：{e}")
        return None
    if snaps.empty or '快照收盤' not in snaps.columns:
        return None

    # 依 priority 選分數欄
    score_col = next((c for c in ('三層融合', '混合分數', '綜合分數', '規則分數')
                      if c in snaps.columns), None)
    if score_col is None:
        return None

    snaps['日期'] = pd.to_datetime(snaps['日期'], errors='coerce')
    snaps = snaps.dropna(subset=['日期'])
    snaps['證券代號'] = snaps['證券代號'].astype(str).str.strip()
    snaps['快照收盤'] = pd.to_numeric(snaps['快照收盤'], errors='coerce')

    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=window_days)
    eligible_dates = sorted(snaps['日期'].unique())
    eligible_dates = [d for d in eligible_dates if d <= cutoff]
    if not eligible_dates:
        return None

    rows = []
    for date in eligible_dates:
        sub = snaps[snaps['日期'] == date].dropna(subset=['快照收盤', score_col])
        if sub.empty:
            continue
        sub = sub.sort_values(score_col, ascending=False).head(top_n)
        rets = []
        for _, r in sub.iterrows():
            sid = r['證券代號']
            snap_close = r['快照收盤']
            if not snap_close or snap_close <= 0:
                continue
            cur = (_technicals_cache.get(sid, {}) or {}).get('close')
            if cur is None or cur <= 0:
                continue
            rets.append((cur - snap_close) / snap_close)
        if not rets:
            continue
        rows.append({
            '日期': date.strftime('%Y-%m-%d'),
            '評估窗口(日)': window_days,
            '樣本數': len(rets),
            '平均報酬': round(float(np.mean(rets)), 4),
            '正報酬比例': round(float(np.mean([1 if r > 0 else 0 for r in rets])), 3),
            '最佳': round(float(max(rets)), 4),
            '最差': round(float(min(rets)), 4),
        })

    if not rows:
        return None
    out = pd.DataFrame(rows)
    out.to_csv(OOS_MONITOR_FILE, index=False, encoding='utf-8-sig')

    # 取最近 90 天的彙總
    recent = out.tail(90)
    return {
        'mean_ret': float(recent['平均報酬'].mean()),
        'hit_rate': float(recent['正報酬比例'].mean()),
        'n_dates': int(len(recent)),
        'window_days': window_days,
        'best_date_ret': float(recent['平均報酬'].max()),
        'worst_date_ret': float(recent['平均報酬'].min()),
    }


def _save_strategy_snapshot(buy_dfs_map):
    """C11：把當天每個策略的命中標的（含當日收盤價）寫入 STRATEGY_SNAPSHOT_FILE。
    後續 compute_strategy_oos() 可以配對 forward return 評估每個策略的真實命中率。
    C21：附加當日市況（bull / bear）以利做 regime sensitivity 分析。
    """
    if not buy_dfs_map:
        return
    today = datetime.now().strftime('%Y-%m-%d')
    # C21：偵測當日市況
    try:
        regime = 'bull' if is_market_bullish() else 'bear'
    except Exception:
        regime = 'unknown'
    rows = []
    for sname, df in buy_dfs_map.items():
        if df is None or df.empty:
            continue
        for sid in df['證券代號'].astype(str).str.strip().unique():
            close = (_technicals_cache.get(sid, {}) or {}).get('close')
            if close is None:
                continue
            rows.append({
                '日期': today,
                '策略': sname,
                '證券代號': sid,
                '快照收盤': close,
                '市況': regime,
            })
    if not rows:
        return
    snap = pd.DataFrame(rows)
    file_exists = os.path.exists(STRATEGY_SNAPSHOT_FILE)
    snap.to_csv(STRATEGY_SNAPSHOT_FILE, mode='a', header=not file_exists,
                index=False, encoding='utf-8-sig')


def compute_strategy_oos(window_days: int = 10, max_history_days: int = 180):
    """C11：對每個策略計算「過去快照日的 forward 平均報酬與勝率」。
    寫入 STRATEGY_OOS_FILE（細項）、STRATEGY_OOS_SUMMARY_FILE（彙總）。
    回傳 dict: {strategy_name: {'mean_ret': X, 'hit_rate': Y, 'n_signals': N, 'n_dates': D}}
    """
    if not os.path.exists(STRATEGY_SNAPSHOT_FILE):
        return None
    try:
        snaps = pd.read_csv(STRATEGY_SNAPSHOT_FILE)
    except Exception:
        return None
    if snaps.empty:
        return None

    snaps['日期'] = pd.to_datetime(snaps['日期'], errors='coerce')
    snaps = snaps.dropna(subset=['日期'])
    snaps['證券代號'] = snaps['證券代號'].astype(str).str.strip()
    snaps['快照收盤'] = pd.to_numeric(snaps['快照收盤'], errors='coerce')
    snaps = snaps.dropna(subset=['快照收盤'])

    today = pd.Timestamp.now().normalize()
    cutoff_max = today - pd.Timedelta(days=window_days)
    cutoff_min = today - pd.Timedelta(days=window_days + max_history_days)
    snaps = snaps[(snaps['日期'] <= cutoff_max) & (snaps['日期'] >= cutoff_min)]
    if snaps.empty:
        return None

    detail_rows = []
    has_regime = '市況' in snaps.columns
    for _, r in snaps.iterrows():
        sid = r['證券代號']
        snap_close = r['快照收盤']
        if not snap_close or snap_close <= 0:
            continue
        cur = (_technicals_cache.get(sid, {}) or {}).get('close')
        if cur is None or cur <= 0:
            continue
        detail_rows.append({
            '日期': r['日期'].strftime('%Y-%m-%d'),
            '策略': r['策略'],
            '證券代號': sid,
            '快照收盤': snap_close,
            '當前收盤': cur,
            'forward_return': (cur - snap_close) / snap_close,
            '市況': r.get('市況', 'unknown') if has_regime else 'unknown',
        })
    if not detail_rows:
        return None
    df = pd.DataFrame(detail_rows)
    try:
        df.to_csv(STRATEGY_OOS_FILE, index=False, encoding='utf-8-sig')
    except Exception:
        pass

    summary = {}
    for sname, sub in df.groupby('策略'):
        rets = sub['forward_return'].astype(float)
        std = float(rets.std()) if len(rets) > 1 else 0.0
        # C20：簡化 Sharpe（per-signal mean / std × √(252/window)）— 年化近似
        # forward_return 是 N 日報酬，年化倍數 = √(252 / window_days)
        ann_factor = (252 / max(window_days, 1)) ** 0.5
        sharpe = (float(rets.mean()) / std * ann_factor) if std > 1e-9 else 0.0
        # C20：daily-aggregated 最大回撤（按日均報酬累積）
        try:
            daily_mean = sub.groupby('日期')['forward_return'].mean().sort_index()
            cumret = (1 + daily_mean).cumprod()
            running_max = cumret.cummax()
            drawdown = (cumret / running_max - 1.0).min()
            max_dd = float(drawdown) if pd.notna(drawdown) else 0.0
        except Exception:
            max_dd = 0.0
        # C21：依市況分組
        regime_breakdown = {}
        if '市況' in sub.columns:
            for reg, reg_sub in sub.groupby('市況'):
                if reg in (None, '', 'nan', float('nan')):
                    continue
                reg_rets = reg_sub['forward_return'].astype(float)
                if len(reg_rets) < 3:
                    continue
                regime_breakdown[str(reg)] = {
                    'mean_ret': round(float(reg_rets.mean()), 4),
                    'hit_rate': round(float((reg_rets > 0).mean()), 3),
                    'n_signals': int(len(reg_rets)),
                }
        summary[sname] = {
            'mean_ret': round(float(rets.mean()), 4),
            'median_ret': round(float(rets.median()), 4),
            'hit_rate': round(float((rets > 0).mean()), 3),
            'n_signals': int(len(rets)),
            'n_dates': int(sub['日期'].nunique()),
            'best': round(float(rets.max()), 4),
            'worst': round(float(rets.min()), 4),
            'std': round(std, 4),
            'sharpe': round(sharpe, 3),
            'max_dd': round(max_dd, 4),
            'regime_breakdown': regime_breakdown,   # C21
        }
    try:
        with open(STRATEGY_OOS_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    except Exception:
        pass
    return summary


STRATEGY_DECAY_FILE = 'strategy_decay.json'  # C18：策略衰退偵測結果


def compute_strategy_decay(recent_window_days: int = 60,
                           full_window_days: int = 180,
                           min_signals: int = 8):
    """C18：比較「最近 N 天」與「完整視窗」的策略 OOS mean_ret 是否惡化。
    若 recent_mean - full_mean ≤ -0.015 且 recent 樣本 ≥ min_signals → 衰退。
    寫入 STRATEGY_DECAY_FILE 並回傳 dict {strategy: {full, recent, drop, decayed}}。
    """
    if not os.path.exists(STRATEGY_OOS_FILE):
        return {}
    try:
        df = pd.read_csv(STRATEGY_OOS_FILE)
    except Exception:
        return {}
    if df.empty or '策略' not in df.columns:
        return {}
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期', 'forward_return'])
    today = pd.Timestamp.now().normalize()
    cutoff_recent = today - pd.Timedelta(days=recent_window_days)
    cutoff_full = today - pd.Timedelta(days=full_window_days)
    full_df = df[df['日期'] >= cutoff_full]
    recent_df = df[df['日期'] >= cutoff_recent]
    if full_df.empty:
        return {}

    out = {}
    for sname, full_sub in full_df.groupby('策略'):
        full_rets = full_sub['forward_return'].astype(float)
        recent_sub = recent_df[recent_df['策略'] == sname]
        if recent_sub.empty:
            continue
        recent_rets = recent_sub['forward_return'].astype(float)
        if len(recent_rets) < min_signals:
            continue
        full_mean = float(full_rets.mean())
        recent_mean = float(recent_rets.mean())
        drop = recent_mean - full_mean
        full_hit = float((full_rets > 0).mean())
        recent_hit = float((recent_rets > 0).mean())
        decayed = drop <= -0.015 and recent_mean < 0.0
        out[sname] = {
            'full_mean': round(full_mean, 4),
            'recent_mean': round(recent_mean, 4),
            'drop': round(drop, 4),
            'full_hit': round(full_hit, 3),
            'recent_hit': round(recent_hit, 3),
            'full_n': int(len(full_rets)),
            'recent_n': int(len(recent_rets)),
            'decayed': bool(decayed),
        }
    try:
        with open(STRATEGY_DECAY_FILE, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    except Exception:
        pass
    return out


PORTFOLIO_BT_FILE = 'portfolio_backtest.json'


def backtest_topn_portfolio(top_n: int = 10,
                            rebalance_days: int = 20,
                            lookback_days: int = 365,
                            initial_capital: float = 1_000_000.0,
                            score_col: str = None,
                            cost_pct: float = 0.005):
    """C22：Top-N 月度再平衡投組回測。
    用 ranking_snapshots.csv 重建歷史 Top-N 名單，
    每 `rebalance_days` 個交易日再平衡一次（等權重），
    出場/換股扣 `cost_pct` 雙邊成本（含手續費 + 滑價）。
    回傳並寫入 PORTFOLIO_BT_FILE：
      total_return, ann_return, sharpe, max_drawdown,
      win_rate, n_trades, equity_curve [(date, equity)], months_summary
    若快照不足回傳 None。
    """
    if not os.path.exists(RANKING_SNAPSHOT_FILE):
        return None
    try:
        snaps = pd.read_csv(RANKING_SNAPSHOT_FILE)
    except Exception:
        return None
    if snaps.empty:
        return None
    if score_col is None:
        for c in ('三層融合', '元模型分數', '混合分數', '綜合分數'):
            if c in snaps.columns:
                score_col = c
                break
    if score_col is None or '快照收盤' not in snaps.columns:
        return None
    snaps['日期'] = pd.to_datetime(snaps['日期'], errors='coerce')
    snaps = snaps.dropna(subset=['日期', score_col, '快照收盤'])
    snaps['證券代號'] = snaps['證券代號'].astype(str).str.strip()
    snaps['快照收盤'] = pd.to_numeric(snaps['快照收盤'], errors='coerce')
    snaps[score_col] = pd.to_numeric(snaps[score_col], errors='coerce')
    snaps = snaps.dropna(subset=['快照收盤', score_col])

    today = pd.Timestamp.now().normalize()
    cutoff_min = today - pd.Timedelta(days=lookback_days)
    snaps = snaps[(snaps['日期'] >= cutoff_min)]
    if snaps.empty:
        return None

    # 每天的快照（取每日該股最後一筆，避免重複）
    snaps = snaps.sort_values(['日期', '證券代號'])
    snaps = snaps.drop_duplicates(subset=['日期', '證券代號'], keep='last')

    # 構建每日 close lookup（用各日 snapshot 中的「快照收盤」）
    price_map = snaps.set_index(['日期', '證券代號'])['快照收盤'].to_dict()
    score_map_by_date = {}
    for d, grp in snaps.groupby('日期'):
        gg = grp.sort_values(score_col, ascending=False).head(top_n * 3)
        score_map_by_date[d] = list(zip(gg['證券代號'].tolist(), gg[score_col].tolist()))
    rebal_dates = sorted(score_map_by_date.keys())
    if len(rebal_dates) < 2:
        return None

    # 取再平衡日（每隔 rebalance_days 個交易日）
    selected_rebal = rebal_dates[::max(1, rebalance_days)]
    if rebal_dates[-1] not in selected_rebal:
        selected_rebal.append(rebal_dates[-1])

    cur_cash = initial_capital
    cur_holdings = {}    # sid → (shares, entry_price)
    equity_curve = []
    trade_results = []   # 每筆 (entry, exit, return)
    last_eval_date = None

    for i, rd in enumerate(selected_rebal):
        # 1. 用今日快照 close 估算當前市值
        mkt_value = 0.0
        for sid, (sh, ent) in cur_holdings.items():
            cur_p = price_map.get((rd, sid))
            if cur_p is None:
                # 找最近 5 個交易日內的價格
                for back in range(1, 6):
                    cur_p = price_map.get((rd - pd.Timedelta(days=back), sid))
                    if cur_p is not None:
                        break
            if cur_p is None:
                cur_p = ent
            mkt_value += sh * cur_p
        equity = cur_cash + mkt_value
        equity_curve.append((rd.strftime('%Y-%m-%d'), equity))

        # 2. 取目前 ranking top-N
        cands = score_map_by_date.get(rd, [])
        new_topn = [sid for sid, _ in cands if price_map.get((rd, sid))][:top_n]
        if not new_topn:
            continue

        # 3. 出場：不在 new_topn 的持股
        next_holdings = {}
        for sid, (sh, ent) in list(cur_holdings.items()):
            cur_p = price_map.get((rd, sid)) or ent
            if sid not in new_topn:
                proceeds = sh * cur_p * (1 - cost_pct)
                cur_cash += proceeds
                trade_results.append({
                    'sid': sid, 'entry': ent, 'exit': cur_p,
                    'return': (cur_p - ent) / ent,
                })
            else:
                next_holdings[sid] = (sh, ent)

        # 4. 進場：new_topn 中尚未持有的，等權重買入
        equity_now = cur_cash + sum(sh * (price_map.get((rd, s)) or e)
                                    for s, (sh, e) in next_holdings.items())
        target_per = equity_now / max(top_n, 1)
        for sid in new_topn:
            if sid in next_holdings:
                continue
            p = price_map.get((rd, sid))
            if not p or p <= 0:
                continue
            buy_cost = target_per
            shares = (buy_cost * (1 - cost_pct)) / p
            if shares <= 0:
                continue
            cur_cash -= buy_cost
            next_holdings[sid] = (shares, p)

        cur_holdings = next_holdings
        last_eval_date = rd

    # 結算最後一日
    if last_eval_date is not None:
        rd = rebal_dates[-1]
        mkt_value = 0.0
        for sid, (sh, ent) in cur_holdings.items():
            cur_p = price_map.get((rd, sid)) or ent
            mkt_value += sh * cur_p
        final_equity = cur_cash + mkt_value
        if equity_curve and equity_curve[-1][0] != rd.strftime('%Y-%m-%d'):
            equity_curve.append((rd.strftime('%Y-%m-%d'), final_equity))
    else:
        return None

    if len(equity_curve) < 2:
        return None

    # 績效統計
    eq = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date').reset_index(drop=True)
    eq['ret'] = eq['equity'].pct_change().fillna(0.0)

    total_return = (eq['equity'].iloc[-1] / eq['equity'].iloc[0]) - 1
    days_span = max(1, (eq['date'].iloc[-1] - eq['date'].iloc[0]).days)
    ann_return = (1 + total_return) ** (365 / days_span) - 1 if total_return > -1 else -1

    rebal_per_year = 252.0 / max(1, rebalance_days)
    if eq['ret'].std() > 0:
        sharpe = (eq['ret'].mean() / eq['ret'].std()) * (rebal_per_year ** 0.5)
    else:
        sharpe = 0.0

    cummax = eq['equity'].cummax()
    drawdown = (eq['equity'] - cummax) / cummax
    max_dd = float(drawdown.min())

    if trade_results:
        wins = [t for t in trade_results if t['return'] > 0]
        win_rate = len(wins) / len(trade_results)
        avg_win = float(np.mean([t['return'] for t in wins])) if wins else 0.0
        losses = [t for t in trade_results if t['return'] <= 0]
        avg_loss = float(np.mean([t['return'] for t in losses])) if losses else 0.0
    else:
        win_rate, avg_win, avg_loss = 0.0, 0.0, 0.0

    summary = {
        'config': {
            'top_n': top_n, 'rebalance_days': rebalance_days,
            'lookback_days': lookback_days, 'initial_capital': initial_capital,
            'cost_pct': cost_pct, 'score_col': score_col,
        },
        'period': {
            'start': eq['date'].iloc[0].strftime('%Y-%m-%d'),
            'end':   eq['date'].iloc[-1].strftime('%Y-%m-%d'),
            'days':  days_span,
        },
        'metrics': {
            'total_return': float(total_return),
            'ann_return': float(ann_return),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'n_trades': len(trade_results),
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'final_equity': float(eq['equity'].iloc[-1]),
        },
        'equity_curve': equity_curve,
    }
    try:
        with open(PORTFOLIO_BT_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return summary


def _render_portfolio_bt_card(bt: dict) -> str:
    """C22：投組回測結果卡（含 equity curve）。"""
    if not bt:
        return ''
    m = bt.get('metrics', {})
    p = bt.get('period', {})
    cfg = bt.get('config', {})
    curve = bt.get('equity_curve', [])
    if len(curve) < 2:
        return ''
    dates = [c[0] for c in curve]
    eqs = [c[1] for c in curve]

    def _color_ret(v):
        return '#3a9' if v >= 0 else '#c55'

    chart_div = f'pf_bt_{int(time.time())%100000}'
    chart_html = f"""
<div id="{chart_div}" style="height:260px;"></div>
<script>
Plotly.newPlot('{chart_div}', [{{
    x: {json.dumps(dates)},
    y: {json.dumps(eqs)},
    type: 'scatter', mode: 'lines',
    line: {{color: '#5af', width: 2}},
    fill: 'tozeroy', fillcolor: 'rgba(90,170,255,0.15)',
    name: '投組權益'
}}], {{
    paper_bgcolor: '#0d1118', plot_bgcolor: '#0d1118',
    font: {{color: '#d5d5d5', size: 11}},
    xaxis: {{title: '日期', gridcolor: '#22272e'}},
    yaxis: {{title: '權益 (NT$)', gridcolor: '#22272e'}},
    margin: {{t: 20, l: 60, r: 30, b: 40}}
}}, {{responsive: true, displayModeBar: false}});
</script>"""

    return f"""
<div class="alert-card" style="border-left:4px solid #5af;">
  <h3 style="margin-top:0;">📈 C22 Top-N 投組回測（月度再平衡）</h3>
  <p style="color:#a0a0b0; font-size:11.5px; margin:2px 0 10px;">
    Top-{cfg.get('top_n')} 等權重 · 每 {cfg.get('rebalance_days')} 個交易日再平衡 ·
    雙邊成本 {cfg.get('cost_pct', 0)*100:.1f}% · 起始資金 {cfg.get('initial_capital', 0):,.0f}
  </p>
  <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin-bottom:12px;">
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">期間</div>
      <div style="font-size:13px; color:#d5d5d5;">{p.get('start')} → {p.get('end')}</div>
    </div>
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">總報酬</div>
      <div style="font-size:18px; font-weight:600; color:{_color_ret(m.get('total_return',0))};">{m.get('total_return',0)*100:+.2f}%</div>
    </div>
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">年化</div>
      <div style="font-size:18px; font-weight:600; color:{_color_ret(m.get('ann_return',0))};">{m.get('ann_return',0)*100:+.2f}%</div>
    </div>
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">Sharpe</div>
      <div style="font-size:18px; font-weight:600; color:{'#3a9' if m.get('sharpe',0)>=1 else '#fa3' if m.get('sharpe',0)>=0 else '#c55'};">{m.get('sharpe',0):.2f}</div>
    </div>
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">Max DD</div>
      <div style="font-size:18px; font-weight:600; color:#c55;">{m.get('max_drawdown',0)*100:.2f}%</div>
    </div>
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">勝率</div>
      <div style="font-size:18px; font-weight:600;">{m.get('win_rate',0)*100:.1f}%</div>
    </div>
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">交易筆數</div>
      <div style="font-size:18px; font-weight:600;">{m.get('n_trades',0)}</div>
    </div>
    <div style="background:#0d1118; padding:8px; border-radius:6px; text-align:center;">
      <div style="font-size:10.5px; color:#888;">最終權益</div>
      <div style="font-size:13px; color:#d5d5d5;">{m.get('final_equity',0):,.0f}</div>
    </div>
  </div>
  {chart_html}
  <p style="color:#a0a0b0; font-size:11px; margin-top:8px;">
    ※ 用歷史 ranking_snapshots.csv 與當日快照收盤計算；不含除權息調整與借券成本。
  </p>
</div>
"""


def annotate_ranking_with_oos_percentile(ranking_df, score_col=None,
                                         max_history_days: int = 180,
                                         window_days: int = 10,
                                         n_bins: int = 10):
    """C23：用歷史 ranking_snapshots.csv 計算「分數分位 → 平均 forward 報酬」表，
    然後對當前 ranking_df 各列附加：
      - `分位` (1~n_bins，10 = 最高 10%)
      - `歷史同分位 OOS` (該分位平均報酬)
      - `OOS 信心` (五星：基於 mean_ret 的對應星級)
    若樣本不足則回傳原 df。"""
    if ranking_df is None or ranking_df.empty:
        return ranking_df
    if not os.path.exists(RANKING_SNAPSHOT_FILE):
        return ranking_df
    if score_col is None:
        for c in ('三層融合', '混合分數', '綜合分數', '規則分數'):
            if c in ranking_df.columns:
                score_col = c
                break
    if score_col is None or score_col not in ranking_df.columns:
        return ranking_df
    try:
        snaps = pd.read_csv(RANKING_SNAPSHOT_FILE)
    except Exception:
        return ranking_df
    if snaps.empty or score_col not in snaps.columns or '快照收盤' not in snaps.columns:
        return ranking_df

    snaps['日期'] = pd.to_datetime(snaps['日期'], errors='coerce')
    snaps = snaps.dropna(subset=['日期', score_col, '快照收盤'])
    snaps['證券代號'] = snaps['證券代號'].astype(str).str.strip()
    snaps['快照收盤'] = pd.to_numeric(snaps['快照收盤'], errors='coerce')
    snaps[score_col] = pd.to_numeric(snaps[score_col], errors='coerce')
    snaps = snaps.dropna(subset=['快照收盤', score_col])
    today = pd.Timestamp.now().normalize()
    cutoff_max = today - pd.Timedelta(days=window_days)
    cutoff_min = today - pd.Timedelta(days=window_days + max_history_days)
    snaps = snaps[(snaps['日期'] <= cutoff_max) & (snaps['日期'] >= cutoff_min)]
    if len(snaps) < n_bins * 5:
        return ranking_df

    # forward return（用當前 cache 的 close）
    rows = []
    for _, r in snaps.iterrows():
        sid = r['證券代號']; snap_close = r['快照收盤']; score = r[score_col]
        if not snap_close or snap_close <= 0:
            continue
        cur = (_technicals_cache.get(sid, {}) or {}).get('close')
        if cur is None or cur <= 0:
            continue
        rows.append({'score': float(score),
                     'forward_return': (cur - snap_close) / snap_close})
    if len(rows) < n_bins * 5:
        return ranking_df
    hist = pd.DataFrame(rows)
    # 用 quantile 切 bins（避免邊界 NaN）
    try:
        hist['分位'] = pd.qcut(hist['score'], q=n_bins,
                              labels=list(range(1, n_bins + 1)),
                              duplicates='drop')
    except Exception:
        return ranking_df
    bin_stats = hist.groupby('分位', observed=True).agg(
        mean_ret=('forward_return', 'mean'),
        hit_rate=('forward_return', lambda x: float((x > 0).mean())),
        n=('forward_return', 'count'),
    ).to_dict('index')

    # bin edges
    try:
        _, edges = pd.qcut(hist['score'], q=n_bins, retbins=True, duplicates='drop')
    except Exception:
        return ranking_df
    edges = list(edges)

    def _bin_for_score(x):
        try:
            v = float(x)
        except Exception:
            return None
        for i in range(1, len(edges)):
            if v <= edges[i]:
                return i
        return len(edges) - 1

    def _stars(mean_ret):
        if mean_ret >= 0.05:  return '★★★★★'
        if mean_ret >= 0.03:  return '★★★★'
        if mean_ret >= 0.015: return '★★★'
        if mean_ret >= 0.0:   return '★★'
        return '★'

    out = ranking_df.copy()
    bins_col, oos_col, conf_col = [], [], []
    for _, row in out.iterrows():
        b = _bin_for_score(row[score_col])
        if b is None or b not in bin_stats:
            bins_col.append('—'); oos_col.append('—'); conf_col.append('—')
            continue
        st_b = bin_stats[b]
        bins_col.append(f"{int(b)}/{n_bins}")
        oos_col.append(f"{st_b['mean_ret']*100:+.2f}%")
        conf_col.append(_stars(st_b['mean_ret']))
    out['分位'] = bins_col
    out['歷史同分位OOS'] = oos_col
    out['OOS信心'] = conf_col
    return out


def _get_strategy_decay_multipliers(min_signals: int = 8):
    """C18：把衰退策略再乘 0.5 倍率，回傳 {strategy: 0.5}。"""
    out = {}
    if not os.path.exists(STRATEGY_DECAY_FILE):
        return out
    try:
        with open(STRATEGY_DECAY_FILE, 'r', encoding='utf-8') as f:
            decay = json.load(f)
    except Exception:
        return out
    for sname, st in decay.items():
        if st.get('decayed') and int(st.get('recent_n', 0)) >= min_signals:
            out[sname] = 0.5
    return out


def compute_oos_ic_attribution(window_days: int = 10, lookback_days: int = 60):
    """C8：對 ranking 快照中各分數欄計算 IC（Information Coefficient = Spearman），
    用來知道規則分數 / ML / Deep / 上漲機率「現在」對 forward return 的解釋力，
    寫入 OOS_IC_FILE 並回傳 dict。

    回傳：
      {
        'window_days': 10, 'lookback_days': 60, 'n_samples': N,
        'ic': {'規則分數': 0.05, 'ML 分數': 0.12, 'Deep 分數': 0.18, '三層融合': 0.21, '上漲機率(20d)': 0.10}
      }
    IC > 0 → 該分數高的個股確實漲得多；IC < 0 → 該分數失靈。
    """
    if not os.path.exists(RANKING_SNAPSHOT_FILE):
        return None
    try:
        snaps = pd.read_csv(RANKING_SNAPSHOT_FILE)
    except Exception:
        return None
    if snaps.empty or '快照收盤' not in snaps.columns:
        return None

    snaps['日期'] = pd.to_datetime(snaps['日期'], errors='coerce')
    snaps = snaps.dropna(subset=['日期'])
    snaps['證券代號'] = snaps['證券代號'].astype(str).str.strip()
    snaps['快照收盤'] = pd.to_numeric(snaps['快照收盤'], errors='coerce')
    snaps = snaps.dropna(subset=['快照收盤'])

    today = pd.Timestamp.now().normalize()
    cutoff_max = today - pd.Timedelta(days=window_days)            # 至少 window_days 前
    cutoff_min = today - pd.Timedelta(days=window_days + lookback_days)  # 不超過 lookback
    snaps = snaps[(snaps['日期'] <= cutoff_max) & (snaps['日期'] >= cutoff_min)]
    if snaps.empty:
        return None

    rows = []
    for _, r in snaps.iterrows():
        sid = r['證券代號']
        snap_close = r['快照收盤']
        if not snap_close or snap_close <= 0:
            continue
        cur = (_technicals_cache.get(sid, {}) or {}).get('close')
        if cur is None or cur <= 0:
            continue
        rec = {'fwd_ret': (cur - snap_close) / snap_close}
        for col in ('規則分數', 'ML 分數', 'Deep 分數', '混合分數',
                    '三層融合', '元模型分數', '上漲機率(20d)'):
            if col in snaps.columns:
                v = r.get(col)
                try:
                    rec[col] = float(v) if v is not None and pd.notna(v) else None
                except Exception:
                    rec[col] = None
        rows.append(rec)

    if len(rows) < 30:
        return None
    df = pd.DataFrame(rows)

    ic_dict = {}
    for col in df.columns:
        if col == 'fwd_ret':
            continue
        sub = df[[col, 'fwd_ret']].dropna()
        if len(sub) < 30 or sub[col].std() == 0:
            continue
        try:
            ic = float(sub[col].rank().corr(sub['fwd_ret'].rank()))
        except Exception:
            ic = None
        if ic is not None and np.isfinite(ic):
            ic_dict[col] = round(ic, 4)

    payload = {
        'window_days': window_days,
        'lookback_days': lookback_days,
        'n_samples': int(len(df)),
        'ic': ic_dict,
        'updated': datetime.now().isoformat(timespec='seconds'),
    }
    try:
        with open(OOS_IC_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  · C8 IC 寫入失敗：{e}")
    return payload


def _get_strategy_oos_multipliers(min_signals: int = 5,
                                  shrinkage_k: int = 30):
    """C13 + C19：依 OOS 表現給倍率（0.4 ~ 1.4），並用 Bayesian credibility
    shrinkage 把樣本少的策略往中性 1.0 拉。

    規則（mean_ret 為過去 forward 10 日平均報酬）：
      mean ≥ +3% & 勝率 ≥ 55% → 1.4
      mean ≥ +1.5%             → 1.2
      -1% < mean < +1.5%       → 1.0（中性）
      mean ≤ -1%               → 0.7
      mean ≤ -3%               → 0.4
      樣本 < min_signals 不調整（保留預設權重）。

    C19：credibility = n / (n + shrinkage_k)
      最終 mult = 1.0 + (raw_mult − 1.0) × credibility
      預設 k=30 → n=10 → 25% 套用、n=30 → 50%、n=90 → 75%、n=300 → ~91%
    """
    out = {}
    if not os.path.exists(STRATEGY_OOS_SUMMARY_FILE):
        return out
    try:
        with open(STRATEGY_OOS_SUMMARY_FILE, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception:
        return out

    # C21：取得當前市況以做 regime-aware lookup
    try:
        cur_regime = 'bull' if is_market_bullish() else 'bear'
    except Exception:
        cur_regime = None

    for sname, st_row in summary.items():
        n = int(st_row.get('n_signals') or 0)
        if n < min_signals:
            continue
        mean = float(st_row.get('mean_ret') or 0.0)
        hit = float(st_row.get('hit_rate') or 0.5)
        # C21：若當前 regime 有足夠樣本，採用 regime 內的 mean / hit
        rb = st_row.get('regime_breakdown') or {}
        if cur_regime and cur_regime in rb and rb[cur_regime].get('n_signals', 0) >= min_signals:
            mean = float(rb[cur_regime].get('mean_ret', mean))
            hit = float(rb[cur_regime].get('hit_rate', hit))
            n = int(rb[cur_regime].get('n_signals', n))
        if mean >= 0.03 and hit >= 0.55:
            raw = 1.4
        elif mean >= 0.015:
            raw = 1.2
        elif mean <= -0.03:
            raw = 0.4
        elif mean <= -0.01:
            raw = 0.7
        else:
            raw = 1.0
        credibility = n / (n + shrinkage_k) if shrinkage_k > 0 else 1.0
        out[sname] = round(1.0 + (raw - 1.0) * credibility, 3)
    return out


def _get_oos_weight_multipliers():
    """C8：讀 OOS_IC_FILE，把 IC 轉成 layer 權重倍率（給 apply_ml/deep_ranking 使用）。
    規則：
      IC ≥ 0.10 → multiplier = 1.5（信心足）
      0.05 ≤ IC < 0.10 → 1.2
     -0.05 ≤ IC < 0.05 → 1.0
     -0.10 ≤ IC < -0.05 → 0.7
      IC < -0.10 → 0.4（明顯失靈，大幅降權）

    回傳 dict: {'ml': 1.2, 'deep': 0.7, 'rule': 1.0}（缺值預設 1.0）
    """
    out = {'ml': 1.0, 'deep': 1.0, 'rule': 1.0}
    if not os.path.exists(OOS_IC_FILE):
        return out
    try:
        with open(OOS_IC_FILE, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        ic = payload.get('ic', {})
    except Exception:
        return out

    def _scale(v):
        if v is None or not np.isfinite(v):
            return 1.0
        if v >= 0.10:
            return 1.5
        if v >= 0.05:
            return 1.2
        if v >= -0.05:
            return 1.0
        if v >= -0.10:
            return 0.7
        return 0.4

    out['ml'] = _scale(ic.get('ML 分數'))
    out['deep'] = _scale(ic.get('Deep 分數'))
    out['rule'] = _scale(ic.get('規則分數'))
    return out


def _maybe_retrain_stacking_meta(min_samples: int = 200, min_days: int = 7):
    """C10：自動偵測 stacking_train_log.csv 是否累積了足夠新樣本，
    並且距上次訓練超過 min_days 天 → 觸發 stacking_meta 自動重訓。

    觸發條件（任一）：
      a) 從未訓練過 + 樣本數 ≥ min_samples
      b) 樣本數 ≥ min_samples 且 (距上次訓練 ≥ min_days 天 或 新增樣本 ≥ 100 筆)
    """
    if not os.path.exists(STACKING_TRAIN_LOG_FILE):
        return False
    try:
        log = pd.read_csv(STACKING_TRAIN_LOG_FILE)
    except Exception:
        return False
    n_total = len(log)
    if n_total < min_samples:
        print(f"  · C10：stacking 訓練樣本 {n_total}/{min_samples}，未達門檻，跳過 retrain")
        return False

    try:
        from stacking_meta import StackingBlender, META_FILE
    except Exception as e:
        print(f"  · C10：stacking_meta 模組無法載入：{e}")
        return False

    # 讀現有 meta 判斷上次訓練資訊
    last_trained = None
    last_n_train = 0
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            last_trained = meta.get('trained_at')
            last_n_train = int(meta.get('n_train', 0) or 0)
        except Exception:
            pass

    need_retrain = False
    reason = ''
    if last_trained is None:
        need_retrain = True
        reason = '首次訓練'
    else:
        try:
            dt_last = pd.to_datetime(last_trained)
            days_since = (pd.Timestamp.now() - dt_last).days
        except Exception:
            days_since = 999
        new_samples = n_total - last_n_train
        if days_since >= min_days and new_samples >= 30:
            need_retrain = True
            reason = f'距上次 {days_since} 天且新增 {new_samples} 筆'
        elif new_samples >= 100:
            need_retrain = True
            reason = f'新增 {new_samples} 筆樣本'

    if not need_retrain:
        print(f"  · C10：stacking 模型已最新（n_train={last_n_train}/{n_total}），跳過 retrain")
        return False

    print(f"  · C10：觸發 stacking meta retrain — {reason}（樣本 {n_total}）")
    rows = log.to_dict(orient='records')
    blender = StackingBlender.train(rows, min_samples=min_samples)
    if blender is None:
        return False
    try:
        blender.save()
        print(f"  ✓ C10：stacking_meta.json 已更新（n_train={blender.n_train}）")
        return True
    except Exception as e:
        print(f"  · C10：模型儲存失敗：{e}")
        return False


def integrate_trade_history_to_stacking():
    """把 trade_history.csv（已關閉的交易）配對 ranking_snapshots.csv（買進日當天分數），
    產出 stacking_train_log.csv 供 stacking_meta train 使用。
    每筆 row：rule, ml, deep, up_prob, market, actual_ret"""
    if not os.path.exists(TRADE_HISTORY_FILE):
        return 0
    if not os.path.exists(RANKING_SNAPSHOT_FILE):
        return 0
    try:
        trades = pd.read_csv(TRADE_HISTORY_FILE)
        snaps = pd.read_csv(RANKING_SNAPSHOT_FILE)
    except Exception as e:
        print(f"  · D19 資料讀取失敗：{e}")
        return 0
    if trades.empty or snaps.empty:
        return 0

    # 標準化型別
    trades['證券代號'] = trades['證券代號'].astype(str).str.strip()
    snaps['證券代號'] = snaps['證券代號'].astype(str).str.strip()
    trades['買進日期'] = trades['買進日期'].astype(str).str.strip()
    snaps['日期'] = snaps['日期'].astype(str).str.strip()

    merged = trades.merge(
        snaps, how='inner',
        left_on=['買進日期', '證券代號'],
        right_on=['日期', '證券代號'],
    )
    if merged.empty:
        return 0

    def _norm_score(s):
        try:
            v = float(s)
            return max(0, min(100, v))
        except Exception:
            return None

    def _norm_pct(s):
        if s is None or pd.isna(s):
            return None
        if isinstance(s, str) and s.endswith('%'):
            try:
                return float(s.rstrip('%')) / 100
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    rows = []
    for _, r in merged.iterrows():
        ret_pct = r.get('損益%')
        if pd.isna(ret_pct):
            continue
        rule = _norm_score(r.get('規則分數') or r.get('綜合分數'))
        ml = _norm_score(r.get('ML 分數'))
        deep = _norm_score(r.get('Deep 分數'))
        if rule is None:
            continue
        rows.append({
            'date': r.get('買進日期'),
            'stock_id': r.get('證券代號'),
            'rule': rule,
            'ml': ml,
            'deep': deep,
            'up_prob': _norm_pct(r.get('上漲機率(20d)')),
            'market': r.get('市場狀態') or 'neutral',
            'actual_ret': float(ret_pct) / 100,
            'hold_days': r.get('持有天數'),
            'exit_reason': r.get('出場原因') or '',
        })
    if not rows:
        return 0

    df_out = pd.DataFrame(rows)
    file_exists = os.path.exists(STACKING_TRAIN_LOG_FILE)
    if file_exists:
        try:
            existed = pd.read_csv(STACKING_TRAIN_LOG_FILE)
            # 以 (date, stock_id) 去重
            df_out = pd.concat([existed, df_out], ignore_index=True)
            df_out = df_out.drop_duplicates(subset=['date', 'stock_id'], keep='last')
        except Exception:
            pass
    df_out.to_csv(STACKING_TRAIN_LOG_FILE, index=False, encoding='utf-8-sig')
    return len(rows)


def check_portfolio_risk(holdings_df, market_state=None):
    """投組風險總覽：總成本、總市值、總風險、集中度警示。
    D6：依市場壓力等級附帶風險預算。
    D7：族群集中度警示（半導體/金融/...）。
    """
    result = {
        'total_cost': 0, 'total_market': 0, 'total_risk': 0,
        'mtm_pct': 0, 'warnings': [], 'positions': 0,
        'max_concentration': 0, 'bullish': is_market_bullish(),
        'sector_concentration': {},        # D7
        'risk_budget': None,               # D6
    }
    if holdings_df is None or holdings_df.empty:
        return result
    total_cost = 0
    total_market = 0
    total_risk = 0
    cost_by_sid = {}
    cost_by_strategy = {}
    cost_by_sector = {}        # D7
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
        sector = _infer_industry(sid)
        cost_by_sector[sector] = cost_by_sector.get(sector, 0) + cost
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

    # D7：族群集中度
    if total_cost > 0 and cost_by_sector:
        sector_pcts = {s: c / total_cost for s, c in cost_by_sector.items()}
        result['sector_concentration'] = {
            s: round(p, 4) for s, p in sorted(
                sector_pcts.items(), key=lambda kv: kv[1], reverse=True)
        }
        for s, p in sector_pcts.items():
            if p > 0.5 and len(cost_by_sector) > 1:
                result['warnings'].append(
                    f'⚠ D7 族群「{s}」占 {p*100:.1f}%（超過 50%），族群集中風險高'
                )
            elif p > 0.4 and len(cost_by_sector) > 2:
                result['warnings'].append(
                    f'D7 族群「{s}」占 {p*100:.1f}%，建議分散到其他產業'
                )

        # D9：持股族群轉弱警示
        # 計算大盤 20d 報酬作基準
        taiex_ret = 0.0
        if market_state and isinstance(market_state, dict):
            taiex_ret = float(market_state.get('return_20d', 0.0) or 0.0)
        sector_ret_map = {}
        for sec in sector_pcts.keys():
            # 收集所有快取中該族群的個股 20d 報酬
            rets = []
            for sid, t in _technicals_cache.items():
                if _infer_industry(sid) != sec:
                    continue
                r = _stock_20d_return(t)
                if r is not None:
                    rets.append(r)
            if len(rets) >= 5:
                sector_ret_map[sec] = float(np.mean(rets))
        result['sector_returns'] = {
            s: round(r, 4) for s, r in sector_ret_map.items()
        }
        for sec, r in sector_ret_map.items():
            lead = r - taiex_ret
            pct = sector_pcts.get(sec, 0)
            if pct < 0.05:
                continue   # 微小持倉不警示
            if lead < -0.05:
                result['warnings'].append(
                    f'⚠ D9 持股族群「{sec}」（占 {pct*100:.0f}%）20日落後大盤 '
                    f'{abs(lead)*100:.1f}%，建議優先檢視該族群是否減倉'
                )
            elif lead < -0.02:
                result['warnings'].append(
                    f'D9 族群「{sec}」（占 {pct*100:.0f}%）20日相對弱勢 '
                    f'{abs(lead)*100:.1f}%，留意是否轉弱'
                )

    # 大盤空頭 + D4 shock 加重警告
    stress = _market_stress_level(market_state)
    if stress == 'shock':
        result['warnings'].append(
            f'🚨 大盤跌破 MA60 > {abs(MARKET_DRAWDOWN_THRESHOLD)*100:.0f}%（shock）：'
            f'所有 Chandelier 停損縮緊至 {CHANDELIER_ATR_MULT_SHOCK}×ATR、'
            f'全面禁止加碼，建議降倉至 50%')
    elif not result['bullish']:
        result['warnings'].append(
            f'⚠ 大盤空頭：Chandelier 停損縮緊至 {CHANDELIER_ATR_MULT_BEAR}×ATR，禁止加碼')

    # D6：附帶當前風險預算
    budget = _market_risk_budget(stress)
    result['risk_budget'] = {
        'stress': stress,
        'desc': budget['desc'],
        'max_total_exposure': budget['max_total_exposure'],
        'max_per_position': budget['max_per_position'],
        'position_risk_mult': budget['position_risk_mult'],
    }
    # D6：總曝險超出預算 → 警告
    exposure_pct = 0
    if total_cost > 0:
        acc = _get_account_size()
        exposure_pct = total_cost / acc if acc > 0 else 0
        if exposure_pct > budget['max_total_exposure']:
            result['warnings'].append(
                f'🚨 D6 總曝險 {exposure_pct*100:.0f}% 超過 {stress} 階段預算 '
                f'{budget["max_total_exposure"]*100:.0f}%，建議減倉至預算內'
            )

    # D20：避險建議 — 空頭 / shock 市場 + 高曝險時提示反向 ETF / 減倉
    hedge_payload = None
    if total_cost > 0:
        bearish = not result['bullish']
        high_exposure = exposure_pct >= 0.30
        if (stress == 'shock' or (bearish and high_exposure)):
            # 動態計算 hedge ratio：shock 50%，bear 30%
            hedge_ratio = 0.50 if stress == 'shock' else 0.30
            hedge_value = total_market * hedge_ratio
            # 反向 ETF：00632R 元大台灣50反一（對沖大盤）
            instrument = '00632R 元大台灣50反 / 期指空單'
            # 估算所需股數（假設 00632R 約 5 元 → 1 張 = 5000 NTD）
            est_lots = max(1, int(hedge_value / 5000))
            hedge_payload = {
                'enabled': True,
                'stress': stress,
                'ratio': round(hedge_ratio, 2),
                'value': round(hedge_value, 0),
                'instrument': instrument,
                'est_lots_00632R': est_lots,
                'note': (f'{("shock" if stress == "shock" else "空頭")} 市場且持股市值 '
                         f'{total_market:,.0f}，建議避險約 {hedge_value:,.0f}'
                         f'（{hedge_ratio*100:.0f}%）'),
            }
            result['warnings'].append(
                f'🛡 D20 建議避險：{stress} 市場 + 持股 {total_market:,.0f}，'
                f'可佈空 {instrument} 約 {est_lots} 張（hedge ratio {hedge_ratio*100:.0f}%）'
            )
    result['hedge_suggestion'] = hedge_payload
    return result


# ===================== 互動式 HTML 報告 =====================

def _df_to_html_table(df):
    """將 DataFrame 轉為 styled HTML table 字串。
    C13：把 None / NaN / '—' 統一渲染為 .na-cell（灰色），讓使用者明確區分「資料缺失」與「真實 0」。"""
    if df is None or df.empty:
        return '<p class="empty">無符合</p>'
    df_render = df.copy()
    # 把 NaN / None 顯示為 '—'
    for col in df_render.columns:
        df_render[col] = df_render[col].apply(
            lambda v: '—' if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        )
    html = df_render.to_html(index=False, classes='data-table', border=0,
                             float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x,
                             escape=True)
    # 把單獨的 '—' 細胞包成 .na-cell 以套用灰色樣式
    html = html.replace('<td>—</td>', '<td class="na-cell" title="此檔無對應模型/資料">—</td>')
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


def _build_holdings_correlation_html(holdings_df):
    """D21：對持股計算近 20 日報酬序列的 Pearson 相關矩陣，產生 Plotly heatmap。
    高度相關（≥ 0.7）的持股 → 實質單一暴險，標紅警示。"""
    if holdings_df is None or holdings_df.empty:
        return ''
    sids = [str(s).strip() for s in holdings_df['證券代號'].astype(str).tolist()
            if str(s).strip()]
    sids = list(dict.fromkeys(sids))   # 去重保留順序
    if len(sids) < 2:
        return ''
    # 收集 close 序列 + 計算日報酬
    series = {}
    for sid in sids:
        t = _technicals_cache.get(sid) or {}
        hist = t.get('close_history_30') or []
        if len(hist) < 11:
            continue
        try:
            arr = np.array(hist[-21:], dtype=float)
        except Exception:
            continue
        if (arr <= 0).any() or len(arr) < 11:
            continue
        rets = np.diff(arr) / arr[:-1]
        series[sid] = rets
    if len(series) < 2:
        return ''
    # 對齊長度
    min_len = min(len(v) for v in series.values())
    series = {k: v[-min_len:] for k, v in series.items()}
    sid_list = list(series.keys())
    n = len(sid_list)
    # 帶上股票名稱
    name_map = {}
    if '證券名稱' in holdings_df.columns:
        for _, row in holdings_df.iterrows():
            sid = str(row.get('證券代號', '')).strip()
            nm = row.get('證券名稱')
            if sid and nm and not pd.isna(nm):
                name_map[sid] = str(nm)
    labels = [(f"{s} {name_map.get(s, '')}").strip() for s in sid_list]

    matrix = []
    high_pairs = []
    for i, ai in enumerate(sid_list):
        row = []
        for j, bj in enumerate(sid_list):
            if i == j:
                row.append(1.0)
                continue
            try:
                a = series[ai]; b = series[bj]
                if a.std() < 1e-12 or b.std() < 1e-12:
                    row.append(0.0)
                else:
                    c = float(np.corrcoef(a, b)[0, 1])
                    if not np.isfinite(c):
                        c = 0.0
                    row.append(round(c, 3))
            except Exception:
                row.append(0.0)
        matrix.append(row)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] >= 0.7:
                high_pairs.append((labels[i], labels[j], matrix[i][j]))
    high_pairs.sort(key=lambda x: x[2], reverse=True)

    div_id = 'holdingsCorrHeatmap'
    fig_data = json.dumps([{
        'type': 'heatmap',
        'z': matrix, 'x': labels, 'y': labels,
        'colorscale': [
            [0.0, '#3a7a55'], [0.4, '#1a2a4a'],
            [0.6, '#0a0e1a'], [0.7, '#9c5'],
            [0.85, '#e8a'], [1.0, '#c44'],
        ],
        'zmin': -1, 'zmax': 1,
        'showscale': True,
        'text': [[f"{v:.2f}" for v in row] for row in matrix],
        'texttemplate': '%{text}',
        'textfont': {'size': 10, 'color': '#e0e0e0'},
        'hovertemplate': '%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>',
    }], ensure_ascii=False)
    layout = json.dumps({
        'title': {'text': 'D21 持股 20 日報酬相關性',
                  'font': {'color': '#e0e0e0', 'size': 14}},
        'plot_bgcolor': '#0a0e1a', 'paper_bgcolor': '#0a0e1a',
        'font': {'color': '#cde'},
        'xaxis': {'tickangle': -30, 'gridcolor': '#1a223a'},
        'yaxis': {'autorange': 'reversed', 'gridcolor': '#1a223a'},
        'margin': {'t': 50, 'b': 110, 'l': 110, 'r': 50},
        'height': max(320, 28 * n + 130),
    }, ensure_ascii=False)
    pair_html = ''
    if high_pairs:
        items = ' / '.join([f'<b>{a} ↔ {b}</b> r={v:.2f}' for a, b, v in high_pairs[:5]])
        pair_html = (f'<p style="color:#fc8; font-size:11.5px; margin:6px 0 0;">'
                     f'⚠ 高度相關（r≥0.7）配對：{items} — 實質單一暴險，建議分散到低相關標的。</p>')

    return f'''
    <div style="margin-top:14px; background:var(--bg-card); padding:14px 18px;
                border-radius:8px; border-left:4px solid #5af;">
        <div style="font-size:14px; color:var(--text-primary); font-weight:600;">
            🔗 D21 持股相關性
        </div>
        <p style="color:var(--text-secondary); font-size:11.5px; margin:6px 0;">
            Pearson 相關（近 20 日日報酬）。對角線恆 1。
            r ≥ 0.7 表示「同步漲跌」極強，下行風險不會被分散。
        </p>
        <div id="{div_id}"></div>
        {pair_html}
        <script>
            (function(){{
                if (typeof Plotly === 'undefined') return;
                Plotly.newPlot('{div_id}', {fig_data}, {layout},
                               {{displayModeBar:false, responsive:true}});
            }})();
        </script>
    </div>'''


def _build_holdings_section(sell_alerts, initial_holdings_df, all_data,
                            portfolio_risk=None,
                            deep_preds=None, breakout_preds=None, mc_probs=None,
                            market_state=None,
                            ranking_df=None):
    """產生『我的持股』互動式 section（純前端 localStorage）。
    第三階段強化：初始/當前停損、分層狀態、健檢分數、加碼訊號、投組風險卡。
    D18：新增模型訊號顯示（上漲機率 / 突破機率 / MC P(+10%)）。"""
    # 方案 B：持股保底補抓，避免 batch 下載漏掉導致現價遺失
    _retry_missing_holdings_data(initial_holdings_df)

    # 蒐集股票資料：代號 → {close, atr, name, rs, ma_status, sell*, health, ...}
    stock_data = {}
    acc_size = _get_account_size()
    deep_preds = deep_preds or {}
    breakout_preds = breakout_preds or {}
    mc_probs = mc_probs or {}
    for sid, t in _technicals_cache.items():
        close = _as_num(t.get('close'))
        if close is None:
            continue
        atr = _as_num(t.get('atr'))
        entry = {
            'close': round(close, 2),
            'atr': round(atr, 3) if atr else None,
            'rs': round(t['rs_vs_taiex'], 2) if t.get('rs_vs_taiex') is not None else None,
            'ma_status': t.get('ma_status') or None,
            'rsi': round(t['rsi'], 1) if t.get('rsi') is not None else None,
            'sugShares': _suggest_position_size(close, atr, acc_size, None,
                                                market_state=market_state),
        }
        # D18：模型訊號（若該檔有預測）
        dp = deep_preds.get(sid)
        upper_pred = None
        pred20 = None
        if dp is not None:
            up = dp.get('up_prob')
            if up is not None and not pd.isna(up):
                entry['upProb'] = round(float(up), 3)
            pred20 = dp.get('pred_20d')
            if pred20 is not None and not pd.isna(pred20):
                entry['pred20'] = round(float(pred20), 3)
            upper_pred = dp.get('upper_price')
            if upper_pred is not None and pd.isna(upper_pred):
                upper_pred = None
        bp = breakout_preds.get(sid)
        if bp is not None and not pd.isna(bp):
            entry['breakoutProb'] = round(float(bp), 3)
        mc = mc_probs.get(sid)
        if mc is not None:
            p10 = mc.get('p_up_10')
            if p10 is not None and not pd.isna(p10):
                entry['mcUp10'] = round(float(p10), 3)

        # D10：動態目標價建議
        try:
            cands = []
            if atr is not None and atr > 0:
                cands.append(round(close + 3 * atr, 2))
            if upper_pred is not None and upper_pred > close:
                cands.append(round(float(upper_pred), 2))
            h30 = t.get('high_30d')
            if h30 is not None and h30 > close:
                cands.append(round(h30 * 1.05, 2))
            if pred20 is not None and pred20 > 0:
                cands.append(round(close * (1 + float(pred20)), 2))
            if cands:
                entry['suggestedTarget'] = max(cands)
        except Exception:
            pass

        # D11：加碼點訊號（pullback to MA20 zone, RSI 再轉強）
        # 條件：MA20 上揚 + close 在 MA20 ± 2% + RSI 從低位反彈（45~60）
        try:
            ma20 = t.get('ma_20')
            ma20_sl = t.get('ma_20_slope')
            rsi = t.get('rsi')
            if (ma20 and ma20_sl is not None and ma20_sl > 0
                    and rsi is not None and 45 <= rsi <= 60
                    and close >= ma20 * 0.98 and close <= ma20 * 1.02):
                vr = t.get('volume_ratio')
                vol_ok = vr is not None and vr >= 0.8
                entry['addZone'] = {
                    'type': 'pullback',
                    'ma20': round(ma20, 2),
                    'rsi': round(rsi, 1),
                    'vol_ok': bool(vol_ok),
                    'note': f'回測 MA20 ({ma20:.2f}) + RSI {rsi:.0f}，符合加碼條件',
                }
        except Exception:
            pass

        # D13：RSI 高位 + 量縮 → 出貨警示（高機率短線回檔）
        try:
            rsi_d13 = t.get('rsi')
            vol_ratio_d13 = t.get('volume_ratio')
            macd_hist_d13 = t.get('macd_hist')
            close_d13 = close
            high_30 = t.get('high_30d')
            ma5_d13 = t.get('ma_5')
            distrib_signals = []
            if rsi_d13 is not None and rsi_d13 >= 78:
                distrib_signals.append(f'RSI {rsi_d13:.0f}（過熱）')
            if vol_ratio_d13 is not None and vol_ratio_d13 < 0.7 and rsi_d13 and rsi_d13 >= 70:
                distrib_signals.append(f'量比 {vol_ratio_d13:.2f}（價漲量縮）')
            if (macd_hist_d13 is not None and macd_hist_d13 < 0
                    and rsi_d13 and rsi_d13 >= 70):
                distrib_signals.append('MACD 柱轉負')
            if (high_30 and close_d13 and close_d13 < high_30 * 0.97
                    and ma5_d13 and close_d13 < ma5_d13):
                distrib_signals.append('已跌破 5MA + 距 30D 高 −3%')
            if len(distrib_signals) >= 2:
                entry['distribAlert'] = {
                    'level': 'warning' if len(distrib_signals) == 2 else 'danger',
                    'signals': distrib_signals,
                    'note': '出貨訊號，建議分批減碼或縮停損',
                }
        except Exception:
            pass

        # D12：個股回撤資料（前端用持股 entry_date 後最高與當前比）
        # 這裡只供應「自 60 日高的回撤」當作通用 reference
        try:
            h60 = t.get('high_30d')   # 名稱沿用，但實為 30 日高
            if h60 is not None and h60 > 0:
                dd = (close - h60) / h60
                entry['ddFrom30dHigh'] = round(dd, 4)
        except Exception:
            pass

        stock_data[sid] = entry

    # D14：持股換股建議（RS 明顯弱於排名 Top10）
    swap_pool = []
    held_ids = set()
    if initial_holdings_df is not None and not initial_holdings_df.empty:
        held_ids = {str(s).strip() for s in initial_holdings_df['證券代號'].astype(str).tolist()}
    if ranking_df is not None and not ranking_df.empty and held_ids:
        try:
            rk = ranking_df.head(10).copy()
            for _, rk_row in rk.iterrows():
                cand_sid = str(rk_row.get('證券代號', '')).strip()
                if not cand_sid or cand_sid in held_ids:
                    continue
                cand_t = _technicals_cache.get(cand_sid) or {}
                cand_rs = cand_t.get('rs_vs_taiex')
                if cand_rs is None:
                    continue
                swap_pool.append({
                    'stock_id': cand_sid,
                    'name': str(rk_row.get('證券名稱', '') or ''),
                    'rs': round(float(cand_rs), 2),
                    'rank': int(rk_row.get('排名', 0) or 0),
                    'score': round(float(rk_row.get('三層融合')
                                          or rk_row.get('規則分數') or 0.0), 1),
                })
            for sid in list(held_ids):
                t = _technicals_cache.get(sid) or {}
                held_rs = t.get('rs_vs_taiex')
                if held_rs is None or not swap_pool:
                    continue
                weak_candidates = [c for c in swap_pool if c['rs'] - held_rs >= 5.0]
                if not weak_candidates:
                    continue
                # D23：取 top-3 候選（按 RS 排序），並計算各自 reason
                weak_candidates.sort(key=lambda c: c['rs'], reverse=True)
                top3 = weak_candidates[:3]
                cand_list = []
                for cand in top3:
                    cand_t = _technicals_cache.get(cand['stock_id']) or {}
                    reasons = []
                    reasons.append(f"RS+{cand['rs']:.1f}（差 {cand['rs'] - held_rs:+.1f}）")
                    if cand_t.get('ma_status') == '多頭排列':
                        reasons.append('多頭排列')
                    if cand_t.get('rsi') is not None and cand_t['rsi'] < 70:
                        reasons.append(f'RSI {cand_t["rsi"]:.0f} 未超買')
                    cand_list.append({
                        'cand_id': cand['stock_id'],
                        'cand_name': cand['name'],
                        'cand_rs': cand['rs'],
                        'cand_rank': cand['rank'],
                        'cand_score': cand['score'],
                        'rs_gap': round(cand['rs'] - float(held_rs), 2),
                        'reasons': reasons,
                    })
                top_cand = top3[0]
                stock_data.setdefault(sid, {})
                # 主要建議（兼容舊版 D14）
                stock_data[sid]['swapSuggestion'] = {
                    'cand_id': top_cand['stock_id'],
                    'cand_name': top_cand['name'],
                    'cand_rs': top_cand['rs'],
                    'cand_rank': top_cand['rank'],
                    'cand_score': top_cand['score'],
                    'rs_gap': round(top_cand['rs'] - float(held_rs), 2),
                    'note': f"換股提示：{top_cand['stock_id']} RS {top_cand['rs']} 高於本持股 {held_rs:.2f}",
                }
                # D23：完整候選清單
                stock_data[sid]['swapCandidates'] = cand_list
        except Exception as e:
            print(f"  D14 換股建議計算失敗：{e}")

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
    # C5：Kelly fraction 統計
    kelly = _kelly_fraction_from_history()
    # D5：黑名單
    blacklist = get_recent_stoploss_blacklist()

    portfolio_json_obj = {
        'total_cost': portfolio_payload.get('total_cost', 0),
        'total_market': portfolio_payload.get('total_market', 0),
        'total_risk': portfolio_payload.get('total_risk', 0),
        'mtm_pct': portfolio_payload.get('mtm_pct', 0),
        'max_concentration': portfolio_payload.get('max_concentration', 0),
        'warnings': portfolio_payload.get('warnings', []),
        'bullish': bool(portfolio_payload.get('bullish', True)),
        'account_size': acc_size,
        'kelly': kelly,
        'blacklist': blacklist,
        'risk_budget': portfolio_payload.get('risk_budget'),               # D6
        'sector_concentration': portfolio_payload.get('sector_concentration', {}),  # D7
        'sector_returns': portfolio_payload.get('sector_returns', {}),     # D9
        'hedge_suggestion': portfolio_payload.get('hedge_suggestion'),     # D20
    }

    stock_json = json.dumps(stock_data, ensure_ascii=False)
    initial_json = json.dumps(initial_list, ensure_ascii=False)
    portfolio_json = json.dumps(portfolio_json_obj, ensure_ascii=False, default=str)
    has_initial = 'true' if initial_list else 'false'

    strategy_options = ''.join(
        f'<option>{s}</option>' for s in [
            '策略1 外資連續買超', '策略2 投信連續買超', '策略3 三法人共識',
            '策略4 量價齊揚', '策略5 RSI超賣反彈', '策略6 布林收斂',
            '策略7 營收+法人', '策略8 融資/融券', '策略9 MACD金叉',
            '策略10 KD黃金叉', '策略11 Darvas突破', '策略12 價量背離',
            '策略13 相對強度',
            '策略14 RS Line 領漲', '策略15 量縮回測', '策略16 季線首站',
            '策略17 主力建倉代理', '策略18 中期盤整突破',
            '策略19 Pocket Pivot', '策略20 營收驚喜後進',
            '策略21 三聯共振',
            '策略23 內包日突破', '策略24 跳空缺口突破',
            '策略25 籌碼結構', '策略26 三重底', '策略27 融資斷頭反彈',
            '策略28 Cup with Handle',
            '策略29 Pivot Breakout',
            '策略31 法人三連買整理突破',
            '策略33 V-shape反轉',
            '策略34 軋空動能突破',
            '策略36 吸籌型（連跌量縮+法人偷買）',
            '策略37 RS領頭（大盤盤整）',
            '策略38 跌破MA20反吃（短打）',
            '策略39 外資+投信皆連2日買超（強共識）',
            '策略40 MA20/60黃金交叉（經典）',
            '自訂'
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

/* 編輯模式 banner */
#holdings .form-mode-banner {
    background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
    color:#fff; padding:8px 14px; border-radius:6px;
    font-size:13px; margin-bottom:12px;
    display:flex; align-items:center; gap:8px;
}
#holdings .form-mode-banner.edit-mode {
    background: linear-gradient(135deg, #ef6c00 0%, #e65100 100%);
}

/* 操作鈕：編輯（藍）/ 刪除（紅）並列 */
#holdings .btn-edit { background:#42a5f5; color:#fff; margin-right:4px; }
#holdings .btn-edit:hover { background:#1976d2; }
#holdings .ops-cell { white-space:nowrap; }

/* 行內可編輯欄位（雙擊變輸入） */
#holdings .editable {
    cursor:pointer; border-bottom:1px dashed #555;
    padding:1px 4px; border-radius:3px;
}
#holdings .editable:hover { background:rgba(66,165,245,.18); border-bottom-color:#42a5f5; }
#holdings .editable.is-edit { padding:0; border-bottom:none; }
#holdings .editable input {
    width:80px; padding:3px 5px; background:#0f0f1a; color:#42a5f5;
    border:1px solid #42a5f5; border-radius:3px; font-size:12.5px;
    font-family:'Consolas', monospace;
}
#holdings .editable .save-hint {
    font-size:9px; color:#a0a0b0; margin-left:4px;
}

/* 編輯中的 row 高亮 */
#holdings .row-editing { background:rgba(255,152,0,.12) !important; }

/* 垃圾桶 / 復原視窗 */
#holdings .trash-modal {
    position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,.65); z-index:1000;
    display:flex; align-items:center; justify-content:center;
}
#holdings .trash-card {
    background:#1a1a2e; padding:20px 24px; border-radius:10px;
    border:1px solid #2a2a4a; max-width:680px; width:90%;
    max-height:80vh; overflow-y:auto;
}
#holdings .trash-card h3 { margin:0 0 12px 0; color:#ffb74d; font-size:15px; }
#holdings .trash-card .trash-row {
    display:grid; grid-template-columns:1fr 80px 90px 80px 80px;
    gap:10px; padding:8px 0; border-bottom:1px solid #2a2a4a;
    align-items:center; font-size:12.5px;
}
#holdings .trash-card .trash-row:last-child { border-bottom:none; }
#holdings .trash-card .trash-row .trash-info { color:#e0e0e0; }
#holdings .trash-card .trash-row .trash-info small { color:#a0a0b0; display:block; }
#holdings .btn-restore { background:#26a69a; color:#fff; }
#holdings .btn-restore:hover { background:#00897b; }

/* 「加入持股後」的浮動 toast */
#holdings .toast {
    position:fixed; bottom:20px; right:20px; z-index:1100;
    background:#1a1a2e; color:#e0e0e0; padding:12px 18px;
    border-left:4px solid #26a69a; border-radius:6px;
    box-shadow:0 4px 12px rgba(0,0,0,.45); font-size:13px;
    display:flex; align-items:center; gap:10px;
    animation:toast-in .25s ease-out;
}
#holdings .toast.toast-warn { border-left-color:#ffb74d; }
#holdings .toast.toast-err  { border-left-color:#ef5350; }
@keyframes toast-in { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
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

    __HOLDINGS_CORR_HTML__

    <div class="holdings-toolbar">
        <button class="btn-primary" onclick="openAddHoldingForm()">➕ 新增持股</button>
        <button class="btn-secondary" onclick="showExitSimulation()">📤 D22 模擬平倉</button>
        <button class="btn-secondary" onclick="showStressTest()">⚡ D24 壓力測試</button>
        <button class="btn-secondary" onclick="exportHoldingsCSV()">⬇ 匯出 CSV</button>
        <button class="btn-secondary" onclick="document.getElementById('csv-import-input').click()">⬆ 匯入 CSV</button>
        <input type="file" id="csv-import-input" accept=".csv,text/csv"
               style="display:none" onchange="handleCSVImport(event)">
        <button class="btn-secondary" onclick="loadInitialHoldings()" id="btn-load-initial">📂 從專案 holdings.csv 載入</button>
        <button class="btn-secondary" onclick="showTrash()" id="btn-trash" title="復原近 24 小時刪除的持股">🗂 最近刪除 <span id="trash-count">(0)</span></button>
        <button class="btn-danger" onclick="clearHoldings()">🗑 清空全部</button>
    </div>

    <div id="holdings-summary" class="holdings-summary"></div>

    <div id="holdings-form" class="holdings-form" style="display:none">
        <div id="form-mode-banner" class="form-mode-banner" style="display:none"></div>
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
            <button id="btn-form-submit" class="btn-primary" onclick="submitHoldingForm()">加入</button>
            <button class="btn-secondary" onclick="cancelHoldingForm()">取消</button>
            <span id="form-hint" style="color:#a0a0b0; font-size:12px; align-self:center;"></span>
        </div>
    </div>

    <!-- 復原最近刪除的彈窗 -->
    <div id="trash-modal" class="trash-modal" style="display:none">
        <div class="trash-card">
            <h3>🗂 最近 24 小時刪除的持股</h3>
            <div id="trash-list"></div>
            <div style="margin-top:12px; text-align:right;">
                <button class="btn-secondary" onclick="hideTrash()">關閉</button>
            </div>
        </div>
    </div>

    <div id="stress-modal" class="trash-modal" style="display:none">
        <div class="trash-card" style="max-width:820px;">
            <h3>⚡ D24 壓力測試（市場下跌情境）</h3>
            <p style="color:#a0a0b0; font-size:11.5px; margin:4px 0 12px;">
                假設大盤分別下跌 −5% / −10% / −15%，依個股 RS 估算的 β 值推算持股市值與損失。
                β 預設 1.0；當 PORTFOLIO 內有 rs_vs_taiex 時用 1 + rs/100 近似（不超過 2.5）。
                也會比對「初始停損是否被觸發」。
            </p>
            <div id="stress-list"></div>
            <div style="margin-top:12px; text-align:right;">
                <button class="btn-secondary" onclick="hideStressTest()">關閉</button>
            </div>
        </div>
    </div>

    <div id="exit-sim-modal" class="trash-modal" style="display:none">
        <div class="trash-card" style="max-width:760px;">
            <h3>📤 D22 模擬平倉（以現價立即出場估算）</h3>
            <p style="color:#a0a0b0; font-size:11.5px; margin:4px 0 12px;">
                預設成本：手續費 0.1425% × 0.6 折 + 證交稅 0.3%（賣方）+ 滑價 0.2%。
                可在右上角調整總成本率。
                <span style="float:right;">總成本率：
                    <input id="exit-cost-rate" type="number" step="0.05" value="0.7" style="width:60px; padding:2px 4px;"> %
                    <button class="btn-mini" onclick="renderExitSim()">套用</button>
                </span>
            </p>
            <div id="exit-sim-list"></div>
            <div style="margin-top:12px; text-align:right;">
                <button class="btn-secondary" onclick="hideExitSim()">關閉</button>
            </div>
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
const LS_KEY_TRASH = 'stockHoldings_trash_v1';   // 24 小時保留的垃圾桶
const LS_KEY_AUDIT = 'stockHoldings_audit_v1';   // 變更歷史（最近 50 筆）
const TRASH_TTL_MS = 24 * 60 * 60 * 1000;        // 24 小時
const AUDIT_MAX = 50;
let EDITING_IDX = null;                          // null=新增模式；數字=該索引編輯中

function loadHoldings() {
    let raw = localStorage.getItem(LS_KEY);
    if (raw === null) {
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

// ── 垃圾桶（誤刪保護）──
function loadTrash() {
    try {
        const arr = JSON.parse(localStorage.getItem(LS_KEY_TRASH) || '[]');
        const now = Date.now();
        const fresh = arr.filter(it => (now - (it.deleted_at || 0)) < TRASH_TTL_MS);
        if (fresh.length !== arr.length) {
            localStorage.setItem(LS_KEY_TRASH, JSON.stringify(fresh));
        }
        return fresh;
    } catch { return []; }
}
function pushTrash(holding) {
    const trash = loadTrash();
    trash.unshift({ ...holding, deleted_at: Date.now() });
    localStorage.setItem(LS_KEY_TRASH, JSON.stringify(trash.slice(0, 30)));
    refreshTrashCount();
}
function refreshTrashCount() {
    const el = document.getElementById('trash-count');
    if (el) {
        const n = loadTrash().length;
        el.textContent = `(${n})`;
        el.parentElement.style.opacity = n === 0 ? '0.5' : '1';
    }
}

// ── 變更歷史（簡易 audit log）──
function pushAudit(action, before, after) {
    try {
        const log = JSON.parse(localStorage.getItem(LS_KEY_AUDIT) || '[]');
        log.unshift({ ts: Date.now(), action, before, after });
        localStorage.setItem(LS_KEY_AUDIT, JSON.stringify(log.slice(0, AUDIT_MAX)));
    } catch {}
}

// ── Toast 通知 ──
function showToast(msg, kind) {
    const t = document.createElement('div');
    t.className = 'toast' + (kind === 'warn' ? ' toast-warn' : kind === 'err' ? ' toast-err' : '');
    t.innerHTML = msg;
    document.body.appendChild(t);
    setTimeout(() => { t.style.opacity='0'; t.style.transition='opacity .3s'; }, 2400);
    setTimeout(() => t.remove(), 2800);
}

function openHoldingForm() {
    const el = document.getElementById('holdings-form');
    el.style.display = '';
    const bd = document.getElementById('f-buy-date');
    if (!bd.value) bd.value = new Date().toISOString().slice(0,10);
    el.scrollIntoView({behavior:'smooth', block:'center'});
}
function closeHoldingForm() {
    document.getElementById('holdings-form').style.display = 'none';
}
function toggleHoldingForm() {
    const el = document.getElementById('holdings-form');
    if (el.style.display === 'none' || !el.style.display) openHoldingForm();
    else closeHoldingForm();
}

function autoFillHoldingName() {
    const sid = document.getElementById('f-stock-id').value.trim();
    const data = STOCK_DATA[sid];
    const hint = document.getElementById('form-hint');
    // D5：黑名單檢查
    const bl = (PORTFOLIO.blacklist || {})[sid];
    let blText = '';
    if (bl) {
        const loss = bl.loss_pct != null ? `${(bl.loss_pct*100).toFixed(1)}%` : '—';
        blText = ` <span style="color:#ef5350;">📛 黑名單：${bl.days_ago} 天前停損 ${loss}（${bl.reason}）</span>`;
    }
    if (data) {
        if (data.name && !document.getElementById('f-name').value) {
            document.getElementById('f-name').value = data.name;
        }
        const parts = [];
        if (data.close) parts.push(`現價 ${data.close}`);
        if (data.atr) parts.push(`ATR ${data.atr}`);
        if (data.rs) parts.push(`RS ${data.rs}`);
        if (data.sugShares) {
            const k = PORTFOLIO.kelly || {};
            const pct = k.kelly_q ? `${(k.kelly_q*100).toFixed(2)}%` : '1%';
            parts.push(`建議 ${data.sugShares} 張（C5 動態風險 ${pct}）`);
        }
        hint.innerHTML = parts.join(' | ') + blText;
        const bp = parseFloat(document.getElementById('f-buy-price').value);
        const stpEl = document.getElementById('f-stop');
        if (!stpEl.value && data.atr && bp && !isNaN(bp)) {
            stpEl.value = (bp - 1.5 * data.atr).toFixed(2);
        }
    } else {
        hint.innerHTML = (sid ? '⚠ 找不到此代號資料' : '') + blText;
    }
}

function resetHoldingForm() {
    EDITING_IDX = null;
    ['f-stock-id','f-name','f-buy-price','f-target','f-stop','f-note'].forEach(
        id => document.getElementById(id).value = ''
    );
    document.getElementById('f-shares').value = 1;
    document.getElementById('f-strategy').value = '';
    document.getElementById('f-buy-date').value = '';
    document.getElementById('form-hint').textContent = '';
    document.getElementById('btn-form-submit').textContent = '加入';
    document.getElementById('btn-form-submit').className = 'btn-primary';
    const banner = document.getElementById('form-mode-banner');
    banner.style.display = 'none';
    banner.classList.remove('edit-mode');
}
function openAddHoldingForm() {
    resetHoldingForm();
    const banner = document.getElementById('form-mode-banner');
    banner.style.display = '';
    banner.innerHTML = '<span>📝</span><span>新增模式：填好後按「加入」</span>';
    openHoldingForm();
}
function cancelHoldingForm() {
    resetHoldingForm();
    closeHoldingForm();
}

function startEditHolding(idx) {
    const all = loadHoldings();
    const h = all[idx];
    if (!h) { showToast('⚠ 找不到該筆持股', 'err'); return; }
    EDITING_IDX = idx;
    document.getElementById('f-stock-id').value  = h.stock_id || '';
    document.getElementById('f-name').value      = h.name || '';
    document.getElementById('f-buy-date').value  = h.buy_date || '';
    document.getElementById('f-buy-price').value = h.buy_price ?? '';
    document.getElementById('f-shares').value    = h.shares ?? 1;
    document.getElementById('f-target').value    = h.target_price ?? '';
    document.getElementById('f-stop').value      = h.stop_loss ?? '';
    document.getElementById('f-strategy').value  = h.strategy || '';
    document.getElementById('f-note').value      = h.note || '';
    const submit = document.getElementById('btn-form-submit');
    submit.textContent = '💾 儲存修改';
    submit.className = 'btn-primary';
    const banner = document.getElementById('form-mode-banner');
    banner.style.display = '';
    banner.classList.add('edit-mode');
    banner.innerHTML = `<span>✏️</span><span>編輯模式：${h.stock_id} ${h.name||''}（取消可放棄）</span>`;
    openHoldingForm();
    autoFillHoldingName();
}

function submitHoldingForm() {
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
    // D5：若是黑名單股票，先二次確認
    const bl = (PORTFOLIO.blacklist || {})[h.stock_id];
    if (bl && EDITING_IDX === null) {
        const loss = bl.loss_pct != null ? `${(bl.loss_pct*100).toFixed(1)}%` : '';
        if (!confirm(`⚠ ${h.stock_id} 在最近 ${bl.days_ago} 天剛因「${bl.reason}」停損 ${loss}\n\n通常建議至少等 30 天再進場（避免被反覆洗）。\n仍要新增嗎？`)) {
            return;
        }
    }
    if (!h.name && STOCK_DATA[h.stock_id] && STOCK_DATA[h.stock_id].name) {
        h.name = STOCK_DATA[h.stock_id].name;
    }
    const all = loadHoldings();
    if (EDITING_IDX !== null && all[EDITING_IDX]) {
        const old = all[EDITING_IDX];
        // 保留風險管理動態欄位
        const keep = ['max_high','initial_atr','initial_stop','tier_status'];
        keep.forEach(k => { if (old[k] != null) h[k] = old[k]; });
        // 若買進價/停損被改動，提示是否重置
        const priceChanged = (old.buy_price !== h.buy_price);
        const stopChanged = (old.stop_loss !== h.stop_loss);
        if (priceChanged || stopChanged) {
            if (confirm(`買進價或停損已變更，是否重置「入場後最高 / 初始ATR / 初始停損 / 分層狀態」？\n（建議：是 — 讓系統重新計算）`)) {
                delete h.max_high; delete h.initial_atr;
                delete h.initial_stop; delete h.tier_status;
            }
        }
        all[EDITING_IDX] = h;
        pushAudit('edit', old, h);
        saveHoldings(all);
        showToast(`✅ 已更新 ${h.stock_id} ${h.name||''}`);
    } else {
        all.push(h);
        pushAudit('add', null, h);
        saveHoldings(all);
        showToast(`✅ 已新增 ${h.stock_id} ${h.name||''}`);
    }
    resetHoldingForm();
    closeHoldingForm();
}
// 相容舊命名：addHolding 沿用 submitHoldingForm（避免外部 inline onclick 失效）
function addHolding() { submitHoldingForm(); }

function deleteHolding(idx) {
    const all = loadHoldings();
    const h = all[idx];
    if (!h) return;
    if (!confirm(`刪除 ${h.stock_id} ${h.name||''}？\n（24 小時內可從「🗂 最近刪除」復原）`)) return;
    all.splice(idx, 1);
    pushTrash(h);
    pushAudit('delete', h, null);
    saveHoldings(all);
    showToast(`🗑 已刪除 ${h.stock_id}（24 小時內可復原）`, 'warn');
}

// ── 行內快速調整：雙擊 target / stop / shares / note 直接編輯 ──
function inlineEdit(idx, field, displaySpan) {
    const all = loadHoldings();
    const h = all[idx];
    if (!h) return;
    const cur = h[field];
    const isNum = ['target_price','stop_loss','shares','buy_price'].includes(field);
    const wrap = displaySpan;
    wrap.classList.add('is-edit');
    const orig = wrap.innerHTML;
    const initVal = (cur == null ? '' : cur);
    wrap.innerHTML =
        `<input type="${isNum?'number':'text'}" step="0.01" value="${initVal}">` +
        `<span class="save-hint">Enter 存 / Esc 取消</span>`;
    const inp = wrap.querySelector('input');
    inp.focus(); inp.select();
    const finish = (commit) => {
        if (commit) {
            const raw = inp.value.trim();
            const newVal = isNum ? (raw === '' ? null : parseFloat(raw)) : raw;
            if (isNum && raw !== '' && (newVal == null || isNaN(newVal))) {
                showToast('⚠ 數值無效', 'err'); return;
            }
            const before = { ...h };
            h[field] = newVal;
            all[idx] = h;
            pushAudit('inline', before, h);
            saveHoldings(all);
            showToast(`✅ ${h.stock_id} ${field} 已更新`);
        } else {
            wrap.classList.remove('is-edit');
            wrap.innerHTML = orig;
        }
    };
    inp.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); finish(true); }
        else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    });
    inp.addEventListener('blur', () => finish(true));
}

// ── 垃圾桶視窗 ──
function showTrash() {
    const trash = loadTrash();
    const list = document.getElementById('trash-list');
    if (trash.length === 0) {
        list.innerHTML = '<p style="color:#a0a0b0; font-size:13px;">無最近刪除紀錄。（保留 24 小時）</p>';
    } else {
        list.innerHTML = trash.map((t, i) => {
            const ago = Math.round((Date.now() - t.deleted_at) / 60000);
            const remain = Math.max(0, Math.round((TRASH_TTL_MS - (Date.now() - t.deleted_at)) / 3600000));
            return `<div class="trash-row">
                <div class="trash-info"><b>${t.stock_id}</b> ${t.name||''}
                    <small>買進 ${t.buy_date||'-'} @ ${t.buy_price ?? '-'}　${ago}分鐘前刪除（剩 ${remain}h）</small></div>
                <div>${t.shares||1}張</div>
                <div>停損 ${t.stop_loss ?? '-'}</div>
                <div>目標 ${t.target_price ?? '-'}</div>
                <div><button class="btn-mini btn-restore" onclick="restoreFromTrash(${i})">↩ 復原</button></div>
            </div>`;
        }).join('');
    }
    document.getElementById('trash-modal').style.display = 'flex';
}
function hideTrash() { document.getElementById('trash-modal').style.display = 'none'; }

function showStressTest() {
    document.getElementById('stress-modal').style.display = 'flex';
    renderStressTest();
}
function hideStressTest() { document.getElementById('stress-modal').style.display = 'none'; }

function renderStressTest() {
    const list = document.getElementById('stress-list');
    if (!list) return;
    const holdings = JSON.parse(localStorage.getItem(LS_KEY) || '[]');
    if (holdings.length === 0) {
        list.innerHTML = '<p class="empty">尚無持股可壓力測試。</p>';
        return;
    }
    const scenarios = [-0.05, -0.10, -0.15];
    let totals = { now: 0, s5: 0, s10: 0, s15: 0, riskHits: 0, stopHits: [0, 0, 0] };
    const rows = holdings.map(h => {
        const data = STOCK_DATA[h.stock_id] || {};
        const live = (data.close != null) ? Number(data.close) : null;
        const shares = h.shares || 1;
        if (!live || live <= 0 || !h.buy_price) {
            return `<tr><td>${h.stock_id}</td><td>${h.name||'-'}</td><td colspan="9" style="color:#888;">無現價，跳過</td></tr>`;
        }
        // β 估算：RS=+5% → β=1.05；RS=-5% → β=0.95；clip 0.4 ~ 2.5
        let beta = 1.0;
        if (data.rs != null) beta = Math.max(0.4, Math.min(2.5, 1 + Number(data.rs) / 100));
        const nowMkt = live * shares * 1000;
        totals.now += nowMkt;
        const cells = scenarios.map((shock, idx) => {
            const stockShock = beta * shock;     // 個股下跌幅
            const newPrice = live * (1 + stockShock);
            const newMkt = newPrice * shares * 1000;
            const lossNT = newMkt - nowMkt;
            // 是否跌破初始停損
            let stopBreak = '';
            if (h.initial_stop && newPrice < h.initial_stop) {
                stopBreak = ` 🛑`;
                totals.stopHits[idx]++;
            }
            const cls = lossNT < -1000 ? 'loss' : '';
            if (idx === 0) totals.s5 += newMkt;
            if (idx === 1) totals.s10 += newMkt;
            if (idx === 2) totals.s15 += newMkt;
            return `<td class="${cls}">${newPrice.toFixed(1)}<br><span style="font-size:10px;">(${(stockShock*100).toFixed(1)}%)</span>${stopBreak}</td>
                    <td class="${cls}">${Math.round(lossNT).toLocaleString()}</td>`;
        }).join('');
        return `<tr>
            <td>${h.stock_id}</td>
            <td>${h.name||'-'}</td>
            <td>${shares}</td>
            <td>${live.toFixed(2)}</td>
            <td title="β = 1 + rs/100, clip [0.4, 2.5]">${beta.toFixed(2)}</td>
            ${cells}
        </tr>`;
    }).join('');
    const sumLoss = (mkt) => mkt - totals.now;
    const sumPct = (mkt) => totals.now > 0 ? (sumLoss(mkt) / totals.now * 100).toFixed(2) : '0';
    list.innerHTML = `
        <div style="overflow-x:auto;">
        <table class="data-table holdings-table" style="font-size:12px;">
            <thead><tr>
                <th rowspan="2">代號</th><th rowspan="2">名稱</th><th rowspan="2">張數</th>
                <th rowspan="2">現價</th><th rowspan="2">β</th>
                <th colspan="2" style="background:#1a1a2e;">大盤 −5%</th>
                <th colspan="2" style="background:#1a0d0d;">大盤 −10%</th>
                <th colspan="2" style="background:#2a0a0a;">大盤 −15%</th>
            </tr><tr>
                <th>新價</th><th>損失</th>
                <th>新價</th><th>損失</th>
                <th>新價</th><th>損失</th>
            </tr></thead>
            <tbody>${rows}</tbody>
            <tfoot><tr style="background:#0d1118; font-weight:600;">
                <td colspan="5" style="text-align:right;">合計損失</td>
                <td colspan="2" class="loss">${Math.round(sumLoss(totals.s5)).toLocaleString()}<br>(${sumPct(totals.s5)}%)</td>
                <td colspan="2" class="loss">${Math.round(sumLoss(totals.s10)).toLocaleString()}<br>(${sumPct(totals.s10)}%)</td>
                <td colspan="2" class="loss">${Math.round(sumLoss(totals.s15)).toLocaleString()}<br>(${sumPct(totals.s15)}%)</td>
            </tr><tr style="background:#0d1118;">
                <td colspan="5" style="text-align:right;">觸發初始停損檔數</td>
                <td colspan="2">${totals.stopHits[0]}</td>
                <td colspan="2">${totals.stopHits[1]}</td>
                <td colspan="2">${totals.stopHits[2]}</td>
            </tr></tfoot>
        </table>
        </div>
        <p style="color:#a0a0b0; font-size:11.5px; margin-top:10px;">
            🛑 標記表示「個股新價已跌破設定的初始停損」— 該情境下會被強制掃出場。
        </p>`;
}

function showExitSimulation() {
    document.getElementById('exit-sim-modal').style.display = 'flex';
    renderExitSim();
}
function hideExitSim() { document.getElementById('exit-sim-modal').style.display = 'none'; }

function renderExitSim() {
    const list = document.getElementById('exit-sim-list');
    if (!list) return;
    const holdings = JSON.parse(localStorage.getItem(LS_KEY) || '[]');
    if (holdings.length === 0) {
        list.innerHTML = '<p class="empty">尚無持股可模擬。</p>';
        return;
    }
    const rateInput = document.getElementById('exit-cost-rate');
    const costRate = (rateInput && parseFloat(rateInput.value) > 0)
        ? parseFloat(rateInput.value) / 100 : 0.007;
    let totalCost = 0, totalProceeds = 0, totalFees = 0;
    const rows = holdings.map(h => {
        const data = STOCK_DATA[h.stock_id] || {};
        const live = (data.close != null) ? Number(data.close) : null;
        const shares = h.shares || 1;
        const cost = (h.buy_price || 0) * shares * 1000;
        if (!live || live <= 0 || !h.buy_price) {
            return `<tr><td>${h.stock_id}</td><td>${h.name||'-'}</td><td>${shares}</td>
                <td>${h.buy_price||'-'}</td><td colspan="5" style="color:#888;">無現價，跳過</td></tr>`;
        }
        const grossProceeds = live * shares * 1000;
        const fees = grossProceeds * costRate;
        const netProceeds = grossProceeds - fees;
        const pnl = netProceeds - cost;
        const pct = cost > 0 ? (pnl / cost * 100) : 0;
        totalCost += cost; totalProceeds += netProceeds; totalFees += fees;
        const cls = pnl >= 0 ? 'gain' : 'loss';
        return `<tr>
            <td>${h.stock_id}</td>
            <td>${h.name||'-'}</td>
            <td>${shares}</td>
            <td>${h.buy_price.toFixed(2)}</td>
            <td>${live.toFixed(2)}</td>
            <td>${Math.round(grossProceeds).toLocaleString()}</td>
            <td>${Math.round(fees).toLocaleString()}</td>
            <td class="${cls}">${(pnl>=0?'+':'')}${Math.round(pnl).toLocaleString()}</td>
            <td class="${cls}">${(pct>=0?'+':'')}${pct.toFixed(2)}%</td>
        </tr>`;
    }).join('');
    const totalPnl = totalProceeds - totalCost;
    const totalPct = totalCost > 0 ? (totalPnl / totalCost * 100) : 0;
    const totalCls = totalPnl >= 0 ? 'gain' : 'loss';
    list.innerHTML = `
        <div style="overflow-x:auto;">
        <table class="data-table holdings-table" style="font-size:12.5px;">
            <thead><tr>
                <th>代號</th><th>名稱</th><th>張數</th>
                <th>買價</th><th>現價</th><th>毛收</th>
                <th>成本扣</th><th>淨損益</th><th>%</th>
            </tr></thead>
            <tbody>${rows}</tbody>
            <tfoot><tr style="background:#0d1118;">
                <td colspan="5" style="text-align:right; font-weight:600;">合計</td>
                <td>${Math.round(totalProceeds + totalFees).toLocaleString()}</td>
                <td>${Math.round(totalFees).toLocaleString()}</td>
                <td class="${totalCls}" style="font-weight:600;">${(totalPnl>=0?'+':'')}${Math.round(totalPnl).toLocaleString()}</td>
                <td class="${totalCls}" style="font-weight:600;">${(totalPct>=0?'+':'')}${totalPct.toFixed(2)}%</td>
            </tr></tfoot>
        </table>
        </div>
        <p style="color:#a0a0b0; font-size:11.5px; margin-top:10px;">
            ⚠ 此為理論平倉值，不含跳空、流動性不足造成的滑價。實際成交需按市場深度評估。
        </p>`;
}
function restoreFromTrash(trashIdx) {
    const trash = loadTrash();
    const item = trash[trashIdx];
    if (!item) return;
    const restored = { ...item };
    delete restored.deleted_at;
    const all = loadHoldings();
    all.push(restored);
    saveHoldings(all);
    trash.splice(trashIdx, 1);
    localStorage.setItem(LS_KEY_TRASH, JSON.stringify(trash));
    refreshTrashCount();
    showTrash();
    showToast(`↩ 已復原 ${restored.stock_id}`);
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

function _d10TargetHintCell(h, data) {
    // D10：建議調整目標價提示
    const sug = data && data.suggestedTarget;
    if (sug == null) return '<span style="color:#666;">—</span>';
    const cur = h.target_price != null ? Number(h.target_price) : null;
    const sugV = Number(sug);
    if (cur == null) {
        return `<span style="color:#7c8;" title="尚未設定目標價，建議參考此值">建議 ${sugV.toFixed(2)}</span>`;
    }
    const diff = sugV - cur;
    const pct = cur > 0 ? (diff / cur * 100) : 0;
    if (diff > cur * 0.03) {
        return `<span style="color:#3a7; font-weight:600;" title="模型/技術指標皆指向更高目標">▲ 調至 ${sugV.toFixed(2)} (+${pct.toFixed(0)}%)</span>`;
    } else if (diff < -cur * 0.03) {
        return `<span style="color:#c94;" title="模型認為原目標可能過於樂觀">▼ 建議 ${sugV.toFixed(2)} (${pct.toFixed(0)}%)</span>`;
    } else {
        return `<span style="color:#888;" title="目標價合理區間內">≈ ${sugV.toFixed(2)}</span>`;
    }
}

function _kellyTagHtml() {
    const k = PORTFOLIO.kelly || {};
    if (!k || !k.kelly_q) return '';
    const pct = (k.kelly_q * 100).toFixed(2);
    const wr  = k.win_rate != null ? (k.win_rate * 100).toFixed(0) + '%' : '—';
    const pf  = k.payoff != null ? k.payoff.toFixed(2) + 'x' : '—';
    const src = k.source === 'trade_history'
        ? `（基於 ${k.n} 筆歷史，勝率 ${wr} / 賠率 ${pf}）`
        : '（歷史資料不足，使用預設 1%）';
    return `<div style="font-size:12.5px; color:#a0a0b0; margin-top:8px;">
        💰 <b>C5 動態風險</b>：單筆 ${pct}% × 帳戶 ${src}
    </div>`;
}

function _blacklistTagHtml() {
    const bl = PORTFOLIO.blacklist || {};
    const sids = Object.keys(bl);
    if (sids.length === 0) return '';
    const items = sids.slice(0, 10).map(sid => {
        const info = bl[sid];
        const loss = info.loss_pct != null ? `${(info.loss_pct*100).toFixed(1)}%` : '—';
        return `<span style="display:inline-block; margin:2px 4px; padding:2px 8px; background:#3e2723; color:#ffab91; border-radius:10px; font-size:11.5px;" title="${info.exit_date} 出場 / ${info.reason}">📛 ${sid} (${loss}, ${info.days_ago}d前)</span>`;
    }).join('');
    const more = sids.length > 10 ? `<span style="color:#a0a0b0; font-size:11px;">+${sids.length-10} 檔</span>` : '';
    return `<div style="margin-top:10px; font-size:12.5px;">
        <div style="color:#ffab91; margin-bottom:4px;">📛 <b>D5 黑名單</b>（最近 30 日內停損，已從 ranking / 潛力 / 低估剔除）：</div>
        <div>${items}${more}</div>
    </div>`;
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
    const kellyHtml = _kellyTagHtml();
    const blackHtml = _blacklistTagHtml();
    const budgetHtml = _riskBudgetHtml();   // D6 + D19
    const sectorHtml = _sectorConcHtml();   // D7
    const hedgeHtml = _hedgeSuggestionHtml(); // D20
    if (warnings.length === 0 && PORTFOLIO.bullish) {
        card.innerHTML = `<div class="risk-card">
            <h3>🛡️ 投組風險健檢：✅ 正常</h3>
            <p style="color:#a0a0b0; font-size:12.5px; margin:0;">
                大盤多頭，未偵測到集中度、虧損或風險過大警示。
                帳戶基準 ${accSize.toLocaleString()}，風險上限 ${Math.round(accSize*0.05).toLocaleString()}。
            </p>
            ${budgetHtml}${sectorHtml}${hedgeHtml}${kellyHtml}${blackHtml}
        </div>`;
        return;
    }
    card.innerHTML = `<div class="risk-card ${bearish?'bear':''}">
        <h3>🛡️ 投組風險健檢：${bearish?'⚠ 空頭 + ':''}${warnings.length} 項警示</h3>
        <ul class="warn-list">${warnings.map(w=>`<li>${w}</li>`).join('')}</ul>
        ${budgetHtml}${sectorHtml}${hedgeHtml}${kellyHtml}${blackHtml}
    </div>`;
}

function _hedgeSuggestionHtml() {
    // D20：避險建議卡（空頭 + 高曝險時顯示）
    const h = PORTFOLIO.hedge_suggestion;
    if (!h || !h.enabled) return '';
    const stressZh = { normal: '常態', bear: '空頭', shock: '崩跌' }[h.stress] || h.stress;
    return `<div style="margin-top:8px; padding:8px 12px; background:#1a0e0a; border-left:3px solid #f63; border-radius:4px;">
        <div style="font-size:12.5px; color:#fc8; font-weight:600; margin-bottom:4px;">
            🛡 D20 避險建議（${stressZh}）
        </div>
        <div style="font-size:12px; color:#fde;">
            ${h.note}
        </div>
        <div style="font-size:12px; color:#fbd; margin-top:4px;">
            建議工具：<strong>${h.instrument}</strong>　約 <strong>${h.est_lots_00632R}</strong> 張
            （hedge ratio <strong>${(h.ratio*100).toFixed(0)}%</strong>）
        </div>
        <div style="font-size:10.5px; color:#a76; margin-top:3px;">
            註：避險為短期操作，待大盤回穩或持股降至 30% 以下可逐步平倉。
        </div>
    </div>`;
}

function _riskBudgetHtml() {
    // D6：顯示當前市場壓力等級下的風險預算
    const b = PORTFOLIO.risk_budget;
    if (!b) return '';
    const stressColors = { normal: '#3a7', bear: '#c94', shock: '#c44' };
    const color = stressColors[b.stress] || '#3a7';
    const stressZh = { normal: '常態', bear: '空頭', shock: '崩跌' }[b.stress] || b.stress;

    // D19：依目前持股計算「風險預算用量」進度條
    let usageHtml = '';
    try {
        const holdings = JSON.parse(localStorage.getItem(LS_KEY) || '[]');
        const acct = (window.PORTFOLIO && PORTFOLIO.account_size) || 1_000_000;
        let totalRiskValue = 0;
        let totalExposureValue = 0;
        let perPosMaxPct = 0;
        const overItems = [];
        for (const h of holdings) {
            if (!h.buy_price || !h.shares) continue;
            const cost = h.buy_price * h.shares * 1000;
            totalExposureValue += cost;
            if (h.initial_stop && h.buy_price > h.initial_stop) {
                totalRiskValue += (h.buy_price - h.initial_stop) * h.shares * 1000;
            }
            const posPct = cost / Math.max(acct, 1);
            if (posPct > perPosMaxPct) perPosMaxPct = posPct;
            if (b.max_per_position && posPct > b.max_per_position) {
                overItems.push(`${h.stock_id} ${(posPct*100).toFixed(0)}%`);
            }
        }
        const riskBudgetCash = acct * (b.position_risk_mult * 0.01) * 8;  // 預算 ≈ 倍率 × 1% × 8 檔
        const riskUsage = riskBudgetCash > 0 ? totalRiskValue / riskBudgetCash : 0;
        const expUsage = b.max_total_exposure ? totalExposureValue / (acct * b.max_total_exposure) : 0;
        const perPosUsage = b.max_per_position ? perPosMaxPct / b.max_per_position : 0;

        const bar = (label, ratio, tip) => {
            const pct = Math.max(0, Math.min(150, ratio * 100));
            const cls = pct >= 100 ? '#c44' : (pct >= 80 ? '#e8a' : '#3a7');
            return `<div style="margin-top:5px;" title="${tip}">
                <div style="font-size:10.5px; color:#9ab; display:flex; justify-content:space-between;">
                    <span>${label}</span>
                    <span style="color:${cls}; font-weight:600;">${pct.toFixed(0)}%</span>
                </div>
                <div style="background:#0a0e1a; height:6px; border-radius:3px; margin-top:2px; overflow:hidden;">
                    <div style="width:${Math.min(100, pct)}%; height:100%; background:${cls};"></div>
                </div>
            </div>`;
        };
        usageHtml = `<div style="margin-top:6px;">
            ${bar('總曝險用量', expUsage, `${(expUsage*100).toFixed(0)}% of ${(b.max_total_exposure*100).toFixed(0)}% 上限`)}
            ${bar('單檔最大占比', perPosUsage, `最大單檔 ${(perPosMaxPct*100).toFixed(1)}% / 上限 ${(b.max_per_position*100).toFixed(0)}%`)}
            ${bar('1R 風險用量', riskUsage, `預估 1R 損失 NTD ${Math.round(totalRiskValue).toLocaleString()} / 預算 NTD ${Math.round(riskBudgetCash).toLocaleString()}`)}
        </div>`;
        if (overItems.length > 0) {
            usageHtml += `<div style="margin-top:5px; font-size:11px; color:#fa8;">⚠ 超限單檔：${overItems.slice(0,3).join(', ')}${overItems.length>3?'...':''}</div>`;
        }
    } catch (e) {}

    return `<div style="margin-top:8px; padding:7px 10px; background:#0d1118; border-left:3px solid ${color}; border-radius:4px;">
        <div style="font-size:12px; color:#9ab; margin-bottom:3px;">D6 動態風險預算（${stressZh}）／ D19 用量</div>
        <div style="font-size:12px; color:#cde;">
            單筆風險倍率 <strong>${b.position_risk_mult.toFixed(2)}×</strong> ／
            單檔上限 <strong>${(b.max_per_position*100).toFixed(0)}%</strong> ／
            總曝險上限 <strong>${(b.max_total_exposure*100).toFixed(0)}%</strong>
        </div>
        <div style="font-size:11px; color:#789; margin-top:3px;">${b.desc}</div>
        ${usageHtml}
    </div>`;
}

function _sectorConcHtml() {
    // D7：族群集中度，D9：附帶族群 20d 報酬（與大盤比較）
    const sc = PORTFOLIO.sector_concentration || {};
    const sr = PORTFOLIO.sector_returns || {};
    const entries = Object.entries(sc);
    if (entries.length === 0) return '';
    const sorted = entries.slice(0, 5);
    const items = sorted.map(([s, p]) => {
        const pct = (p * 100).toFixed(1);
        const color = p > 0.5 ? '#c44' : p > 0.4 ? '#c94' : '#3a7';
        const ret = sr[s];
        const retLabel = (ret == null) ? '' :
            ` <span style="color:${ret > 0 ? '#3a7' : '#c44'}; font-size:10.5px;">(20d ${ret >= 0 ? '+' : ''}${(ret * 100).toFixed(1)}%)</span>`;
        return `<span style="display:inline-block; margin:2px 4px; padding:2px 8px; background:#0d1118; border:1px solid ${color}; border-radius:3px; font-size:11.5px; color:#cde;">
            ${s} <strong style="color:${color};">${pct}%</strong>${retLabel}
        </span>`;
    }).join('');
    return `<div style="margin-top:8px;">
        <div style="font-size:12px; color:#9ab; margin-bottom:3px;">D7 族群集中度（含 D9 族群 20 日報酬）</div>
        <div>${items}</div>
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
    // D18：累積 R 值統計
    let rValues = [];
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
                let mult = PORTFOLIO.bullish ? 3.0 : 2.0;
                // D15：依累積漲幅縮緊倍數
                const gain = (maxH - h.buy_price) / h.buy_price;
                if (gain >= 0.30) mult = Math.max(0.9, mult * 0.6);
                else if (gain >= 0.20) mult = Math.max(1.2, mult * 0.75);
                else if (gain >= 0.10) mult = Math.max(1.5, mult * 0.9);
                stops.push(maxH - mult * atr);
                // D16：盈利保護（漲幅 ≥ 15% 啟動緊縮停損）
                if (gain >= 0.15 && hasLive) {
                    stops.push(liveClose - 1.0 * atr);
                }
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
            // D12：自入場後最高 → 現價的回撤
            const peak = (h.max_high && h.max_high > 0) ? Number(h.max_high) : null;
            if (hasLive && peak && peak > h.buy_price && liveClose < peak) {
                const dd = (liveClose - peak) / peak;
                if (dd <= -0.10) {
                    const ddStr = (dd*100).toFixed(1) + '%';
                    const ddCls = dd <= -0.20 ? 'loss' : '';
                    pnlCell += `<div style="font-size:10.5px; margin-top:2px;" title="自入場後最高 ${peak.toFixed(2)} 的回撤" class="${ddCls}">
                        ⚠ DD ${ddStr}
                    </div>`;
                    if (dd <= -0.15) pnlCls = pnlCls || 'loss';
                }
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

        // D18：R 值（profit / risk_per_share）— 衡量「該筆持股實際走多少 R」
        let rValue = null;
        if (h.buy_price && h.initial_stop && effectiveClose) {
            const riskPerShare = h.buy_price - h.initial_stop;
            if (riskPerShare > 0) {
                rValue = (effectiveClose - h.buy_price) / riskPerShare;
                rValues.push({ id: h.stock_id, r: rValue, hasLive: hasLive });
            }
        }

        let addSignal = data.addSignal && data.addSignal !== '-'
            ? `<span class="add-signal">${data.addSignal}</span>` : '-';
        // D11：附加加碼點訊號（pullback to MA20）
        if (data.addZone && data.addZone.type === 'pullback') {
            const az = data.addZone;
            const volTag = az.vol_ok ? '✓' : '⚠ 量縮';
            const zoneHtml = `<div style="margin-top:3px; font-size:11px; color:#83b;" title="${az.note}">
                💡 D11 加碼區：MA20=${az.ma20} RSI=${az.rsi} ${volTag}
            </div>`;
            addSignal = (addSignal === '-') ? zoneHtml.replace('margin-top:3px;', '') : (addSignal + zoneHtml);
        }

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

        // D13：出貨警示（高位量縮）疊加為次要訊息
        let distribHtml = '';
        if (data.distribAlert && data.distribAlert.signals && data.distribAlert.signals.length) {
            const da = data.distribAlert;
            const cls = da.level === 'danger' ? 'action-stoploss' : 'action-alert';
            const sigs = da.signals.join(' / ');
            distribHtml = `<div class="${cls}" style="margin-top:3px; font-size:10.5px; padding:2px 4px;" title="${da.note}">
                📉 D13 出貨：${sigs}
            </div>`;
            if (action === '持有') {
                action = '⚠ 高位量縮';
                actCls = da.level === 'danger' ? 'action-stoploss' : 'action-alert';
            }
        }

        // D14 + D23：換股建議（前 3 名候選）
        if (data.swapCandidates && data.swapCandidates.length > 0) {
            const cands = data.swapCandidates.slice(0, 3);
            const items = cands.map((c, i) => {
                const reasonStr = (c.reasons || []).join(' / ');
                const tag = i === 0 ? '🥇' : (i === 1 ? '🥈' : '🥉');
                return `<div style="font-size:10.5px;" title="${reasonStr}">
                    ${tag} #${c.cand_rank} ${c.cand_id} ${c.cand_name}
                    <span style="color:#3a7;">+${c.rs_gap}</span>
                </div>`;
            }).join('');
            distribHtml += `<div style="margin-top:3px; padding:3px 6px; background:#fff7e6; border-left:3px solid #ff9800; color:#c77;">
                🔁 D23 換股 Top3：${items}
            </div>`;
        } else if (data.swapSuggestion) {
            const sw = data.swapSuggestion;
            distribHtml += `<div style="margin-top:3px; font-size:10.5px; padding:2px 4px; background:#fff7e6; border-left:3px solid #ff9800; color:#c77;" title="${sw.note}">
                🔁 D14 換股：#${sw.cand_rank} ${sw.cand_id} ${sw.cand_name} (RS ${sw.cand_rs}, 強 ${sw.rs_gap})
            </div>`;
        }

        // D17：目標價上調建議（停損已 ratchet 接近目標，剩餘 RR < 0.5）
        if (h.target_price && currentStop && hasLive) {
            const remainUp = h.target_price - liveClose;
            const remainDown = liveClose - currentStop;
            if (remainUp > 0 && remainDown > 0 && remainUp / remainDown < 0.5) {
                // 上行空間 < 下行風險的一半 → 應上調目標
                const sug = data.suggestedTarget != null
                    ? Number(data.suggestedTarget).toFixed(2)
                    : (liveClose * 1.10).toFixed(2);
                distribHtml += `<div style="margin-top:3px; font-size:10.5px; padding:2px 4px; background:#1f3a2a; border-left:3px solid #3a7; color:#9cd9b8;" title="目標價剩餘 ${remainUp.toFixed(2)}（${(remainUp/liveClose*100).toFixed(1)}%），下行風險 ${remainDown.toFixed(2)}，已不對稱">
                    🎯 D17 上調目標：建議 ${sug}（剩餘 RR=${(remainUp/remainDown).toFixed(2)} 過低）
                </div>`;
            }
        }

        const fmt = (v, d=2) => (v == null ? '-' : Number(v).toFixed(d));
        // D15/D16：停損旁附小標籤
        let stopBadge = '';
        if (h.buy_price && h.max_high) {
            const gainNow = (h.max_high - h.buy_price) / h.buy_price;
            if (gainNow >= 0.30) stopBadge = '<span style="font-size:9px;color:#c33;margin-left:3px;" title="D15: 漲幅≥30%, ATR×0.6">🔒30</span>';
            else if (gainNow >= 0.20) stopBadge = '<span style="font-size:9px;color:#e66;margin-left:3px;" title="D15: 漲幅≥20%, ATR×0.75">🔒20</span>';
            else if (gainNow >= 0.15) stopBadge = '<span style="font-size:9px;color:#fa3;margin-left:3px;" title="D16: 盈利保護啟動">🔒15</span>';
            else if (gainNow >= 0.10) stopBadge = '<span style="font-size:9px;color:#fb6;margin-left:3px;" title="D15: 漲幅≥10%, ATR×0.9">🔒10</span>';
        }
        const stopDisp = currentStop
            ? `<span class="stop-value">${currentStop.toFixed(2)}</span>${stopBadge}` : '-';
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

        // D18：模型訊號（上漲機率 / 突破機率 / MC P+10%）
        const fmtPct = (v) => (v == null ? '<span class="na-cell" title="無模型預測">—</span>'
                                       : `${(v*100).toFixed(0)}%`);
        const upProbCell = data.upProb != null
            ? `<span title="Deep 模型 T+20 > +5% 機率，越高越樂觀" class="${data.upProb >= 0.6 ? 'gain' : (data.upProb < 0.4 ? 'loss' : '')}">${(data.upProb*100).toFixed(0)}%</span>`
            : '<span class="na-cell" title="無 Deep 模型預測">—</span>';
        const brkCell = data.breakoutProb != null
            ? `<span title="T+10 內漲幅≥+10%機率" class="${data.breakoutProb >= 0.5 ? 'gain' : ''}">${(data.breakoutProb*100).toFixed(0)}%</span>`
            : '<span class="na-cell" title="無突破分類器">—</span>';

        // 行內可編輯欄位包裝
        const editSpan = (val, field, formatted) => {
            const safe = (formatted ?? (val == null ? '-' : val));
            return `<span class="editable" title="雙擊編輯 ${field}" ondblclick="inlineEdit(${idx},'${field}',this)">${safe}</span>`;
        };

        return `<tr>
            <td>${h.stock_id}</td>
            <td>${editSpan(h.name, 'name', h.name || '-')}</td>
            <td>${h.buy_date || '-'}</td>
            <td>${editSpan(h.buy_price, 'buy_price', fmt(h.buy_price))}</td>
            <td>${editSpan(shares, 'shares', shares)}</td>
            <td>${closeCell}</td>
            <td class="${pnlCls}">${pnlCell}</td>
            <td>${upProbCell}</td>
            <td>${brkCell}</td>
            <td>${tierCell}</td>
            <td>${initStopDisp}</td>
            <td>${stopDisp}</td>
            <td class="${stpCls}">${distStpCell}</td>
            <td>${healthCell}</td>
            <td>${addSignal}</td>
            <td class="${actCls}" title="${sig.replace(/"/g,'&quot;')}">${action}${distribHtml}</td>
            <td>${editSpan(h.target_price, 'target_price', h.target_price != null ? Number(h.target_price).toFixed(2) : '-')}</td>
            <td>${_d10TargetHintCell(h, data)}</td>
            <td>${editSpan(h.stop_loss, 'stop_loss', h.stop_loss != null ? Number(h.stop_loss).toFixed(2) : '-')}</td>
            <td>${h.strategy || '-'}</td>
            <td>${editSpan(h.note, 'note', h.note || '-')}</td>
            <td class="ops-cell">
                <button class="btn-mini btn-edit" onclick="startEditHolding(${idx})" title="開啟完整編輯表單">✏ 改</button>
                <button class="btn-mini" onclick="deleteHolding(${idx})" title="可在 24 小時內復原">🗑 刪</button>
            </td>
        </tr>`;
    }).join('');

    const thead = `<tr>
        <th>代號</th><th>名稱</th><th>買進日</th><th title="雙擊可編輯">買進價</th><th title="雙擊可編輯">張數</th>
        <th>現價</th><th>損益%</th>
        <th title="Deep 模型 T+20 > +5% 機率">📈 上漲機率</th>
        <th title="突破分類器：T+10 內漲幅 ≥ +10% 機率">🚀 突破機率</th>
        <th>分層</th><th title="初始停損">初始停損</th>
        <th title="動態停損（初始 / 保本 / 分層 / Chandelier 取高）">當前停損</th>
        <th>距停損</th>
        <th title="綜合技術+法人+距停損的健檢分數">健檢</th>
        <th>加碼訊號</th>
        <th>動作</th>
        <th title="雙擊可編輯">目標價</th>
        <th title="D10：依 ATR 3R / 模型 95% 上界 / 30 日高 × 1.05 / 模型預測四者最大值，動態提示是否調高目標">建議調整</th>
        <th title="雙擊可編輯">停損價</th>
        <th>策略</th>
        <th title="雙擊可編輯">備註</th>
        <th>操作</th>
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

    // D18：R-value 統計
    let rPanel = '';
    if (rValues.length > 0) {
        const rs = rValues.map(x => x.r);
        const avgR = rs.reduce((a,b)=>a+b,0) / rs.length;
        const winners = rs.filter(r => r >= 1.0).length;
        const losers = rs.filter(r => r <= -0.5).length;
        const sorted = rValues.slice().sort((a,b) => b.r - a.r);
        const top3 = sorted.slice(0, 3).map(x =>
            `<span style="display:inline-block; margin:1px 3px; padding:1px 6px; background:#1f3a2a; color:#9cd9b8; border-radius:6px;">${x.id} ${x.r >= 0 ? '+' : ''}${x.r.toFixed(2)}R</span>`
        ).join('');
        const bot3 = sorted.slice(-3).reverse().map(x =>
            `<span style="display:inline-block; margin:1px 3px; padding:1px 6px; background:#3a1f1f; color:#ffab91; border-radius:6px;">${x.id} ${x.r >= 0 ? '+' : ''}${x.r.toFixed(2)}R</span>`
        ).join('');
        const avgCls = avgR >= 1.0 ? 'gain' : (avgR <= 0 ? 'loss' : '');
        rPanel = `
        <div style="margin-top:12px; background:#0a1118; border-radius:8px; padding:10px 14px; border-left:3px solid #5af; font-size:12px;">
            <div style="color:#9ab; margin-bottom:6px;">📐 <b>D18 R 值分布</b>（R = (現價 − 買價) / (買價 − 初始停損)）</div>
            <div style="display:flex; gap:18px; flex-wrap:wrap; margin-bottom:6px;">
                <span>樣本：<b>${rValues.length}</b></span>
                <span>平均：<b class="${avgCls}">${(avgR >= 0 ? '+' : '')}${avgR.toFixed(2)}R</b></span>
                <span>≥ +1R：<b class="gain">${winners}</b></span>
                <span>≤ −0.5R：<b class="loss">${losers}</b></span>
            </div>
            <div>領先：${top3 || '—'}</div>
            <div style="margin-top:3px;">墊底：${bot3 || '—'}</div>
        </div>`;
    }

    // D27：投組總覽儀表板 - 統計各類警示與行動建議
    let alertCounts = {distrib:0, swap:0, trail:0, addon:0, drawdown:0, targetUp:0, exit:0};
    let actionUrgent = 0;     // 急需處理（賣出/出場）
    let actionWatch = 0;      // 注意觀察
    let topGainer = null, topLoser = null;
    holdings.forEach(h => {
        const data = STOCK_DATA[h.stock_id];
        if (!data || data.close == null) return;
        if (data.distribAlert) alertCounts.distrib++;
        if (data.swapCandidates && data.swapCandidates.length > 0) alertCounts.swap++;
        const livePct = ((data.close - h.buy_price) / h.buy_price) * 100;
        if (h.entry_high && (h.entry_high - data.close) / h.entry_high >= 0.10) {
            alertCounts.drawdown++; actionWatch++;
        }
        // 找最大贏家/輸家
        if (topGainer == null || livePct > topGainer.pct) topGainer = {h, pct: livePct, data};
        if (topLoser == null || livePct < topLoser.pct) topLoser = {h, pct: livePct, data};
        // 急迫性：跌破停損 / D13 出貨 / 健檢 < 30
        if (data.distribAlert) actionUrgent++;
        const hSc = data.healthScore;
        if (hSc != null && hSc < 30) actionUrgent++;
    });

    // 生成行動建議
    let actionAdvice = '';
    if (actionUrgent === 0 && actionWatch === 0) {
        actionAdvice = `<span style="color:#3a9;">✅ 投組狀態良好，無急迫行動需求</span>`;
    } else {
        const parts = [];
        if (actionUrgent > 0) parts.push(`<span style="color:#f55;">🚨 ${actionUrgent} 檔需立即處理</span>`);
        if (actionWatch > 0) parts.push(`<span style="color:#fa3;">⚠ ${actionWatch} 檔需密切觀察</span>`);
        if (alertCounts.swap > 0) parts.push(`<span style="color:#ff9800;">🔁 ${alertCounts.swap} 檔可換股優化</span>`);
        actionAdvice = parts.join(' &nbsp;|&nbsp; ');
    }

    let dashboardPanel = '';
    if (holdings.length > 0) {
        const pnlClass = trackedPnL >= 0 ? 'gain' : 'loss';
        const pnlBgClass = trackedPnL >= 0 ? '#1f3a2a' : '#3a1f1f';
        const tgIcon = topGainer && topGainer.pct >= 0 ? '🥇' : '📉';
        const tlIcon = topLoser && topLoser.pct < 0 ? '🆘' : '📊';

        // 健檢分數視覺化
        const healthBar = avgHealth != null ?
            `<div style="background:#22272e; height:8px; border-radius:4px; overflow:hidden; margin-top:4px;">
                <div style="background:${avgHealth>=70?'#3a9':(avgHealth>=40?'#fa3':'#c55')}; width:${avgHealth}%; height:100%;"></div>
            </div>` : '';
        const riskBar = totalCost > 0 ?
            `<div style="background:#22272e; height:8px; border-radius:4px; overflow:hidden; margin-top:4px;">
                <div style="background:${parseFloat(riskPct)<5?'#3a9':(parseFloat(riskPct)<10?'#fa3':'#c55')}; width:${Math.min(100, parseFloat(riskPct)*5)}%; height:100%;"></div>
            </div>` : '';

        dashboardPanel = `
        <div style="margin-top:14px; background:linear-gradient(135deg, #0a1118 0%, #0d1422 100%); border-radius:10px; padding:14px 18px; border:1px solid #1f3a52;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-size:14px; font-weight:600; color:#9ad;">📊 D27 投組總覽儀表板</div>
                <div style="font-size:12px;">${actionAdvice}</div>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px;">
                <div style="background:#0d1118; padding:10px; border-radius:6px; border-left:3px solid ${trackedPnL>=0?'#3a9':'#c55'};">
                    <div style="font-size:11px; color:#888; margin-bottom:4px;">浮動損益</div>
                    <div style="font-size:18px; font-weight:700;" class="${pnlClass}">${(trackedPnL>=0?'+':'')}${Math.round(trackedPnL).toLocaleString()}</div>
                    <div style="font-size:13px;" class="${pnlClass}">(${trackedPnLPct}%)</div>
                </div>
                <div style="background:#0d1118; padding:10px; border-radius:6px; border-left:3px solid #5af;">
                    <div style="font-size:11px; color:#888; margin-bottom:4px;">勝率</div>
                    <div style="font-size:18px; font-weight:700;">${winRate}%</div>
                    <div style="font-size:11px; color:#a0a0b0;">${tracked.filter(h => STOCK_DATA[h.stock_id].close > h.buy_price).length} / ${tracked.length} 檔獲利</div>
                </div>
                <div style="background:#0d1118; padding:10px; border-radius:6px; border-left:3px solid #c55;">
                    <div style="font-size:11px; color:#888; margin-bottom:4px;">投組風險</div>
                    <div style="font-size:18px; font-weight:700;">${riskPct}%</div>
                    ${riskBar}
                </div>
                <div style="background:#0d1118; padding:10px; border-radius:6px; border-left:3px solid ${avgHealth==null?'#666':(avgHealth>=70?'#3a9':(avgHealth>=40?'#fa3':'#c55'))};">
                    <div style="font-size:11px; color:#888; margin-bottom:4px;">平均健檢</div>
                    <div style="font-size:18px; font-weight:700;">${avgHealth == null ? '—' : avgHealth + '/100'}</div>
                    ${healthBar}
                </div>
                ${topGainer ? `<div style="background:${topGainer.pct>=0?'#0f1d14':'#1d0f0f'}; padding:10px; border-radius:6px; border-left:3px solid ${topGainer.pct>=0?'#3a9':'#c55'};">
                    <div style="font-size:11px; color:#888; margin-bottom:4px;">${tgIcon} 最強</div>
                    <div style="font-size:13px;">${topGainer.h.stock_id} ${topGainer.h.name||''}</div>
                    <div style="font-size:16px; font-weight:600;" class="${topGainer.pct>=0?'gain':'loss'}">${topGainer.pct>=0?'+':''}${topGainer.pct.toFixed(1)}%</div>
                </div>` : ''}
                ${topLoser && topLoser.h.stock_id !== (topGainer && topGainer.h.stock_id) ? `<div style="background:${topLoser.pct>=0?'#0f1d14':'#1d0f0f'}; padding:10px; border-radius:6px; border-left:3px solid ${topLoser.pct>=0?'#3a9':'#c55'};">
                    <div style="font-size:11px; color:#888; margin-bottom:4px;">${tlIcon} 最弱</div>
                    <div style="font-size:13px;">${topLoser.h.stock_id} ${topLoser.h.name||''}</div>
                    <div style="font-size:16px; font-weight:600;" class="${topLoser.pct>=0?'gain':'loss'}">${topLoser.pct>=0?'+':''}${topLoser.pct.toFixed(1)}%</div>
                </div>` : ''}
            </div>
            <div style="margin-top:10px; display:flex; gap:14px; flex-wrap:wrap; font-size:11.5px; color:#a0a0b0;">
                ${alertCounts.distrib > 0 ? `<span>📉 出貨警示 <b style="color:#c55;">${alertCounts.distrib}</b></span>` : ''}
                ${alertCounts.drawdown > 0 ? `<span>📊 高位回落 <b style="color:#fa3;">${alertCounts.drawdown}</b></span>` : ''}
                ${alertCounts.swap > 0 ? `<span>🔁 換股建議 <b style="color:#ff9800;">${alertCounts.swap}</b></span>` : ''}
                ${(alertCounts.distrib + alertCounts.drawdown + alertCounts.swap === 0) ? `<span style="color:#3a9;">✅ 無重要警示</span>` : ''}
            </div>
        </div>`;
    }

    summary.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item"><div class="label">持股檔數</div><div class="value">${holdings.length}</div></div>
            <div class="summary-item"><div class="label">總成本</div><div class="value">${Math.round(totalCost).toLocaleString()}</div></div>
            <div class="summary-item profit-item"><div class="label">可追蹤市值</div><div class="value">${Math.round(trackedMkt).toLocaleString()}</div></div>
            <div class="summary-item ${trackedPnL>=0?'profit-item':'risk-item'}"><div class="label">浮動損益</div><div class="value ${trackedPnL>=0?'gain':'loss'}">${(trackedPnL>=0?'+':'')}${Math.round(trackedPnL).toLocaleString()} (${trackedPnLPct}%)</div></div>
            <div class="summary-item"><div class="label">獲利檔比</div><div class="value">${tracked.filter(h => STOCK_DATA[h.stock_id].close > h.buy_price).length}/${tracked.length} (${winRate}%)</div></div>
            <div class="summary-item risk-item"><div class="label" title="全部持股若碰初始停損的總損失">投組總風險</div><div class="value">${Math.round(totalRisk).toLocaleString()} (${riskPct}%)</div></div>
            <div class="summary-item ${avgHealth==null?'':(avgHealth>=70?'profit-item':(avgHealth>=40?'warn-item':'risk-item'))}"><div class="label">平均健檢分數</div><div class="value">${avgHealth == null ? '-' : avgHealth + '/100'}</div></div>
        </div>
        ${dashboardPanel}
        ${rPanel}`;

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
    refreshTrashCount();
    // ESC 關閉編輯表單與垃圾桶與 D22 平倉模擬與 D24 壓力測試
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const tm = document.getElementById('trash-modal');
            if (tm && tm.style.display === 'flex') hideTrash();
            const em = document.getElementById('exit-sim-modal');
            if (em && em.style.display === 'flex') hideExitSim();
            const sm = document.getElementById('stress-modal');
            if (sm && sm.style.display === 'flex') hideStressTest();
        }
    });
})();
</script>
"""
    script = (script
              .replace('__STOCK_JSON__', stock_json)
              .replace('__INITIAL_JSON__', initial_json)
              .replace('__PORTFOLIO_JSON__', portfolio_json)
              .replace('__HAS_INITIAL__', has_initial))

    # D21：持股相關性 heatmap（伺服端產生，Plotly.js 渲染）
    corr_html = _build_holdings_correlation_html(initial_holdings_df)
    html = html.replace('__HOLDINGS_CORR_HTML__', corr_html)

    return style + html + script


def _strategy_oos_pill(sname, summary):
    """C11：產生策略名稱旁的 OOS 績效小標籤。
    顯示：mean_ret / hit_rate / n_signals。
    """
    if not summary or sname not in summary:
        return ''
    st = summary[sname]
    mean = st['mean_ret'] * 100
    hit = st['hit_rate'] * 100
    n = st['n_signals']
    if mean >= 2:
        color, icon = '#26a69a', '🏆'
    elif mean >= 0:
        color, icon = '#83b', '✓'
    elif mean >= -2:
        color, icon = '#c94', '⚠'
    else:
        color, icon = '#ef5350', '✗'
    return (f'<span style="font-size:12.5px; font-weight:500; padding:2px 8px; '
            f'background:rgba(255,255,255,0.05); border:1px solid {color}; '
            f'border-radius:12px; color:{color}; margin-left:8px; vertical-align:middle;" '
            f'title="OOS 過去 10 日 forward 平均報酬 / 勝率 / 命中數">'
            f'{icon} OOS {mean:+.1f}% / 勝率 {hit:.0f}% / n={n}</span>')


def _render_strategy_overlap_heatmap(buy_dfs_map):
    """C17：當日各策略命中股票的 Jaccard overlap 熱力圖。
    高重疊（>0.5）→ 兩策略本質相同，命中加總會放大同一檔權重。
    低重疊（<0.1）→ 兩策略互補，融合可分散風險。"""
    if not buy_dfs_map:
        return ''
    sets = {}
    for sname, df in buy_dfs_map.items():
        if df is None or df.empty or '證券代號' not in df.columns:
            continue
        ids = set(df['證券代號'].astype(str).str.strip().tolist())
        if ids:
            sets[sname] = ids
    if len(sets) < 2:
        return ''
    names = list(sets.keys())
    n = len(names)
    matrix = []
    for i, ai in enumerate(names):
        row = []
        for j, bj in enumerate(names):
            a = sets[ai]; b = sets[bj]
            if i == j:
                row.append(1.0)
                continue
            uni = len(a | b)
            row.append(round(len(a & b) / uni, 3) if uni > 0 else 0.0)
        matrix.append(row)

    # 找出最高重疊的 pair（不含對角）
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] >= 0.5:
                high_pairs.append((names[i], names[j], matrix[i][j]))
    high_pairs.sort(key=lambda x: x[2], reverse=True)

    div_id = 'strategyOverlapHeatmap'
    fig_data = json.dumps([{
        'type': 'heatmap',
        'z': matrix, 'x': names, 'y': names,
        'colorscale': [
            [0.0, '#0a0e1a'], [0.1, '#1a2a4a'],
            [0.3, '#2a5a8a'], [0.5, '#c94'],
            [0.7, '#c66'], [1.0, '#c33'],
        ],
        'zmin': 0, 'zmax': 1,
        'showscale': True,
        'text': [[f"{v:.2f}" for v in row] for row in matrix],
        'texttemplate': '%{text}',
        'textfont': {'size': 10, 'color': '#e0e0e0'},
        'hovertemplate': '%{y} ↔ %{x}<br>Jaccard = %{z:.3f}<extra></extra>',
    }], ensure_ascii=False)
    layout = json.dumps({
        'title': {'text': 'C17 策略命中重疊度 Heatmap（Jaccard）',
                  'font': {'color': '#e0e0e0', 'size': 14}},
        'plot_bgcolor': '#0a0e1a', 'paper_bgcolor': '#0a0e1a',
        'font': {'color': '#cde'},
        'xaxis': {'tickangle': -30, 'gridcolor': '#1a223a'},
        'yaxis': {'autorange': 'reversed', 'gridcolor': '#1a223a'},
        'margin': {'t': 50, 'b': 90, 'l': 90, 'r': 50},
        'height': max(360, 28 * n + 120),
    }, ensure_ascii=False)

    pair_html = ''
    if high_pairs:
        top_pairs = high_pairs[:5]
        items = ' / '.join([f'<b>{a}↔{b}</b> {v:.2f}' for a, b, v in top_pairs])
        pair_html = (f'<p style="color:#fc8; font-size:11.5px; margin:6px 0 0;">'
                     f'⚠ 高重疊（≥0.5）配對：{items} — 同類策略，建議檢視是否雙重加分。</p>')

    return f'''
    <div style="margin-top:14px; background:var(--bg-card); padding:14px 18px;
                border-radius:8px; border-left:4px solid #f80;">
        <div style="font-size:14px; color:var(--text-primary); font-weight:600;">
            🔗 C17 策略命中重疊度
        </div>
        <p style="color:var(--text-secondary); font-size:11.5px; margin:6px 0;">
            Jaccard = |A∩B| / |A∪B|。對角線恆 1。色越紅、重疊越高（同類策略）。
        </p>
        <div id="{div_id}"></div>
        {pair_html}
        <script>
            (function(){{
                if (typeof Plotly === 'undefined') return;
                Plotly.newPlot('{div_id}', {fig_data}, {layout},
                               {{displayModeBar:false, responsive:true}});
            }})();
        </script>
    </div>'''


def _render_strategy_oos_dashboard(strategy_oos_summary):
    """C15：策略 OOS Dashboard — Plotly 雙軸 bar：mean_ret + hit_rate。
    依 mean_ret 由高到低排序，視覺化哪些策略長期賺錢、哪些失靈。"""
    if not strategy_oos_summary:
        return ''
    items = []
    for sname, st in strategy_oos_summary.items():
        n = int(st.get('n_signals') or 0)
        if n < 5:
            continue
        items.append({
            'name': sname,
            'mean_ret': float(st.get('mean_ret') or 0.0) * 100,
            'hit_rate': float(st.get('hit_rate') or 0.0) * 100,
            'n_signals': n,
        })
    if not items:
        return ''
    # 加入 C20 Sharpe / max_dd 與 C21 regime breakdown
    for x, sname in [(x, x['name']) for x in items]:
        st_full = strategy_oos_summary.get(sname) or {}
        x['sharpe'] = float(st_full.get('sharpe') or 0.0)
        x['max_dd'] = float(st_full.get('max_dd') or 0.0) * 100
        rb = st_full.get('regime_breakdown') or {}
        x['bull_mean'] = float(rb.get('bull', {}).get('mean_ret') or 0.0) * 100 if 'bull' in rb else None
        x['bear_mean'] = float(rb.get('bear', {}).get('mean_ret') or 0.0) * 100 if 'bear' in rb else None
    items.sort(key=lambda x: x['mean_ret'], reverse=True)
    labels = [x['name'] for x in items]
    means = [round(x['mean_ret'], 2) for x in items]
    hits = [round(x['hit_rate'], 2) for x in items]
    counts = [x['n_signals'] for x in items]
    sharpes = [round(x['sharpe'], 2) for x in items]
    max_dds = [round(x['max_dd'], 1) for x in items]

    # 視覺化顏色：mean ≥ +1.5 綠、≤ -1 紅、其他黃
    colors = []
    for m in means:
        if m >= 1.5:
            colors.append('#3a7')
        elif m <= -1.0:
            colors.append('#c44')
        else:
            colors.append('#c94')

    div_id = 'strategyOosDashboard'
    fig_data = json.dumps([
        {
            'type': 'bar', 'name': '平均 forward 10D 報酬 (%)',
            'x': labels, 'y': means, 'marker': {'color': colors},
            'text': [f"{m:+.2f}% (n={c})\nSh={s:.2f} DD={d:.1f}%"
                     for m, c, s, d in zip(means, counts, sharpes, max_dds)],
            'textposition': 'outside', 'yaxis': 'y',
            'hovertemplate': ('<b>%{x}</b><br>'
                              '平均報酬 %{y:.2f}%<br>'
                              'Sharpe（年化）%{customdata[0]:.2f}<br>'
                              '最大回撤 %{customdata[1]:.1f}%<br>'
                              '樣本數 %{customdata[2]}<br>'
                              '牛市 %{customdata[3]} / 熊市 %{customdata[4]}'
                              '<extra></extra>'),
            'customdata': [
                [s, d, c,
                 (f"{x['bull_mean']:+.2f}%" if x['bull_mean'] is not None else 'N/A'),
                 (f"{x['bear_mean']:+.2f}%" if x['bear_mean'] is not None else 'N/A')]
                for s, d, c, x in zip(sharpes, max_dds, counts, items)
            ],
        },
        {
            'type': 'scatter', 'name': '勝率 (%)',
            'x': labels, 'y': hits, 'mode': 'lines+markers',
            'line': {'color': '#42a5f5', 'width': 2}, 'yaxis': 'y2',
        },
    ], ensure_ascii=False)
    layout = json.dumps({
        'title': {'text': 'C15 策略 OOS Dashboard（最近 6 個月、forward 10 日）',
                  'font': {'color': '#e0e0e0', 'size': 14}},
        'plot_bgcolor': '#0a0e1a', 'paper_bgcolor': '#0a0e1a',
        'font': {'color': '#cde'},
        'xaxis': {'tickangle': -30, 'gridcolor': '#1a223a'},
        'yaxis': {'title': '平均報酬 %', 'gridcolor': '#1a223a', 'side': 'left'},
        'yaxis2': {'title': '勝率 %', 'overlaying': 'y',
                   'side': 'right', 'showgrid': False, 'range': [30, 80]},
        'legend': {'orientation': 'h', 'y': -0.25},
        'margin': {'t': 50, 'b': 100, 'l': 50, 'r': 50},
        'height': 380,
    }, ensure_ascii=False)

    return f'''
    <div style="margin-top:14px; background:var(--bg-card); padding:14px 18px;
                border-radius:8px; border-left:4px solid #5af;">
        <div style="font-size:14px; color:var(--text-primary); font-weight:600;">
            📈 C15 策略 OOS Dashboard
        </div>
        <p style="color:var(--text-secondary); font-size:11.5px; margin:6px 0;">
            綠色 = 長期賺錢（&ge; +1.5%）｜黃色 = 中性｜紅色 = 長期虧損（&le; -1%）。
            樣本 &lt; 5 不顯示。配合 C13 自動倍率 → 強策略加權、弱策略降權。
        </p>
        <div id="{div_id}" style="height:380px;"></div>
        <script>
            (function(){{
                if (typeof Plotly === 'undefined') return;
                Plotly.newPlot('{div_id}', {fig_data}, {layout},
                               {{displayModeBar:false, responsive:true}});
            }})();
        </script>
    </div>'''


def _render_oos_card(oos_summary):
    """C6 lite：產生 OOS 監控小卡 HTML。
    若樣本不足回傳提示卡片；有資料則顯示 mean_ret / hit_rate / 樣本日數，
    並嵌入 Plotly 折線圖（顯示歷史每日 mean_ret 與正報酬比例）。
    同時顯示 C8 OOS IC（各分數層級的解釋力）。
    """
    # 讀 IC（若有）
    ic_html = ''
    try:
        if os.path.exists(OOS_IC_FILE):
            with open(OOS_IC_FILE, 'r', encoding='utf-8') as f:
                ic_payload = json.load(f)
            ic = ic_payload.get('ic', {})
            if ic:
                pills = []
                for k, v in ic.items():
                    color = '#3a7' if v >= 0.05 else ('#c44' if v <= -0.05 else '#c94')
                    pills.append(
                        f'<span style="display:inline-block; margin:2px 4px; padding:2px 8px; '
                        f'background:#0d1118; border:1px solid {color}; border-radius:3px; '
                        f'font-size:11.5px; color:#cde;">'
                        f'{k} <strong style="color:{color};">{v:+.3f}</strong></span>'
                    )
                ic_html = (
                    f'<div style="margin-top:8px;">'
                    f'<div style="font-size:12px; color:#9ab; margin-bottom:3px;">'
                    f'C8 OOS IC（n={ic_payload.get("n_samples", "?")}）— '
                    f'數值越高代表該分數對未來報酬解釋力越強，&lt; 0 表示該層失靈</div>'
                    f'<div>{"".join(pills)}</div></div>'
                )
    except Exception:
        pass

    if not oos_summary:
        return f'''
        <div style="margin-top:16px; background:var(--bg-card); padding:14px 18px;
                    border-radius:8px; border-left:4px solid #777;">
            <div style="font-size:14px; color:var(--text-primary); font-weight:600;">
                📊 C6 OOS 滾動監控
            </div>
            <p style="color:var(--text-secondary); font-size:12.5px; margin:6px 0 0;">
                尚未累積足夠歷史快照（至少 10 個交易日）。
                每日執行會自動將 ranking 快照進 ranking_snapshots.csv，待累積後即可量化策略真實命中率。
            </p>
            {ic_html}
        </div>'''

    mean_ret = oos_summary['mean_ret'] * 100
    hit = oos_summary['hit_rate'] * 100
    n = oos_summary['n_dates']
    win = oos_summary['window_days']
    best = oos_summary.get('best_date_ret', 0) * 100
    worst = oos_summary.get('worst_date_ret', 0) * 100
    color = '#3a7' if mean_ret > 1 else ('#c44' if mean_ret < -1 else '#c94')

    # OOS 折線圖：讀 oos_monitor.csv 取最近 60 個樣本
    chart_html = ''
    try:
        if os.path.exists(OOS_MONITOR_FILE):
            df_oos = pd.read_csv(OOS_MONITOR_FILE).tail(60)
            if len(df_oos) >= 2:
                dates = df_oos['日期'].tolist()
                rets = (df_oos['平均報酬'] * 100).round(2).tolist()
                hits = (df_oos['正報酬比例'] * 100).round(1).tolist()
                ma5 = df_oos['平均報酬'].rolling(5).mean().bfill().mul(100).round(2).tolist()
                chart_id = 'oosChart'
                chart_html = f'''
                <div id="{chart_id}" style="height:260px; margin-top:10px;"></div>
                <script>
                Plotly.newPlot('{chart_id}', [
                    {{
                        x: {json.dumps(dates)}, y: {json.dumps(rets)},
                        type: 'bar', name: '日 Top-10 平均%',
                        marker: {{ color: {json.dumps(rets)}.map(v => v >= 0 ? '#26a69a' : '#ef5350') }}
                    }},
                    {{
                        x: {json.dumps(dates)}, y: {json.dumps(ma5)},
                        type: 'scatter', mode: 'lines', name: '5日均線', line: {{ color: '#42a5f5', width: 2 }}
                    }},
                    {{
                        x: {json.dumps(dates)}, y: {json.dumps(hits)},
                        type: 'scatter', mode: 'lines', name: '勝率%', yaxis: 'y2',
                        line: {{ color: '#c94', width: 1.5, dash: 'dot' }}
                    }}
                ], {{
                    paper_bgcolor: '#0f0f1a', plot_bgcolor: '#0f0f1a',
                    font: {{ color: '#cde', size: 11 }},
                    margin: {{ t: 18, l: 50, r: 50, b: 36 }},
                    xaxis: {{ tickangle: -30, gridcolor: '#222' }},
                    yaxis: {{ title: '平均報酬 %', gridcolor: '#222' }},
                    yaxis2: {{ title: '勝率 %', overlaying: 'y', side: 'right', range: [0, 100], gridcolor: 'transparent' }},
                    legend: {{ orientation: 'h', y: 1.18 }},
                    showlegend: true
                }}, {{ responsive: true }});
                </script>'''
    except Exception as e:
        chart_html = f'<p style="color:#c94; font-size:11.5px; margin-top:6px;">折線圖渲染失敗：{e}</p>'

    return f'''
    <div style="margin-top:16px; background:var(--bg-card); padding:14px 18px;
                border-radius:8px; border-left:4px solid {color};">
        <div style="font-size:14px; color:var(--text-primary); font-weight:600;">
            📊 C6 OOS 滾動監控（最近 {n} 個快照日，Top-10 持有 {win} 個交易日後 forward return）
        </div>
        <div style="display:flex; gap:24px; flex-wrap:wrap; margin-top:8px; font-size:13px;">
            <div><strong>平均報酬：</strong><span style="color:{color}; font-weight:600;">{mean_ret:+.2f}%</span></div>
            <div><strong>勝率：</strong>{hit:.0f}%</div>
            <div><strong>最佳日：</strong><span style="color:#3a7;">{best:+.2f}%</span></div>
            <div><strong>最差日：</strong><span style="color:#c44;">{worst:+.2f}%</span></div>
        </div>
        {chart_html}
        {ic_html}
        <p style="color:var(--text-secondary); font-size:11.5px; margin:8px 0 0;">
            數據來源：ranking_snapshots.csv，逐日累積 → 量化模型實際 OOS 表現，融合權重會自動依 IC 調整（C8）。
        </p>
    </div>'''


def save_html_report(sell_alerts, buy_dfs_map,
                     ranking=None, undervalued=None, early_potential=None,
                     backtest_result=None, all_data=None, revenue_data=None,
                     market_state=None, holdings_df=None,
                     portfolio_risk=None, ig_map=None, mc_probs=None,
                     deep_preds=None, breakout_preds=None,
                     oos_summary=None, strategy_oos_summary=None,
                     portfolio_bt=None):
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
        ("early", "潛力股提前佈局", early_potential, None),
        ("ranking", "綜合買入潛力 TOP 30", ranking, None),
        ("undervalued", "低估股篩選", undervalued, None),
        ("sell", "賣出警報（強化版）", sell_alerts, None),
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
        '策略14': '策略14：RS Line 領漲（先創新高）',
        '策略15': '策略15：量縮回測 MA20/MA60',
        '策略16': '策略16：季線首次站上 + 量增 1.8x',
        '策略17': '策略17：主力建倉代理（法人連買 + 量能 + 占比）',
        '策略18': '策略18：中期盤整突破（30日盤整 + 突破 + ATR 擴張）',
        '策略19': '策略19：Pocket Pivot（量能變身、機構建倉）',
        '策略20': '策略20：營收驚喜後進（YoY≥30% 漂移）',
        '策略21': '策略21：三聯共振（法人連買 + 營收加速 + RS 領先）',
        '策略23': '策略23：Inside Day Breakout（內包日收斂後突破）',
        '策略24': '策略24：Breakaway Gap（跳空缺口 + 量爆 + 新高）',
        '策略25': '策略25：籌碼結構（10 日法人連續性 + 量能溫和擴張）',
        '策略26': '策略26：三重底反轉（多次測低 + 頸線突破）',
        '策略27': '策略27：融資斷頭反彈（散戶洗清後紅 K 反彈）',
        '策略28': '策略28：Cup with Handle（CANSLIM 杯柄突破）',
        '策略29': '策略29：Pivot Breakout（樞紐高點壓力線突破）',
        '策略31': '策略31：法人三連買 + 量能整理突破（短打型）',
        '策略33': '策略33：V-shape 反轉（連跌後紅 K + 量爆）',
        '策略34': '策略34：軋空動能（券資比高 + 法人連買 + 量價突破）',
        '策略36': '策略36：吸籌型（連跌量縮 + 法人偷買 + 中期未失守）',
        '策略37': '策略37：rotation 領頭（大盤盤整 + 個股強相對強度）',
        '策略38': '策略38：跌破 MA20 後反吃（假跌破洗盤完成 → 短打反轉）',
        '策略39': '策略39：外資 + 投信「皆」連 2 日買超（強共識）',
        '策略40': '策略40：MA20 / MA60 黃金交叉 + 量能共振（經典中長線起漲）',
    }
    for idx, (sname, df) in enumerate(buy_dfs_map.items(), 1):
        sec_id = f"s{idx}"
        title = strategy_titles.get(sname, sname)
        sections.append((sec_id, title, df, sname))   # C11：保留原 sname 給 OOS 查詢

    holdings_html = _build_holdings_section(
        sell_alerts, holdings_df, all_data,
        portfolio_risk=portfolio_risk,
        deep_preds=deep_preds, breakout_preds=breakout_preds, mc_probs=mc_probs,
        market_state=market_state,
        ranking_df=ranking,
    )

    table_sections = ''
    nav_items = '<a href="#market">大盤</a><a href="#holdings">我的持股</a><a href="#legend">指標說明</a><a href="#backtest">回測績效</a>'
    if ig_map:
        nav_items += '<a href="#ig">模型解釋</a>'
    for sec_id, title, df, sname in sections:
        nav_items += f'<a href="#{sec_id}">{title[:6]}</a>'
        oos_pill = _strategy_oos_pill(sname, strategy_oos_summary)
        table_sections += f'''
        <section id="{sec_id}">
            <h2>{title} {oos_pill}</h2>
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
    # ═══ 欄位說明 / 新指標圖例 ═══
    legend_html = '''
    <section id="legend">
        <h2>指標說明（v2.0 新增）</h2>
        <div style="background: var(--bg-card); padding: 18px 22px; border-radius: 8px;
                    border-left: 4px solid #9c27b0; font-size: 13px; line-height: 1.8;">
            <strong style="color:#ba68c8;">★ 三層融合（rule + ml + deep）＋ Stacking 元模型</strong>
            <ul style="margin:8px 0 12px 20px; color:var(--text-secondary);">
                <li><strong>規則分數</strong>：13 策略命中加權</li>
                <li><strong>ML 分數</strong>：XGBoost LambdaRank 依技術特徵排名</li>
                <li><strong>Deep 分數</strong>：GRU/Transformer 預測 T+20 報酬 × 上漲機率</li>
                <li><strong>元模型分數</strong>：Stacking 依市場狀態（多/空/震盪）動態加權融合</li>
            </ul>
            <strong style="color:#ba68c8;">★ Tier A：融合預測</strong>
            <ul style="margin:8px 0 12px 20px; color:var(--text-secondary);">
                <li><strong>上漲機率(20d)</strong>：Deep 模型 T+20 大於 +5% 的機率</li>
                <li><strong>模型預測漲幅</strong>：Deep 模型 T+20 預期報酬</li>
                <li><strong>95%區間</strong>：Conformal Prediction 預測區間（真實值落入此區間的機率 95%；用獨立 calibration set 校準）</li>
                <li><strong>預測目標價</strong>：60% 模型 + 40% 技術面（若上漲機率 &gt; 0.6 則 70%:30%）</li>
            </ul>
            <strong style="color:#ba68c8;">★ Tier B / D：獨立預測模型</strong>
            <ul style="margin:8px 0 12px 20px; color:var(--text-secondary);">
                <li><strong>突破機率(10d)</strong>：Focal Loss GRU 預測 T+10 內漲幅 &ge; +10% 的機率</li>
                <li><strong>MC P(+10%)</strong>：蒙地卡羅 10,000 條路徑模擬，20 日內觸及 +10% 的機率</li>
                <li><strong>MC P(+15%)</strong>：同上，門檻為 +15%</li>
            </ul>
            <strong style="color:#ba68c8;">★ Tier C：情境加成</strong>
            <ul style="margin:8px 0 0 20px; color:var(--text-secondary);">
                <li><strong>法人連買天</strong>：三大法人連續買超天數（每天 +1 分）</li>
                <li><strong>營收YoY</strong>：去年同月增減（&gt; 50% 加 6 分，&gt; 20% 加 3 分）</li>
                <li><strong>RS加速</strong>：相對大盤強度的近期加速度（&gt; 0 代表領漲加速）</li>
                <li><strong>相對強度百分位</strong>：上漲機率在全市場的百分位（&gt; 90% 加 3 分）</li>
            </ul>
        </div>
    </section>'''

    # ═══ Tier D3 IG 解釋區塊 ═══
    ig_html = ''
    if ig_map:
        rows_html = ''
        for sid, pairs in list(ig_map.items())[:5]:
            pair_rows = ''.join([
                f'<tr><td>{name}</td><td class="{"pos" if v > 0 else "neg"}">{v:+.4f}</td></tr>'
                for name, v in pairs
            ])
            rows_html += f'''
            <div style="background:var(--bg-card); padding:14px 18px; border-radius:8px;
                        margin-bottom:12px;">
                <h3 style="color:var(--accent);">{sid} — Top 特徵貢獻</h3>
                <table class="ig-table"><thead>
                    <tr><th>特徵</th><th>IG 貢獻</th></tr>
                </thead><tbody>{pair_rows}</tbody></table>
            </div>'''
        ig_html = f'''
        <section id="ig">
            <h2>模型推理解釋（Integrated Gradients / TOP 5）</h2>
            <p style="color:var(--text-secondary); font-size:13px; margin-bottom:14px;">
                正值（綠）代表該特徵推升模型看漲機率；負值（紅）代表抑制。<br/>
                可用來檢視模型判斷背後的主要動能來源（RSI 動能 / 量能擴張 / MA 位置等）。
            </p>
            {rows_html}
        </section>'''

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
        {_render_oos_card(oos_summary)}
        {_render_strategy_oos_dashboard(strategy_oos_summary)}
        {_render_portfolio_bt_card(portfolio_bt)}
        {_render_strategy_overlap_heatmap(buy_dfs_map)}
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
.na-cell {{ color: #607d8b; opacity: 0.55; cursor: help; font-style: italic; }}
.ig-table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
    margin-top: 8px;
}}
.ig-table th, .ig-table td {{
    padding: 6px 10px; border-bottom: 1px solid var(--border);
    text-align: left;
}}
.ig-table .pos {{ color: #26a69a; font-weight: 600; }}
.ig-table .neg {{ color: #ef5350; font-weight: 600; }}
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

{legend_html}

{ig_html}

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
        _raw_ohlcv_cache.clear()  # B6：同步清空共用 OHLCV 快取

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

        # 投組風險總覽（D6 動態風險預算 + D7 族群集中度）
        portfolio_risk = check_portfolio_risk(holdings, market_state=market_state)
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

        print("\n執行中期突破策略 14-16（RS Line / 量縮回測 / 季線首站）...")
        buy14 = strategy14(all_data)
        buy15 = strategy15(all_data)
        buy16 = strategy16(all_data)

        print("\n執行主力 / 中期盤整突破策略 17-18...")
        buy17 = strategy17(all_data)
        buy18 = strategy18(all_data)

        print("\n執行 Pocket Pivot / 營收驚喜後進策略 19-20...")
        buy19 = strategy19(all_data)
        buy20 = strategy20(all_data, revenue_data)

        print("\n執行三聯共振策略 21（法人 + 營收 + RS）...")
        buy21 = strategy21(all_data, revenue_data)

        print("\n執行短期突破策略 23-24（內包日 / 跳空缺口）...")
        buy23 = strategy23(all_data)
        buy24 = strategy24(all_data)

        print("\n執行籌碼結構 / 反轉策略 25-27...")
        buy25 = strategy25(all_data)
        buy26 = strategy26(all_data)
        buy27 = strategy27(all_data, margin_data)

        print("\n執行 Cup with Handle 經典型態策略 28...")
        buy28 = strategy28(all_data)

        print("\n執行 Pivot Breakout 策略 29...")
        buy29 = strategy29(all_data)

        print("\n執行短打型法人連買策略 31...")
        buy31 = strategy31(all_data)

        print("\n執行 V-shape 反轉策略 33...")
        buy33 = strategy33(all_data)

        print("\n執行軋空動能策略 34...")
        buy34 = strategy34(all_data, margin_data)

        print("\n執行吸籌型策略 36 + rotation 領頭策略 37...")
        buy36 = strategy36(all_data)
        buy37 = strategy37(all_data)

        print("\n執行短打反吃策略 38 + 強共識策略 39 + MA20/60 黃金交叉策略 40...")
        buy38 = strategy38(all_data)
        buy39 = strategy39(all_data)
        buy40 = strategy40(all_data)

        buy_dfs_map = {
            '策略1': buy1, '策略2': buy2, '策略3': buy3, '策略4': buy4,
            '策略5': buy5, '策略6': buy6, '策略7': buy7, '策略8': buy8,
            '策略9': buy9, '策略10': buy10, '策略11': buy11,
            '策略12': buy12, '策略13': buy13,
            '策略14': buy14, '策略15': buy15, '策略16': buy16,
            '策略17': buy17, '策略18': buy18,
            '策略19': buy19, '策略20': buy20,
            '策略21': buy21,
            '策略23': buy23, '策略24': buy24,
            '策略25': buy25, '策略26': buy26, '策略27': buy27,
            '策略28': buy28, '策略29': buy29,
            '策略31': buy31, '策略33': buy33,
            '策略34': buy34,
            '策略36': buy36, '策略37': buy37,
            '策略38': buy38, '策略39': buy39,
            '策略40': buy40,
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

        # ═══ D5：剔除最近 30 日剛停損的股票（避免被反覆洗）═══
        blacklist = get_recent_stoploss_blacklist()
        if blacklist:
            print(f"\n📛 D5 黑名單：最近 {BLACKLIST_DAYS} 日內停損 {len(blacklist)} 檔")
            ranking = _apply_blacklist(ranking, blacklist, name='主排名')

        # ═══ Tier C: 情境特徵加成（法人連買天數 / 營收YoY / RS加速度） ═══
        try:
            ranking = _apply_context_boost(ranking, all_data, revenue_data)
            print("  ✓ 已套用情境加成（法人連買 / 營收YoY / RS加速）")
        except Exception as _ctx_err:
            print(f"  ⚠ 情境加成失敗：{_ctx_err}")

        # ═══ C1: 多時框共振（短期 5/10 + 中期 20/60 動能驗證） ═══
        try:
            ranking = _apply_multi_timeframe(ranking)
            n_resonate = (ranking['多時框'] == '✅ 共振').sum() if '多時框' in ranking.columns else 0
            n_trap = (ranking['多時框'] == '⚠ 短強中弱').sum() if '多時框' in ranking.columns else 0
            print(f"  ✓ 多時框驗證完成（共振 {n_resonate} 檔、短強中弱誘多 {n_trap} 檔）")
        except Exception as _mtf_err:
            print(f"  ⚠ 多時框驗證失敗：{_mtf_err}")

        # ═══ C4: 反轉守門員（過濾高位陷阱） ═══
        try:
            ranking = _apply_anti_trap_filter(ranking)
            if '反轉風險' in ranking.columns:
                rs = ranking['反轉風險'].apply(
                    lambda v: float(str(v).rstrip('%')) / 100 if isinstance(v, str) and v.endswith('%') else None)
                n_high = ((rs >= 0.45) & rs.notna()).sum()
                n_top = ((rs >= 0.65) & rs.notna()).sum()
                print(f"  ✓ 反轉守門員：{n_high} 檔反轉風險偏高、其中 {n_top} 檔極高（已扣分重排）")
        except Exception as _at_err:
            print(f"  ⚠ 反轉守門員失敗：{_at_err}")

        # ═══ S18sec: 產業輪動（族群相對強度） ═══
        try:
            ranking = _apply_sector_rotation(ranking, taiex_state=market_state)
            if '族群強度' in ranking.columns:
                lead_mask = ranking['族群強度'].str.contains('龍頭', na=False)
                lag_mask = ranking['族群強度'].str.contains('落後', na=False)
                print(f"  ✓ 產業輪動：龍頭族群 {lead_mask.sum()} 檔、落後 {lag_mask.sum()} 檔")
        except Exception as _sr_err:
            print(f"  ⚠ 產業輪動失敗：{_sr_err}")

        # ═══════════════════════════════════════════════════════
        # Tier A/B/D：預先計算 Deep / Breakout / MC 預測，供下游共用
        # ═══════════════════════════════════════════════════════
        deep_preds = {}
        breakout_preds = {}
        mc_probs = {}
        shared_ohlcv_cache = {}
        # 候選池：所有策略命中 + all_stock_ids 前 200（兼顧效率）
        candidate_sids = set()
        for _df in buy_dfs_map.values():
            if _df is not None and not _df.empty and '證券代號' in _df.columns:
                candidate_sids.update(_df['證券代號'].astype(str).str.strip())
        # 也包含持股、低估股候選（先下載 OHLCV）
        candidate_sids.update(str(s).strip() for s in all_stock_ids[:300])
        candidate_sids = [s for s in candidate_sids if s]

        if candidate_sids:
            print(f"\n=== 預算 Deep / Breakout / MC 預測 ({len(candidate_sids)} 檔候選池) ===")
            try:
                # B6：優先重用 prefetch_technicals 抓過的 OHLCV，缺的才補抓
                shared_ohlcv_cache = get_shared_ohlcv(candidate_sids, period='9mo')
                hit_from_prefetch = sum(1 for s in candidate_sids if s in _raw_ohlcv_cache)
                print(f"  OHLCV 共用快取：{len(shared_ohlcv_cache)} 檔 "
                      f"（重用 prefetch {hit_from_prefetch} 檔，省下重複下載）")
            except Exception as e:
                print(f"  ⚠ 共用 OHLCV 取得失敗（各模組將各自下載）：{e}")

            # Deep 預測（含 conformal 區間）
            try:
                import deep_ranker as dr
                deep_preds = dr.predict_all_for_stocks(
                    candidate_sids, ohlcv_cache=shared_ohlcv_cache,
                )
                print(f"  Deep: {len(deep_preds)} 檔有預測")
            except Exception as e:
                print(f"  ⚠ Deep 預測失敗：{e}")

            # Breakout 預測（T+10 ≥ +10% 機率）
            try:
                import breakout_classifier as bc
                breakout_preds = bc.predict_breakout(
                    candidate_sids, ohlcv_cache=shared_ohlcv_cache,
                )
                print(f"  Breakout: {len(breakout_preds)} 檔有機率")
            except Exception as e:
                print(f"  · Breakout 未啟用（{e}）")

            # Monte Carlo 情境模擬
            try:
                import mc_simulator as mcs
                mc_probs = mcs.simulate_all(
                    candidate_sids, ohlcv_cache=shared_ohlcv_cache,
                    verbose=True,
                )
                print(f"  MC: {len(mc_probs)} 檔有模擬機率")
            except Exception as e:
                print(f"  · MC 模擬失敗（{e}）")

            # D16：deep / breakout 都已用過共用 OHLCV，可釋放特徵快取
            try:
                import deep_ranker as _dr
                _dr.clear_feature_cache(id(shared_ohlcv_cache))
            except Exception:
                pass

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
            # 用預算 deep_preds 加速
            ranking_deep, used_deep = apply_deep_ranking(
                ranking, deep_weight=0.35,
                precomputed_preds=deep_preds if deep_preds else None,
            )
            if used_deep:
                print("  ✓ 已採用 三層融合排名（rule + ml + deep）")
                ranking = ranking_deep
            else:
                print("  · 未啟用 Deep（找不到 deep_model.pt 或無資料），沿用現有排名")
        except Exception as _dp_err:
            import traceback; traceback.print_exc()
            print(f"  ⚠ Deep 融合失敗，沿用現有排名：{_dp_err}")

        # ═══ Tier B3: cohort-relative 百分位調整 ═══
        if deep_preds:
            try:
                ranking = _apply_cohort_relative(ranking, deep_preds)
            except Exception as _cr_err:
                print(f"  ⚠ cohort-relative 失敗：{_cr_err}")

        # ═══ Tier B1/D1: 把 breakout / MC 機率加到 ranking 的 top 候選 ═══
        if not ranking.empty and (breakout_preds or mc_probs):
            sid_key = ranking['證券代號'].astype(str).str.strip()
            if breakout_preds:
                ranking['突破機率(10d)'] = sid_key.map(
                    lambda s: f"{breakout_preds[s]*100:.0f}%" if s in breakout_preds else '—')
            if mc_probs:
                ranking['MC P(+10%)'] = sid_key.map(
                    lambda s: f"{mc_probs[s].get('p_up_10', 0)*100:.0f}%" if s in mc_probs else '—')
                ranking['MC P(+15%)'] = sid_key.map(
                    lambda s: f"{mc_probs[s].get('p_up_15', 0)*100:.0f}%" if s in mc_probs else '—')

        # ═══ Tier D4 / C3: Stacking 元模型做最終融合（regime-aware） ═══
        try:
            from stacking_meta import StackingBlender, detect_regime
            blender = StackingBlender.load()
            market_tag = detect_regime(market_state)
            print(f"  · 市場 regime 判斷：{market_tag}（20日報酬 "
                  f"{(market_state.get('return_20d', 0) or 0)*100:.1f}%，"
                  f"bullish={market_state.get('bullish')}）")
            if not ranking.empty:
                ranking = blender.blend_dataframe(
                    ranking,
                    rule_col='綜合分數' if '綜合分數' in ranking.columns else '規則分數',
                    ml_col='ML 分數' if 'ML 分數' in ranking.columns else None,
                    deep_col='Deep 分數' if 'Deep 分數' in ranking.columns else None,
                    up_prob_col='上漲機率(20d)' if '上漲機率(20d)' in ranking.columns else None,
                    market_state=market_tag,
                    out_col='元模型分數',
                )
                # 重新排序
                if '元模型分數' in ranking.columns:
                    ranking = ranking.sort_values('元模型分數', ascending=False).reset_index(drop=True)
                    if '排名' in ranking.columns:
                        ranking['排名'] = range(1, len(ranking) + 1)
                print(f"  ✓ Stacking 元模型融合完成（市場狀態：{market_tag}）")
        except Exception as _st_err:
            print(f"  ⚠ Stacking 融合失敗：{_st_err}")

        # ═══ C23：附加歷史 OOS 分位推估（rank 加上「分位 / 歷史同分位 OOS / 信心」）═══
        try:
            ranking = annotate_ranking_with_oos_percentile(ranking)
            if '歷史同分位OOS' in ranking.columns:
                top10 = ranking.head(10)
                top_oos = top10['歷史同分位OOS'].tolist()
                print(f"  ✓ C23 OOS 分位附加（Top10 同分位歷史 forward 報酬：{top_oos}）")
        except Exception as _c23_err:
            print(f"  · C23 OOS 分位附加失敗（樣本可能不足）：{_c23_err}")

        print("\n篩選低估股...")
        undervalued = find_undervalued(all_data, deep_preds=deep_preds)
        if blacklist:
            undervalued = _apply_blacklist(undervalued, blacklist, name='低估股')

        print("\n偵測潛力股提前佈局...")
        early_potential = find_early_potential(
            all_data, deep_preds=deep_preds, breakout_preds=breakout_preds,
        )
        if blacklist:
            early_potential = _apply_blacklist(early_potential, blacklist, name='潛力股')

        # Tier D3: 對 TOP 5 做 IG 解釋
        ig_map = {}
        if deep_preds and not ranking.empty:
            try:
                ig_map = _compute_top_ig_explanations(ranking, shared_ohlcv_cache, top_n=5)
                if ig_map:
                    print(f"  ✓ IG 解釋完成（{len(ig_map)} 檔）")
            except Exception as _ig_err:
                print(f"  ⚠ IG 計算失敗：{_ig_err}")

        # D19：當日 ranking 快照（保存供日後配對）
        try:
            _save_ranking_snapshot(ranking, market_state)
        except Exception as _snap_err:
            print(f"  · 排名快照失敗：{_snap_err}")

        # D19：把 trade_history.csv 中已關閉的交易，配對歷史快照後寫入 stacking_train_log.csv
        try:
            n_added = integrate_trade_history_to_stacking()
            if n_added:
                print(f"  · D19：整合 {n_added} 筆交易進 {STACKING_TRAIN_LOG_FILE}")
        except Exception as _int_err:
            print(f"  · stacking 訓練資料整合失敗：{_int_err}")

        # C10：Stacking meta 自動 retrain（樣本足 + 距上次訓練 ≥ 7 天）
        try:
            _maybe_retrain_stacking_meta(min_samples=200, min_days=7)
        except Exception as _rt_err:
            print(f"  · C10 自動 retrain 失敗：{_rt_err}")

        # C6 lite：OOS 滾動精準監控（10 個交易日 forward return）
        oos_summary = None
        try:
            oos_summary = compute_oos_monitor(window_days=10, top_n=10)
            if oos_summary:
                print(f"  · C6 OOS 監控（過去 {oos_summary['n_dates']} 個快照日）："
                      f"Top-10 平均 {oos_summary['mean_ret']*100:+.2f}%，"
                      f"勝率 {oos_summary['hit_rate']*100:.0f}%")
            else:
                print("  · C6 OOS 監控：尚無足夠歷史快照（需累積 10+ 日）")
        except Exception as _oos_err:
            print(f"  · OOS 監控失敗：{_oos_err}")

        # C8：OOS IC attribution（給下次跑用）
        try:
            ic_payload = compute_oos_ic_attribution(window_days=10, lookback_days=60)
            if ic_payload:
                ic = ic_payload['ic']
                pretty = ', '.join(f"{k}={v:+.3f}" for k, v in ic.items())
                print(f"  · C8 OOS IC（n={ic_payload['n_samples']}）：{pretty}")
            else:
                print("  · C8 OOS IC：樣本不足（需 ≥ 30 筆有效配對）")
        except Exception as _ic_err:
            print(f"  · C8 IC 計算失敗：{_ic_err}")

        # C11：保存當日各策略 hits + 計算歷史命中率
        strategy_oos_summary = None
        try:
            _save_strategy_snapshot(buy_dfs_map)
            strategy_oos_summary = compute_strategy_oos(window_days=10, max_history_days=180)
            if strategy_oos_summary:
                top3 = sorted(strategy_oos_summary.items(),
                              key=lambda kv: kv[1]['mean_ret'], reverse=True)[:3]
                print(f"  · C11 各策略歷史 OOS（{len(strategy_oos_summary)} 個策略已累積）")
                for sname, st in top3:
                    print(f"    {sname:8s}  mean {st['mean_ret']*100:+.2f}%  "
                          f"hit {st['hit_rate']*100:.0f}%  n={st['n_signals']}")
            else:
                print("  · C11 策略 OOS：尚無足夠歷史快照")
        except Exception as _soos_err:
            print(f"  · C11 策略 OOS 失敗：{_soos_err}")

        # C18：策略衰退偵測（最近 60D vs 180D）
        try:
            decay = compute_strategy_decay(recent_window_days=60,
                                           full_window_days=180,
                                           min_signals=8)
            decayed = [k for k, v in (decay or {}).items() if v.get('decayed')]
            if decayed:
                print(f"  · C18 策略衰退偵測：{len(decayed)} 個策略表現惡化 → 自動降權 0.5x")
                for sname in decayed[:5]:
                    st = decay[sname]
                    print(f"    {sname:8s}  full {st['full_mean']*100:+.2f}% → "
                          f"recent {st['recent_mean']*100:+.2f}%（drop {st['drop']*100:+.2f}pp）")
            elif decay:
                print(f"  · C18 策略衰退偵測：所有 {len(decay)} 個策略表現穩定")
        except Exception as _decay_err:
            print(f"  · C18 策略衰退偵測失敗：{_decay_err}")

        # C22：Top-N 月度再平衡投組回測
        portfolio_bt = None
        try:
            print("\n📈 C22：Top-N 投組回測（月度再平衡）...")
            portfolio_bt = backtest_topn_portfolio(top_n=10, rebalance_days=20,
                                                   lookback_days=365)
            if portfolio_bt:
                m = portfolio_bt['metrics']
                print(f"  ✓ 期間 {portfolio_bt['period']['start']} ~ {portfolio_bt['period']['end']}")
                print(f"    總報酬 {m['total_return']*100:+.2f}% / 年化 {m['ann_return']*100:+.2f}% / "
                      f"Sharpe {m['sharpe']:.2f} / Max DD {m['max_drawdown']*100:.2f}% / "
                      f"勝率 {m['win_rate']*100:.1f}% (n={m['n_trades']})")
            else:
                print("  · 歷史快照不足 / 缺欄位，略過")
        except Exception as _pbt_err:
            print(f"  · C22 投組回測失敗：{_pbt_err}")

        save_html_report(
            sell_alerts, buy_dfs_map,
            ranking=ranking, undervalued=undervalued,
            early_potential=early_potential,
            backtest_result=bt_result,
            all_data=all_data, revenue_data=revenue_data,
            market_state=market_state,
            holdings_df=holdings,
            portfolio_risk=portfolio_risk,
            ig_map=ig_map, mc_probs=mc_probs,
            deep_preds=deep_preds, breakout_preds=breakout_preds,
            oos_summary=oos_summary,
            strategy_oos_summary=strategy_oos_summary,
            portfolio_bt=portfolio_bt,
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

"""
XGBoost GPU Learning-to-Rank 股票排序模型
————————————————————————————
取代 Strategy_twe.compute_composite_ranking 的手工加權邏輯，
以 LambdaRank (rank:ndcg) 學習全市場每日排名對應的未來 T+20 報酬。

訓練流程：
    python ml_ranker.py train --period 3y --stocks auto      # 從 Strategy_twe 取全市場
    python ml_ranker.py train --period 3y --stocks demo      # 用內建示範股票池
    python ml_ranker.py train --period 3y --stocks 2330 2317 ...

推論流程（由 Strategy_twe.py 自動呼叫）：
    from ml_ranker import predict_ranks, load_model
    model = load_model()
    ranking_df = predict_ranks(model, live_features_df)

設計重點：
    - 每個交易日 = 一個 LambdaRank group
    - 特徵全為技術面 + 策略訊號 binary + 相對大盤動能（不依賴歷史法人資料）
    - Train/valid 以時間序列切分（前 80% 訓練，後 20% 驗證）
    - GPU 訓練（XGBoost 3.x + device='cuda'）RTX 5080 約 1-3 分鐘
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

import backtest as bt


MODEL_FILE = 'ranker_model.json'
FEATURES_META_FILE = 'ranker_features.json'
DEFAULT_HORIZON = 20           # 預測 T+N 日後報酬
DEFAULT_TRAIN_PCT = 0.8        # 時間序列切分比例
DEFAULT_PERIOD = '3y'


# ──────────────────────────────────────────────────────────
# 特徵工程
# ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Returns (過去報酬)
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
    # 波動
    'vol_5d', 'vol_20d',
    # MA 相對位置
    'close_ma5', 'close_ma10', 'close_ma20',
    'ma5_ma10', 'ma10_ma20',
    # RSI
    'rsi', 'rsi_diff5',
    # KD
    'k', 'd', 'kd_diff',
    # MACD
    'macd_hist', 'mhist_diff',
    # 布林
    'bb_pos', 'bb_width',
    # ATR
    'atr_pct', 'atr_trend',
    # 52w / 20d 結構
    'dist_52w', 'dist_20d_h', 'dist_20d_l',
    # 量能
    'vol_ratio', 'volma_trend',
    # 相對大盤
    'rs_5d', 'rs_20d',
    # 策略訊號（binary）
    'sig_s1_s3', 'sig_s4', 'sig_s5', 'sig_s6', 'sig_s9', 'sig_s10',
    'sig_s11', 'sig_s12', 'sig_s13',
]


def add_ml_features(df: pd.DataFrame, taiex_close: Optional[pd.Series] = None):
    """在已經 _add_indicators 過的 df 上加上 ML 特徵（就地修改）。"""
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    df['ret_1d'] = c.pct_change(1)
    df['ret_5d'] = c.pct_change(5)
    df['ret_10d'] = c.pct_change(10)
    df['ret_20d'] = c.pct_change(20)

    df['vol_5d'] = df['ret_1d'].rolling(5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20).std()

    df['close_ma5'] = c / df['MA5'].replace(0, np.nan) - 1
    df['close_ma10'] = c / df['MA10'].replace(0, np.nan) - 1
    df['close_ma20'] = c / df['MA20'].replace(0, np.nan) - 1
    df['ma5_ma10'] = df['MA5'] / df['MA10'].replace(0, np.nan) - 1
    df['ma10_ma20'] = df['MA10'] / df['MA20'].replace(0, np.nan) - 1

    df['rsi'] = df['RSI']
    df['rsi_diff5'] = df['RSI'] - df['RSI'].shift(5)

    df['k'] = df['K']
    df['d'] = df['D']
    df['kd_diff'] = df['K'] - df['D']

    df['macd_hist'] = df['MACD_hist']
    df['mhist_diff'] = df['MACD_hist'] - df['MACD_hist'].shift(1)

    bb_range = (df['BB_upper'] - df['BB_lower']).replace(0, np.nan)
    df['bb_pos'] = (c - df['BB_lower']) / bb_range
    df['bb_width'] = df['BB_width']

    df['atr_pct'] = df['ATR'] / c.replace(0, np.nan)
    df['atr_trend'] = df['ATR'] / df['ATR'].rolling(20).mean().replace(0, np.nan) - 1

    df['dist_52w'] = c / df['High_52w'].replace(0, np.nan) - 1
    df['dist_20d_h'] = c / h.rolling(20).max().replace(0, np.nan) - 1
    df['dist_20d_l'] = c / l.rolling(20).min().replace(0, np.nan) - 1

    df['vol_ratio'] = df['Vol_ratio']
    df['volma_trend'] = (v.rolling(5).mean() /
                         v.rolling(20).mean().replace(0, np.nan)) - 1

    # 相對大盤動能
    if taiex_close is not None:
        stock_r5 = c.pct_change(5)
        stock_r20 = c.pct_change(20)
        idx_r5 = taiex_close.reindex(df.index).pct_change(5)
        idx_r20 = taiex_close.reindex(df.index).pct_change(20)
        df['rs_5d'] = stock_r5 - idx_r5
        df['rs_20d'] = stock_r20 - idx_r20
    else:
        df['rs_5d'] = 0.0
        df['rs_20d'] = 0.0

    # 策略訊號（用 backtest 的 signal_fn 向量化近似）
    df['sig_s1_s3'] = _vec_signal(df, bt._signal_s1_s3)
    df['sig_s4'] = _vec_signal(df, bt._signal_s4)
    df['sig_s5'] = _vec_signal(df, bt._signal_s5)
    df['sig_s6'] = _vec_signal(df, bt._signal_s6)
    df['sig_s9'] = _vec_signal(df, bt._signal_s9)
    df['sig_s10'] = _vec_signal(df, bt._signal_s10)
    df['sig_s11'] = _vec_signal(df, bt._signal_s11)
    df['sig_s12'] = _vec_signal(df, bt._signal_s12)
    df['sig_s13'] = _vec_signal(df, bt._signal_s13)

    return df


def _vec_signal(df: pd.DataFrame, signal_fn) -> np.ndarray:
    """將 backtest 的單日 signal_fn 套用到整支股票，回傳 0/1 陣列。"""
    out = np.zeros(len(df), dtype=np.float32)
    records = df.to_dict('records')
    prev = None
    for i, row in enumerate(records):
        try:
            if signal_fn(row, prev):
                out[i] = 1.0
        except (KeyError, TypeError):
            pass
        prev = row
    return out


# ──────────────────────────────────────────────────────────
# 訓練資料建構
# ──────────────────────────────────────────────────────────

def _download_taiex(period: str) -> pd.Series:
    print(f"  下載 TAIEX 指數 (^TWII, {period})...")
    try:
        raw = yf.download('^TWII', period=period, progress=False)
        raw = bt._flatten_columns(raw)
        if not raw.empty:
            return raw['Close']
    except Exception as e:
        print(f"  ⚠ TAIEX 下載失敗: {e}")
    return pd.Series(dtype=float)


def build_training_dataset(
    stock_ids: Sequence[str],
    period: str = DEFAULT_PERIOD,
    horizon: int = DEFAULT_HORIZON,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """回傳 (features_df, labels, dates_series)
    features_df 包含所有 FEATURE_COLS + 'date' + 'stock_id'，按日期排序。
    labels 為 T+horizon 報酬，dates_series 用於後續 group 切分。"""
    print(f"\n===== 建立訓練資料 ({len(stock_ids)} 股, {period}, horizon=T+{horizon}) =====")
    t0 = time.time()

    cache = bt._batch_download(list(stock_ids), period=period)
    print(f"  成功下載 {len(cache)} 支")
    taiex = _download_taiex(period)

    all_rows = []
    for sid, df in cache.items():
        if len(df) < 120:
            continue
        bt._add_indicators(df)
        add_ml_features(df, taiex_close=taiex if not taiex.empty else None)

        # 標籤：T+horizon 報酬
        df['_label'] = df['Close'].shift(-horizon) / df['Close'] - 1

        sub = df.dropna(subset=FEATURE_COLS + ['_label']).copy()
        if sub.empty:
            continue
        sub['stock_id'] = sid
        sub['date'] = sub.index
        all_rows.append(sub[['date', 'stock_id', '_label'] + FEATURE_COLS])

    if not all_rows:
        raise RuntimeError("無有效訓練資料，請確認下載與指標計算是否正常")

    full = pd.concat(all_rows, ignore_index=True)
    full = full.sort_values(['date', 'stock_id']).reset_index(drop=True)

    labels = full['_label'].values.astype(np.float32)
    X = full[FEATURE_COLS].astype(np.float32)
    dates = full['date']

    print(f"  資料集: {len(full):,} rows × {len(FEATURE_COLS)} 特徵, "
          f"涵蓋 {dates.min().date()} ~ {dates.max().date()}")
    print(f"  建立耗時: {time.time() - t0:.1f}s")
    return full, labels, dates


# ──────────────────────────────────────────────────────────
# 排名標籤轉換
# ──────────────────────────────────────────────────────────

def _returns_to_relevance(labels: np.ndarray, dates: pd.Series) -> np.ndarray:
    """把每日的連續報酬轉成 0-4 離散相關度分數（NDCG 用）。
    依該日所在組內分位數：top20%=4, next20%=3, ..., bottom20%=0。"""
    rel = np.zeros(len(labels), dtype=np.int32)
    df = pd.DataFrame({'label': labels, 'date': dates.values})
    # 每日組內分位數
    df['qnt'] = df.groupby('date')['label'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
        if len(x) >= 5 else 2
    )
    df['qnt'] = df['qnt'].fillna(2).astype(int)
    return df['qnt'].values.astype(np.int32)


def _build_groups(dates: pd.Series) -> np.ndarray:
    """每日一個 group，回傳每 group 的大小陣列（LambdaRank 需要）。"""
    return dates.groupby(dates.values).size().sort_index().values.astype(np.int32)


# ──────────────────────────────────────────────────────────
# 訓練
# ──────────────────────────────────────────────────────────

def train_ranker(
    X: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    train_pct: float = DEFAULT_TRAIN_PCT,
    n_estimators: int = 1000,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> xgb.Booster:
    """以時間序列切分訓練 LambdaRank (NDCG) 模型。"""
    unique_dates = dates.drop_duplicates().sort_values().values
    split_idx = int(len(unique_dates) * train_pct)
    split_date = unique_dates[split_idx]
    print(f"\n===== 訓練 XGBoost Ranker =====")
    print(f"  訓練區間: < {pd.Timestamp(split_date).date()} "
          f"({split_idx} 個交易日)")
    print(f"  驗證區間: >= {pd.Timestamp(split_date).date()} "
          f"({len(unique_dates) - split_idx} 個交易日)")

    train_mask = dates.values < split_date
    valid_mask = ~train_mask

    # 相關度標籤
    relevance = _returns_to_relevance(labels, dates)

    X_train = X[train_mask]
    X_valid = X[valid_mask]
    y_train = relevance[train_mask]
    y_valid = relevance[valid_mask]
    dates_train = dates[train_mask]
    dates_valid = dates[valid_mask]

    g_train = _build_groups(dates_train)
    g_valid = _build_groups(dates_valid)

    dtrain = xgb.DMatrix(X_train.values, label=y_train)
    dtrain.set_group(g_train)
    dvalid = xgb.DMatrix(X_valid.values, label=y_valid)
    dvalid.set_group(g_valid)

    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'rank:ndcg',
        'eval_metric': ['ndcg@10', 'ndcg@30'],
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': random_state,
    }

    print(f"  裝置: cuda | rounds={n_estimators} | max_depth={max_depth} | lr={learning_rate}")
    t0 = time.time()
    evals_result = {}
    booster = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50,
    )
    elapsed = time.time() - t0
    print(f"  訓練完成: {elapsed:.1f}s (best_iter={booster.best_iteration})")

    # 重要度
    imp = booster.get_score(importance_type='gain')
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 15 特徵重要度 (by gain):")
    for i, (feat, score) in enumerate(imp_sorted[:15]):
        feat_name = FEATURE_COLS[int(feat[1:])] if feat.startswith('f') else feat
        print(f"    {i+1:2d}. {feat_name:<15} {score:>10.1f}")

    return booster


# ──────────────────────────────────────────────────────────
# 推論
# ──────────────────────────────────────────────────────────

def save_model(model: xgb.Booster, path: str = MODEL_FILE):
    model.save_model(path)
    meta = {
        'features': FEATURE_COLS,
        'horizon': DEFAULT_HORIZON,
        'trained_at': pd.Timestamp.now().isoformat(),
    }
    with open(FEATURES_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  模型已儲存: {path}")


def load_model(path: str = MODEL_FILE) -> Optional[xgb.Booster]:
    if not os.path.exists(path):
        return None
    try:
        model = xgb.Booster()
        model.load_model(path)
        return model
    except Exception as e:
        print(f"⚠ 載入 {path} 失敗: {e}")
        return None


def predict_scores(model: xgb.Booster, features_df: pd.DataFrame) -> np.ndarray:
    """對一個 DataFrame（每列一支股票）預測 ranker 分數。
    features_df 必須包含 FEATURE_COLS 所有欄位。"""
    X = features_df[FEATURE_COLS].astype(np.float32).fillna(0)
    dtest = xgb.DMatrix(X.values)
    return model.predict(dtest)


def build_live_features(
    stock_ids: Sequence[str],
    ohlcv_cache: Optional[dict] = None,
    taiex_close: Optional[pd.Series] = None,
    period: str = '6mo',
) -> pd.DataFrame:
    """為當日推論建立特徵 DataFrame：每列一支股票（取該股最新一筆）。"""
    if ohlcv_cache is None:
        ohlcv_cache = bt._batch_download(list(stock_ids), period=period)
    if taiex_close is None or taiex_close.empty:
        taiex_close = _download_taiex(period)

    rows = []
    for sid in stock_ids:
        df = ohlcv_cache.get(sid)
        if df is None or len(df) < 60:
            continue
        df_copy = df.copy()
        if 'ATR' not in df_copy.columns:
            bt._add_indicators(df_copy)
        add_ml_features(df_copy,
                        taiex_close=taiex_close if not taiex_close.empty else None)
        # 取最新一筆（確保所有特徵都有值）
        recent = df_copy[FEATURE_COLS].dropna()
        if recent.empty:
            continue
        last = recent.iloc[-1]
        row = last.to_dict()
        row['stock_id'] = sid
        row['date'] = recent.index[-1]
        row['close'] = df_copy['Close'].iloc[-1]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['stock_id', 'date', 'close'] + FEATURE_COLS)
    return pd.DataFrame(rows)


def predict_ranks(
    model: xgb.Booster,
    features_df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """給定當日特徵 DataFrame，回傳前 N 名（按 ml_score 排序）。"""
    if features_df.empty:
        return features_df.assign(ml_score=[]).head(0)
    scores = predict_scores(model, features_df)
    result = features_df.copy()
    result['ml_score'] = scores
    result = result.sort_values('ml_score', ascending=False).reset_index(drop=True)
    return result.head(top_n)


# ──────────────────────────────────────────────────────────
# CLI / main
# ──────────────────────────────────────────────────────────

def _resolve_stocks(choice: str, extra: List[str]) -> List[str]:
    if choice == 'auto':
        try:
            import Strategy_twe as stw
            all_data = stw.get_recent_institutional()
            ids = all_data['證券代號'].astype(str).str.strip().unique().tolist()
            print(f"  已從 Strategy_twe 載入 {len(ids)} 檔全市場代號")
            return ids
        except Exception as e:
            print(f"  ⚠ auto 載入失敗 ({e})，改用 demo")
            return _demo_stocks()
    if choice == 'demo':
        return _demo_stocks()
    if extra:
        return extra
    return _demo_stocks()


def _demo_stocks() -> List[str]:
    """用 60 檔流通大型股做 MVP 訓練，平衡速度與代表性。"""
    return [
        # 台灣 50 成分股主要代表
        '2330', '2317', '2454', '2308', '2881', '2882', '2303', '3711',
        '2412', '2886', '1301', '2891', '3008', '2357', '6505', '2002',
        '2884', '1303', '2885', '1216', '2892', '2207', '2880', '2887',
        '5871', '2890', '2609', '1590', '3045', '2912', '1326', '2603',
        '2610', '2615', '3231', '2379', '6669', '3034', '4938', '2408',
        '3037', '2327', '2474', '6505', '1402', '1102', '2801', '2834',
        '2345', '2301', '2377', '2354', '2395', '2382', '2383', '3706',
        '3017', '8046', '6488', '8454',
    ]


def cmd_train(args):
    stock_ids = _resolve_stocks(args.stocks, args.extra)
    X_full, labels, dates = build_training_dataset(
        stock_ids, period=args.period, horizon=args.horizon
    )
    X = X_full[FEATURE_COLS]
    booster = train_ranker(
        X, labels, dates,
        train_pct=args.train_pct,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        early_stopping_rounds=args.early_stop,
    )
    save_model(booster, args.model_out)
    return booster


def cmd_predict(args):
    model = load_model(args.model_out)
    if model is None:
        raise SystemExit(f"❌ 模型 {args.model_out} 不存在，請先 train")
    stock_ids = _resolve_stocks(args.stocks, args.extra)
    features = build_live_features(stock_ids, period=args.period)
    if features.empty:
        raise SystemExit("❌ 無可用特徵")
    result = predict_ranks(model, features, top_n=args.top_n)
    disp_cols = ['stock_id', 'date', 'close', 'ml_score',
                 'rsi', 'close_ma20', 'rs_20d', 'mom_20d' if 'mom_20d' in result.columns else 'ret_20d']
    disp_cols = [c for c in disp_cols if c in result.columns]
    print(f"\n===== TOP {args.top_n} ML ranking =====")
    print(result[disp_cols].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description='XGBoost GPU 排序模型')
    sub = p.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train', help='訓練模型')
    pt.add_argument('--stocks', default='demo', choices=['auto', 'demo', 'custom'],
                    help='auto=當日全市場 / demo=示範 60 檔 / custom=用 extra')
    pt.add_argument('--extra', nargs='*', default=[], help='自訂股票代號')
    pt.add_argument('--period', default=DEFAULT_PERIOD)
    pt.add_argument('--horizon', type=int, default=DEFAULT_HORIZON)
    pt.add_argument('--train-pct', type=float, default=DEFAULT_TRAIN_PCT)
    pt.add_argument('--n-estimators', type=int, default=1000)
    pt.add_argument('--max-depth', type=int, default=6)
    pt.add_argument('--lr', type=float, default=0.05)
    pt.add_argument('--early-stop', type=int, default=50)
    pt.add_argument('--model-out', default=MODEL_FILE)
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser('predict', help='推論 TOP N')
    pp.add_argument('--stocks', default='auto', choices=['auto', 'demo', 'custom'])
    pp.add_argument('--extra', nargs='*', default=[])
    pp.add_argument('--period', default='6mo')
    pp.add_argument('--top-n', type=int, default=30)
    pp.add_argument('--model-out', default=MODEL_FILE)
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

"""Tier D4：Stacking Meta Learner

取代 Strategy_twe 裡「rule 40% + ml 60% → × 65% + deep 35%」的硬編碼權重，
以 **regime-aware 動態權重** + **可訓練元模型** 的組合：

Ⅰ. 預設：市場狀態感知加權（無需訓練即可使用）
    - 多頭 + 模型高信心 → deep 權重拉高
    - 震盪 + 模型不確定 → 退回規則分數
    - 空頭 → 最保守，全信規則 + 基本面

Ⅱ. 可選：XGBoost 元模型（用 accumulate 的歷史預測 vs 實際報酬訓練）
    儲存於 `stacking_meta.json`；若存在自動啟用，否則用預設。

使用：
    from stacking_meta import StackingBlender
    blender = StackingBlender.load()
    final = blender.blend(rule=70, ml=80, deep=65, up_prob=0.62, market='bull')
    # 或批次：
    df_out = blender.blend_dataframe(df_scores, market_state='bull')
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


META_FILE = 'stacking_meta.json'
TRAIN_LOG_FILE = 'stacking_train_log.csv'


# ──────────────────────────────────────────────────────────
# Regime-aware default weights
# ──────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    # 市場狀態 → (rule, ml, deep) 權重和=1
    'bull':    (0.25, 0.35, 0.40),  # 順風用模型
    'neutral': (0.35, 0.35, 0.30),  # 平衡
    'bear':    (0.55, 0.30, 0.15),  # 逆風退回規則
    'shock':   (0.70, 0.25, 0.05),  # C3：高波動+空頭 → 幾乎全靠規則 + 基本面
    'unknown': (0.40, 0.35, 0.25),
}


# ──────────────────────────────────────────────────────────
# C3：Regime detection（多維判斷大盤狀態）
# ──────────────────────────────────────────────────────────

def detect_regime(market_state: dict) -> str:
    """根據 TAIEX 多維特徵判斷市場 regime：bull / neutral / bear / shock。

    輸入 market_state（來自 Strategy_twe.get_taiex_state()）需含：
        bullish (bool), close (float), ma_60 (float), return_20d (float),
        close_history_30 (list, 至少 21 點)
    """
    if not market_state:
        return 'unknown'
    ret20 = market_state.get('return_20d') or 0.0
    bullish = market_state.get('bullish', False)
    close_hist = market_state.get('close_history_30') or []

    # 1. 計算 20 日年化波動（>20%/年 視為高波動）
    realized_vol = None
    if len(close_hist) >= 21:
        try:
            arr = np.array(close_hist[-21:], dtype=float)
            if (arr > 0).all():
                rets = np.diff(np.log(arr))
                realized_vol = float(np.std(rets) * np.sqrt(252))
        except Exception:
            pass

    high_vol = realized_vol is not None and realized_vol > 0.25

    # 2. 綜合判斷
    if not bullish and (ret20 < -0.04 or high_vol):
        return 'shock'
    if bullish and ret20 > 0.02:
        return 'bull'
    if (not bullish) and ret20 < -0.02:
        return 'bear'
    return 'neutral'

# 信心度調節（每檔股票的 up_prob 差異）
# up_prob 很高（>0.65）或很低（<0.35）→ deep 權重 +0.10
# up_prob 中間（0.45~0.55）→ deep 權重 -0.10
def _confidence_adjust(weights, up_prob):
    if up_prob is None:
        return weights
    r, m, d = weights
    if up_prob > 0.65 or up_prob < 0.35:
        adj = 0.10
    elif 0.45 <= up_prob <= 0.55:
        adj = -0.10
    else:
        adj = 0.0
    d_new = max(0.05, min(0.70, d + adj))
    # 比例縮減 r, m
    remain = 1 - d_new
    if r + m <= 0:
        return weights
    r_new = r / (r + m) * remain
    m_new = m / (r + m) * remain
    return (r_new, m_new, d_new)


# ──────────────────────────────────────────────────────────
# Blender
# ──────────────────────────────────────────────────────────

@dataclass
class StackingBlender:
    weights: Dict[str, tuple] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # 可選：XGBoost 模型（若有訓練）
    xgb_model: Optional[object] = None
    xgb_features: Optional[List[str]] = None
    trained_at: Optional[str] = None
    n_train: int = 0

    # ── 單檔融合 ──
    def blend(self, rule: float, ml: Optional[float], deep: Optional[float],
              up_prob: Optional[float] = None,
              market: str = 'unknown') -> float:
        """輸入 0-100 分，回傳融合後 0-100 分。若 ml/deep 缺值自動重分配權重。"""
        weights = self.weights.get(market, DEFAULT_WEIGHTS['unknown'])
        weights = _confidence_adjust(weights, up_prob)
        r, m, d = weights

        parts = {'rule': (rule, r)}
        if ml is not None:
            parts['ml'] = (ml, m)
        if deep is not None:
            parts['deep'] = (deep, d)

        # 重新 normalize 權重（缺項時）
        total_w = sum(w for _, w in parts.values()) or 1.0
        score = sum(v * (w / total_w) for v, w in parts.values())

        # 若有 xgb 元模型就用它
        if self.xgb_model is not None and self.xgb_features is not None:
            try:
                import xgboost as xgb
                feats = self._build_feature_vec(rule, ml, deep, up_prob, market)
                dm = xgb.DMatrix(np.array([feats]), feature_names=self.xgb_features)
                xgb_pred = float(self.xgb_model.predict(dm)[0])
                # 混合：80% 元模型 + 20% 規則加權（穩健）
                return 0.8 * np.clip(xgb_pred, 0, 100) + 0.2 * score
            except Exception:
                pass

        return float(score)

    def _build_feature_vec(self, rule, ml, deep, up_prob, market):
        regime_onehot = {
            'bull': [1, 0, 0],
            'neutral': [0, 1, 0],
            'bear': [0, 0, 1],
        }.get(market, [0, 1, 0])
        return [
            rule or 0,
            ml or 0,
            deep or 0,
            up_prob or 0.5,
            *regime_onehot,
        ]

    # ── 批次融合（DataFrame 版）──
    def blend_dataframe(self, df: pd.DataFrame,
                        rule_col: str = '綜合分數',
                        ml_col: str = 'ML 分數',
                        deep_col: str = 'Deep 分數',
                        up_prob_col: Optional[str] = 'Deep UpProb',
                        market_state: str = 'unknown',
                        out_col: str = '元模型分數') -> pd.DataFrame:
        if rule_col not in df.columns:
            raise ValueError(f"缺少欄位: {rule_col}")
        out = df.copy()
        scores = []
        for _, row in df.iterrows():
            rule = float(row.get(rule_col) or 0)
            ml = row.get(ml_col)
            ml = float(ml) if ml is not None and not pd.isna(ml) else None
            deep = row.get(deep_col)
            deep = float(deep) if deep is not None and not pd.isna(deep) else None

            up_prob = None
            if up_prob_col and up_prob_col in df.columns:
                v = row.get(up_prob_col)
                if isinstance(v, str) and v.endswith('%'):
                    try:
                        up_prob = float(v.rstrip('%')) / 100.0
                    except ValueError:
                        up_prob = None
                elif isinstance(v, (int, float)) and not pd.isna(v):
                    up_prob = float(v)
            scores.append(self.blend(rule, ml, deep, up_prob, market_state))
        out[out_col] = np.round(scores, 1)
        return out

    # ── 訓練（使用者累積足夠歷史資料後呼叫）──
    @classmethod
    def train(cls, rows: List[dict], min_samples: int = 200,
              time_decay_half_life_days: float = 90.0):
        """
        rows = [{'rule': 70, 'ml': 80, 'deep': 65, 'up_prob': 0.62,
                 'market': 'bull', 'actual_ret': 0.12,
                 'date': '2026-04-01' (optional)}, ...]
        target = actual_ret 映射到 0-100 分（rank normalize）

        C12：對含 'date' 欄位的 rows 加入 **時間衰減 sample weight**，
        最近資料權重 1.0，距今越遠以 half-life 衰減（exp(-ln2 × Δdays / half_life)）。
        """
        if len(rows) < min_samples:
            print(f"⚠ 樣本太少 ({len(rows)} < {min_samples})，無法訓練元模型")
            return None

        try:
            import xgboost as xgb
        except ImportError:
            print("⚠ 未安裝 xgboost，無法訓練元模型")
            return None

        X, y, dates = [], [], []
        features = ['rule', 'ml', 'deep', 'up_prob',
                    'regime_bull', 'regime_neutral', 'regime_bear']
        for r in rows:
            regime_onehot = {
                'bull': [1, 0, 0],
                'neutral': [0, 1, 0],
                'bear': [0, 0, 1],
            }.get(r.get('market', 'neutral'), [0, 1, 0])
            X.append([
                r.get('rule', 0) or 0,
                r.get('ml', 0) or 0,
                r.get('deep', 0) or 0,
                r.get('up_prob', 0.5) or 0.5,
                *regime_onehot,
            ])
            y.append(r['actual_ret'])
            dates.append(r.get('date') or r.get('日期'))

        X = np.array(X)
        y = np.array(y)
        y_rank = pd.Series(y).rank(pct=True).values * 100

        # C12：時間衰減 sample weight
        sample_weight = None
        if any(d is not None for d in dates) and time_decay_half_life_days > 0:
            try:
                ts = pd.to_datetime(pd.Series(dates), errors='coerce')
                if ts.notna().any():
                    latest = ts.max()
                    delta_days = (latest - ts).dt.days.fillna(9999).astype(float)
                    sample_weight = np.exp(-np.log(2) * delta_days.values
                                           / max(1.0, time_decay_half_life_days))
                    sample_weight = np.clip(sample_weight, 0.05, 1.0)
                    print(f"  · C12 時間衰減 weight：half_life={time_decay_half_life_days:.0f} 日，"
                          f"最近 sample weight 平均={sample_weight[-min(20, len(sample_weight)):].mean():.3f}，"
                          f"最舊 weight 平均={sample_weight[:min(20, len(sample_weight))].mean():.3f}")
            except Exception as _e:
                print(f"  · C12 時間衰減失敗，改用等權：{_e}")
                sample_weight = None

        split = int(len(X) * 0.8)
        dtrain_kwargs = dict(label=y_rank[:split], feature_names=features)
        dvalid_kwargs = dict(label=y_rank[split:], feature_names=features)
        if sample_weight is not None:
            dtrain_kwargs['weight'] = sample_weight[:split]
            dvalid_kwargs['weight'] = sample_weight[split:]
        dtrain = xgb.DMatrix(X[:split], **dtrain_kwargs)
        dvalid = xgb.DMatrix(X[split:], **dvalid_kwargs)

        # Tier A5：CUDA fallback CPU（XGBoost CUDA 版可能未裝）
        base_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'verbosity': 0,
        }
        booster = None
        for device_try in ('cuda', 'cpu'):
            params = dict(base_params, device=device_try)
            try:
                booster = xgb.train(
                    params, dtrain, num_boost_round=500,
                    evals=[(dvalid, 'valid')],
                    early_stopping_rounds=30,
                    verbose_eval=50,
                )
                print(f"  ✓ XGBoost 元模型訓練完成（device={device_try}）")
                break
            except Exception as e:
                msg = str(e).lower()
                if device_try == 'cuda' and ('cuda' in msg or 'gpu' in msg or 'device' in msg):
                    print(f"  ⚠ CUDA 不可用（{e}），改用 CPU 重試")
                    continue
                raise
        if booster is None:
            print("⚠ XGBoost 訓練失敗")
            return None
        blender = cls(
            weights=dict(DEFAULT_WEIGHTS),
            xgb_model=booster,
            xgb_features=features,
            trained_at=pd.Timestamp.now().isoformat(),
            n_train=len(X),
        )
        return blender

    # ── 儲存 / 載入 ──
    def save(self, path: str = META_FILE):
        data = {
            'weights': self.weights,
            'xgb_features': self.xgb_features,
            'trained_at': self.trained_at,
            'n_train': self.n_train,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.xgb_model is not None:
            try:
                self.xgb_model.save_model(path.replace('.json', '_xgb.json'))
            except Exception as e:
                print(f"⚠ xgb_model 存檔失敗: {e}")
        print(f"  元模型已儲存: {path}")

    @classmethod
    def load(cls, path: str = META_FILE) -> 'StackingBlender':
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            blender = cls(
                weights={k: tuple(v) for k, v in data.get('weights', DEFAULT_WEIGHTS).items()},
                xgb_features=data.get('xgb_features'),
                trained_at=data.get('trained_at'),
                n_train=data.get('n_train', 0),
            )
            # 嘗試載入 xgb 模型
            xgb_path = path.replace('.json', '_xgb.json')
            if os.path.exists(xgb_path) and blender.xgb_features:
                try:
                    import xgboost as xgb
                    booster = xgb.Booster()
                    booster.load_model(xgb_path)
                    blender.xgb_model = booster
                except Exception as e:
                    print(f"⚠ xgb_model 載入失敗: {e}")
            return blender
        except Exception as e:
            print(f"⚠ 載入 {path} 失敗: {e}")
            return cls()


# ──────────────────────────────────────────────────────────
# 訓練資料蒐集：從 backtest 結果自動產生
# ──────────────────────────────────────────────────────────

def collect_training_data_from_backtest(bt_result) -> List[dict]:
    """把 backtest 的 trades 轉成元模型訓練資料（簡化版）。
    假設 bt_result 帶有每筆交易的最終 return，與股票當時的 rule/ml/deep 分數。
    實務上需累積多次 Strategy_twe 執行紀錄到 CSV。"""
    # 暫時留空，使用者透過 accumulate 方式寫入 TRAIN_LOG_FILE
    return []


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Stacking 元模型訓練')
    sub = p.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train', help='從 stacking_train_log.csv 訓練元模型')
    pt.add_argument('--log', default=TRAIN_LOG_FILE)
    pt.add_argument('--min-samples', type=int, default=200)

    ps = sub.add_parser('status', help='顯示元模型狀態')

    args = p.parse_args()

    if args.cmd == 'status':
        b = StackingBlender.load()
        print(f"模型: {'已訓練' if b.xgb_model else '未訓練（使用預設 regime-aware 權重）'}")
        print(f"訓練時間: {b.trained_at or '—'}")
        print(f"樣本數: {b.n_train}")
        print(f"預設權重:")
        for k, v in b.weights.items():
            print(f"  {k:8s}: rule={v[0]:.2f} ml={v[1]:.2f} deep={v[2]:.2f}")
    elif args.cmd == 'train':
        if not os.path.exists(args.log):
            raise SystemExit(f"❌ 找不到 {args.log}；請先累積資料（每次 Strategy_twe 跑完會 append）")
        df = pd.read_csv(args.log)
        rows = df.to_dict(orient='records')
        blender = StackingBlender.train(rows, min_samples=args.min_samples)
        if blender is not None:
            blender.save()

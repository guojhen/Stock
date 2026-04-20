"""定時任務排程器：在每天執行 Strategy_twe.py 時自動偵測需要重訓/重算的任務。

任務種類與預設週期：
  • gpu_grid  : GPU 參數網格回測（產出 best_params.json）— 每 90 天（季度）
  • ml_train  : XGBoost Learning-to-Rank 重訓（產出 ranker_model.json）— 每 14 天
  • deep_train: PyTorch GRU 深度模型重訓（產出 deep_model.pt）— 每 14 天

狀態檔 `scheduler_state.json` 紀錄每個任務上次成功執行時間；若檔案不存在會
以「產物檔 mtime」作為 bootstrap 起點，避免首次執行就把全部重跑一次。

使用方式（由 Strategy_twe.py 自動呼叫）：
    from scheduler import run_scheduled_tasks
    run_scheduled_tasks()

環境變數：
    STRATEGY_SKIP_SCHEDULER=1            完全略過
    STRATEGY_FORCE_SCHEDULER=ml_train    強制執行指定任務（逗號分隔、all=全部）
    STRATEGY_DRY_RUN=1                   只顯示要做什麼、不實際執行
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

STATE_FILE = 'scheduler_state.json'


@dataclass
class Task:
    key: str
    description: str
    interval_days: int
    command: List[str]
    artifact: str  # 用來 bootstrap 最後執行時間
    timeout_sec: int = 3600  # 1 小時
    enabled: bool = True
    # 條件判斷：回傳 True 才會跑（例如只在週末跑重訓）
    condition: Optional[Callable[[datetime], bool]] = None


# ──────────────────────────────────────────────────────────
# 任務定義
# ──────────────────────────────────────────────────────────
def _is_weekend(now: datetime) -> bool:
    """只在週六(5)/週日(6)跑，避免平日盤中訓練影響主流程。"""
    return now.weekday() >= 5


TASKS: Dict[str, Task] = {
    'gpu_grid': Task(
        key='gpu_grid',
        description='GPU 參數網格回測（季度校準 best_params.json）',
        interval_days=90,
        command=[sys.executable, 'gpu_backtest.py', '--all',
                 '--period', '2y', '--top-k', '10'],
        artifact='best_params.json',
        timeout_sec=60 * 60,  # 60 分鐘
        condition=_is_weekend,
    ),
    'ml_train': Task(
        key='ml_train',
        description='XGBoost Ranker 重訓（ranker_model.json）',
        interval_days=14,
        command=[sys.executable, 'ml_ranker.py', 'train',
                 '--stocks', 'auto', '--period', '3y'],
        artifact='ranker_model.json',
        timeout_sec=30 * 60,
        condition=_is_weekend,
    ),
    'deep_train': Task(
        key='deep_train',
        description='PyTorch GRU 深度模型重訓（deep_model.pt）',
        interval_days=14,
        command=[sys.executable, 'deep_ranker.py', 'train',
                 '--stocks', 'auto', '--period', '3y', '--epochs', '40'],
        artifact='deep_model.pt',
        timeout_sec=60 * 60,
        condition=_is_weekend,
    ),
}


# ──────────────────────────────────────────────────────────
# 狀態檔 I/O
# ──────────────────────────────────────────────────────────
def _load_state() -> Dict[str, dict]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"  ⚠ 讀取 {STATE_FILE} 失敗：{e}，視為空狀態")
        return {}


def _save_state(state: Dict[str, dict]) -> None:
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ⚠ 寫入 {STATE_FILE} 失敗：{e}")


def _last_run(task: Task, state: Dict[str, dict]) -> Optional[datetime]:
    """取得任務上次執行時間；若無紀錄，以產物 mtime 當 bootstrap 起點。"""
    rec = state.get(task.key)
    if rec and rec.get('last_success'):
        try:
            return datetime.fromisoformat(rec['last_success'])
        except Exception:
            pass
    if os.path.exists(task.artifact):
        return datetime.fromtimestamp(os.path.getmtime(task.artifact))
    return None


def _should_run(task: Task, now: datetime,
                state: Dict[str, dict], forced: bool) -> tuple[bool, str]:
    """判斷任務是否該執行，回傳 (should, reason)。"""
    if not task.enabled:
        return False, '已停用'
    if forced:
        return True, '強制執行'
    last = _last_run(task, state)
    if last is None:
        return True, '首次執行（尚無產物）'
    age_days = (now - last).total_seconds() / 86400
    if age_days < task.interval_days:
        remaining = task.interval_days - age_days
        return False, f'距離下次執行還有 {remaining:.1f} 天（上次 {last:%Y-%m-%d}）'
    if task.condition and not task.condition(now):
        return False, f'已達週期但條件未滿足（例：非週末）；已等待 {age_days:.1f} 天'
    return True, f'已 {age_days:.1f} 天未執行（週期 {task.interval_days} 天）'


# ──────────────────────────────────────────────────────────
# 執行
# ──────────────────────────────────────────────────────────
def _run_task(task: Task, dry_run: bool = False) -> dict:
    """執行任務，回傳 {status, elapsed_sec, ...}。"""
    cmd_str = ' '.join(task.command)
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"▶ 排程任務：{task.key} — {task.description}")
    print(f"  指令：{cmd_str}")
    if dry_run:
        print(f"  (DRY RUN — 跳過實際執行)")
        return {'status': 'dry_run', 'elapsed_sec': 0}
    start = time.time()
    try:
        result = subprocess.run(
            task.command,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=task.timeout_sec,
            check=False,  # 不拋例外，讓我們看 returncode
            text=True,
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"✓ {task.key} 完成（耗時 {elapsed/60:.1f} 分鐘）")
            return {'status': 'success', 'elapsed_sec': elapsed,
                    'returncode': 0}
        else:
            print(f"✗ {task.key} 失敗（returncode={result.returncode}，"
                  f"耗時 {elapsed/60:.1f} 分鐘）")
            return {'status': 'failed', 'elapsed_sec': elapsed,
                    'returncode': result.returncode}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"⏱ {task.key} 逾時（>{task.timeout_sec/60:.0f} 分鐘）")
        return {'status': 'timeout', 'elapsed_sec': elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ {task.key} 例外：{e}")
        return {'status': 'error', 'elapsed_sec': elapsed, 'error': str(e)}


def run_scheduled_tasks(verbose: bool = True) -> List[dict]:
    """主入口：檢查所有任務並依需要執行。"""
    if os.environ.get('STRATEGY_SKIP_SCHEDULER') == '1':
        if verbose:
            print("\n[排程器] STRATEGY_SKIP_SCHEDULER=1，略過所有任務")
        return []

    force_env = os.environ.get('STRATEGY_FORCE_SCHEDULER', '').strip()
    forced_set = set()
    if force_env:
        if force_env.lower() == 'all':
            forced_set = set(TASKS.keys())
        else:
            forced_set = {x.strip() for x in force_env.split(',') if x.strip()}

    dry_run = os.environ.get('STRATEGY_DRY_RUN') == '1'
    state = _load_state()
    now = datetime.now()

    if verbose:
        print("\n╔══════════════════════════════════════════╗")
        print("║       定時任務排程器 Scheduler Check     ║")
        print("╚══════════════════════════════════════════╝")
        print(f"現在時間：{now:%Y-%m-%d %H:%M:%S} ({['週一','週二','週三','週四','週五','週六','週日'][now.weekday()]})")

    results = []
    to_run: List[Task] = []
    for key, task in TASKS.items():
        forced = key in forced_set
        should, reason = _should_run(task, now, state, forced)
        if verbose:
            mark = '►' if should else '·'
            print(f"  {mark} [{key:10s}] {reason}")
        if should:
            to_run.append(task)

    if not to_run:
        if verbose:
            print("\n本次執行無需跑任何排程任務，進入主策略流程。\n")
        return results

    if verbose:
        total_est = sum(t.timeout_sec for t in to_run) // 60
        print(f"\n★ 將執行 {len(to_run)} 個任務（預估上限 {total_est} 分鐘）")
        print("  注：此為超時上限，實際通常短很多。若要略過請設定 STRATEGY_SKIP_SCHEDULER=1")

    for task in to_run:
        res = _run_task(task, dry_run=dry_run)
        res['task'] = task.key
        res['timestamp'] = now.isoformat()
        results.append(res)
        # 僅在成功時更新 last_success
        if res['status'] == 'success':
            state[task.key] = {
                'last_success': datetime.now().isoformat(),
                'last_status': 'success',
                'last_elapsed_sec': res['elapsed_sec'],
            }
        else:
            rec = state.get(task.key, {})
            rec.update({
                'last_attempt': datetime.now().isoformat(),
                'last_status': res['status'],
            })
            state[task.key] = rec
        _save_state(state)

    if verbose:
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"排程任務完成（{sum(1 for r in results if r['status']=='success')}/{len(results)} 成功）")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    return results


# ──────────────────────────────────────────────────────────
# CLI：手動查看 / 觸發
# ──────────────────────────────────────────────────────────
def _cli_status():
    state = _load_state()
    now = datetime.now()
    print(f"現在：{now:%Y-%m-%d %H:%M:%S}")
    print(f"{'任務':<12}{'週期':>6}{'上次執行':>22}{'距下次':>12}  條件")
    print('-' * 80)
    for key, task in TASKS.items():
        last = _last_run(task, state)
        last_s = last.strftime('%Y-%m-%d %H:%M') if last else '(尚無紀錄)'
        if last:
            age = (now - last).total_seconds() / 86400
            remain = task.interval_days - age
            remain_s = f'{remain:+.1f} 天'
        else:
            remain_s = '即將執行'
        cond = '僅週末' if task.condition == _is_weekend else '無'
        print(f"{key:<12}{task.interval_days:>4}天{last_s:>22}{remain_s:>12}  {cond}")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Strategy 定時任務排程器')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('status', help='顯示所有任務狀態')
    p_run = sub.add_parser('run', help='檢查並執行到期任務')
    p_run.add_argument('--force', default='', help='強制執行任務 (ml_train,deep_train 或 all)')
    p_run.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    if args.cmd == 'status' or args.cmd is None:
        _cli_status()
    elif args.cmd == 'run':
        if args.force:
            os.environ['STRATEGY_FORCE_SCHEDULER'] = args.force
        if args.dry_run:
            os.environ['STRATEGY_DRY_RUN'] = '1'
        run_scheduled_tasks()


if __name__ == '__main__':
    main()

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


import os
API_KEY = os.getenv("COINGLASS_API_KEY")
BASE_URL = "https://open-api-v4.coinglass.com"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REQUEST_TIMEOUT = 30
REQUEST_INTERVAL_SEC = 0.35

INTERVALS = ("1m", "5m", "15m", "1h")
DEFAULT_LIMIT = "1000"


@dataclass
class UpdateJob:
    file_path: Path
    endpoint_path: str
    interval: str
    base_params: Dict[str, str]


ENDPOINT_PREFIX_TO_PATH = {
    "futures_price_history_btcusdt_binance": "/api/futures/price/history",
    "futures_rsi_history_btcusdt_binance": "/api/futures/indicators/rsi",
    "futures_ma_history_btcusdt_binance": "/api/futures/indicators/ma",
    "futures_ema_history_btcusdt_binance": "/api/futures/indicators/ema",
    "futures_macd_history_btcusdt_binance": "/api/futures/indicators/macd",
    "futures_boll_history_btcusdt_binance": "/api/futures/indicators/boll",
    "futures_whale_index_history_btcusdt_binance": "/api/futures/whale-index/history",
    "futures_top_long_short_account_ratio_history_btcusdt_binance": "/api/futures/top-long-short-account-ratio/history",
    "futures_taker_buy_sell_volume_history_btcusdt_binance": "/api/futures/aggregated-taker-buy-sell-volume/history",
    "futures_open_interest_history_btcusdt_binance": "/api/futures/open-interest/history",
}


def is_success_code(code: Any) -> bool:
    if isinstance(code, str):
        return code.strip() == "0"
    if isinstance(code, (int, float)):
        return int(code) == 0
    return False


def to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def to_epoch_ms(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    try:
        num = float(text)
        ms = int(num)
        if ms < 10_000_000_000:
            ms *= 1000
        return str(ms)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return str(int(dt.timestamp() * 1000))
        except Exception:
            continue
    return ""


def infer_time_ms(row: Dict[str, str]) -> str:
    for key in ("time_ms", "time", "timestamp", "create_time", "createTime", "date", "time_list"):
        if key in row:
            ms = to_epoch_ms(row.get(key, ""))
            if ms:
                return ms
    return ""


def normalize_data_to_rows(data: Any) -> List[Dict[str, str]]:
    if data is None:
        return []
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict):
                row = {k: to_str(v) for k, v in item.items()}
            else:
                row = {"value": to_str(item)}
            row["time_ms"] = infer_time_ms(row)
            out.append(row)
        return out
    if isinstance(data, dict):
        for nested_key in ("list", "items", "data"):
            nested = data.get(nested_key)
            if isinstance(nested, list):
                return normalize_data_to_rows(nested)
        array_keys = [k for k, v in data.items() if isinstance(v, list)]
        max_len = max((len(data[k]) for k in array_keys), default=0)
        if max_len > 0:
            out = []
            for i in range(max_len):
                row = {}
                for k, v in data.items():
                    row[k] = to_str(v[i]) if isinstance(v, list) and i < len(v) else to_str(v)
                row["time_ms"] = infer_time_ms(row)
                out.append(row)
            return out
        row = {k: to_str(v) for k, v in data.items()}
        row["time_ms"] = infer_time_ms(row)
        return [row]
    return [{"value": to_str(data), "time_ms": ""}]


def request_once(session: requests.Session, job: UpdateJob, params: Dict[str, str]) -> List[Dict[str, str]]:
    headers = {"CG-API-KEY": API_KEY, "accept": "application/json"}
    url = f"{BASE_URL}{job.endpoint_path}"
    resp = session.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    payload = resp.json()
    if not is_success_code(payload.get("code")):
        raise RuntimeError(f"business code={payload.get('code')} msg={payload.get('msg')}")

    rows = normalize_data_to_rows(payload.get("data"))
    for row in rows:
        row.setdefault("time_ms", infer_time_ms(row))
    return rows


def unique_row_key(row: Dict[str, str]) -> str:
    t = row.get("time_ms", "").strip()
    if t:
        return f"time_ms::{t}"
    return "row::" + json.dumps(row, ensure_ascii=False, sort_keys=True)


def parse_job_from_file(file_path: Path) -> Optional[UpdateJob]:
    stem = file_path.stem
    interval = next((iv for iv in INTERVALS if stem.endswith(f"_{iv}")), None)
    if not interval:
        return None
    prefix = stem[: -(len(interval) + 1)]
    endpoint_path = ENDPOINT_PREFIX_TO_PATH.get(prefix)
    if not endpoint_path:
        return None

    params = {
        "exchange": "Binance",
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": DEFAULT_LIMIT,
    }
    return UpdateJob(file_path=file_path, endpoint_path=endpoint_path, interval=interval, base_params=params)


def load_existing_rows(file_path: Path) -> Tuple[List[Dict[str, str]], int]:
    rows: List[Dict[str, str]] = []
    max_ts = 0
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            t = (row.get("time_ms") or "").strip()
            if t.isdigit():
                max_ts = max(max_ts, int(t))
            rows.append(row)
    return rows, max_ts


def save_rows(file_path: Path, rows: List[Dict[str, str]]) -> None:
    columns = {"time_ms"}
    for r in rows:
        columns.update(r.keys())
    ordered = ["time_ms"] + sorted(c for c in columns if c != "time_ms")
    rows_sorted = sorted(rows, key=lambda r: int(r["time_ms"]) if (r.get("time_ms") or "").isdigit() else 0)
    with file_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({c: r.get(c, "") for c in ordered})


def collect_new_rows_until_latest(session: requests.Session, job: UpdateJob, existing_latest_ms: int) -> List[Dict[str, str]]:
    """
    Incremental update logic:
    - start from "now" by first request without end_time
    - page backwards with end_time = first row time
    - stop once page touches existing_latest_ms
    - keep only rows newer than existing_latest_ms
    """
    collected: List[Dict[str, str]] = []
    seen = set()
    current_end_time: Optional[str] = None

    while True:
        params = dict(job.base_params)
        params["limit"] = DEFAULT_LIMIT
        if current_end_time:
            params["end_time"] = current_end_time
        else:
            params.pop("end_time", None)

        try:
            rows = request_once(session, job, params)
        except Exception:
            break

        if not rows:
            break

        touched_existing = False
        for row in rows:
            t = row.get("time_ms", "")
            t_int = int(t) if t.isdigit() else 0
            if t_int <= existing_latest_ms and existing_latest_ms > 0:
                touched_existing = True
                continue
            key = unique_row_key(row)
            if key not in seen:
                seen.add(key)
                collected.append(row)

        next_end_time = infer_time_ms(rows[0]) if rows else ""
        if not next_end_time or next_end_time == current_end_time:
            break
        current_end_time = next_end_time

        if touched_existing:
            break
        time.sleep(REQUEST_INTERVAL_SEC)

    return collected


def build_update_jobs() -> List[UpdateJob]:
    jobs: List[UpdateJob] = []
    if not DATA_DIR.exists():
        return jobs
    for p in sorted(DATA_DIR.glob("*.csv")):
        if p.name.startswith("_"):
            continue
        job = parse_job_from_file(p)
        if job:
            jobs.append(job)
    return jobs


def main() -> None:
    jobs = build_update_jobs()
    if not jobs:
        print("No updatable CSV files found in data/.")
        return

    updated = 0
    skipped = 0
    failed = 0

    with requests.Session() as session:
        for idx, job in enumerate(jobs, start=1):
            print(f"[{idx}/{len(jobs)}] Updating {job.file_path.name}")
            try:
                existing_rows, existing_latest_ms = load_existing_rows(job.file_path)
                if existing_latest_ms <= 0:
                    print("  skip: no valid existing time_ms found")
                    skipped += 1
                    continue

                new_rows = collect_new_rows_until_latest(session, job, existing_latest_ms)
                if not new_rows:
                    print("  up-to-date: no new rows")
                    skipped += 1
                    continue

                merged = {unique_row_key(r): r for r in existing_rows}
                for r in new_rows:
                    merged[unique_row_key(r)] = r

                before = len(existing_rows)
                after = len(merged)
                added = after - before
                save_rows(job.file_path, list(merged.values()))
                print(f"  updated: new_rows={len(new_rows)}, net_added={added}")
                updated += 1
            except Exception as e:
                print(f"  fail: {e}")
                failed += 1
            finally:
                time.sleep(REQUEST_INTERVAL_SEC)

    print("\n============================================================")
    print(f"Done. updated={updated} skipped={skipped} failed={failed} total={len(jobs)}")


if __name__ == "__main__":
    main()

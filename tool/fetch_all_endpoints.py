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

INTERVALS = ["1m", "5m", "15m", "1h"]
INTERVAL_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}


@dataclass
class EndpointJob:
    slug: str
    path: str
    interval: str
    base_params: Dict[str, str]


def build_jobs() -> List[EndpointJob]:
    jobs: List[EndpointJob] = []

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_price_history_btcusdt_binance_{interval}",
                path="/api/futures/price/history",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "1000",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_rsi_history_btcusdt_binance_{interval}",
                path="/api/futures/indicators/rsi",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "4500",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_ma_history_btcusdt_binance_{interval}",
                path="/api/futures/indicators/ma",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "4500",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_ema_history_btcusdt_binance_{interval}",
                path="/api/futures/indicators/ema",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "4500",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_macd_history_btcusdt_binance_{interval}",
                path="/api/futures/indicators/macd",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "4500",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_boll_history_btcusdt_binance_{interval}",
                path="/api/futures/indicators/boll",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "4500",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_whale_index_history_btcusdt_binance_{interval}",
                path="/api/futures/whale-index/history",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "1000",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_top_long_short_account_ratio_history_btcusdt_binance_{interval}",
                path="/api/futures/top-long-short-account-ratio/history",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "1000",
                },
            )
        )

    for interval in INTERVALS:
        jobs.append(
            EndpointJob(
                slug=f"futures_taker_buy_sell_volume_history_btcusdt_binance_{interval}",
                path="/api/futures/aggregated-taker-buy-sell-volume/history",
                interval=interval,
                base_params={
                    "exchange": "Binance",
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": "1000",
                },
            )
        )

    return jobs


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
        rows: List[Dict[str, str]] = []
        for item in data:
            if isinstance(item, dict):
                row = {k: to_str(v) for k, v in item.items()}
            else:
                row = {"value": to_str(item)}
            row["time_ms"] = infer_time_ms(row)
            rows.append(row)
        return rows

    if isinstance(data, dict):
        for nested_key in ("list", "items", "data"):
            nested = data.get(nested_key)
            if isinstance(nested, list):
                return normalize_data_to_rows(nested)

        array_keys = [k for k, v in data.items() if isinstance(v, list)]
        max_len = max((len(data[k]) for k in array_keys), default=0)
        if max_len > 0:
            rows: List[Dict[str, str]] = []
            for i in range(max_len):
                row: Dict[str, str] = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        row[k] = to_str(v[i]) if i < len(v) else ""
                    else:
                        row[k] = to_str(v)
                row["time_ms"] = infer_time_ms(row)
                rows.append(row)
            return rows

        row = {k: to_str(v) for k, v in data.items()}
        row["time_ms"] = infer_time_ms(row)
        return [row]

    return [{"value": to_str(data), "time_ms": ""}]


def request_once(session: requests.Session, job: EndpointJob, params: Dict[str, str]) -> Tuple[List[Dict[str, str]], str]:
    headers = {"CG-API-KEY": API_KEY, "accept": "application/json"}
    url = f"{BASE_URL}{job.path}"
    resp = session.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    payload = resp.json()
    if not is_success_code(payload.get("code")):
        raise RuntimeError(f"business code={payload.get('code')} msg={payload.get('msg')}")

    rows = normalize_data_to_rows(payload.get("data"))
    if not rows:
        rows = [{"time_ms": ""}]
    for row in rows:
        row.setdefault("time_ms", infer_time_ms(row))
    return rows, resp.url


def unique_row_key(row: Dict[str, str]) -> str:
    t = row.get("time_ms", "").strip()
    if t:
        return f"time_ms::{t}"
    return "row::" + json.dumps(row, ensure_ascii=False, sort_keys=True)


def collect_history_with_end_time_paging(session: requests.Session, job: EndpointJob) -> Tuple[List[Dict[str, str]], str]:
    collected: List[Dict[str, str]] = []
    seen = set()
    last_url = ""
    current_end_time: Optional[str] = None

    while True:
        params = dict(job.base_params)
        params["limit"] = "1000"
        if current_end_time:
            params["end_time"] = current_end_time
        else:
            params.pop("end_time", None)

        try:
            rows, final_url = request_once(session, job, params)
            last_url = final_url
        except Exception:
            break

        if not rows:
            break

        for row in rows:
            key = unique_row_key(row)
            if key not in seen:
                seen.add(key)
                collected.append(row)

        next_end_time = infer_time_ms(rows[0]) if rows else ""
        if not next_end_time or next_end_time == current_end_time:
            break
        current_end_time = next_end_time
        time.sleep(REQUEST_INTERVAL_SEC)

    if not collected:
        raise RuntimeError("no data collected during paging")
    return collected, last_url


def sort_rows_by_time(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def _time_int(r: Dict[str, str]) -> int:
        t = r.get("time_ms", "").strip()
        return int(t) if t.isdigit() else 0

    return sorted(rows, key=_time_int)


def find_missing_time_points(rows: List[Dict[str, str]], interval: str) -> List[int]:
    if interval not in INTERVAL_TO_MS:
        return []
    step = INTERVAL_TO_MS[interval]

    ts = sorted({int(r["time_ms"]) for r in rows if r.get("time_ms", "").isdigit()})
    if len(ts) < 2:
        return []

    missing: List[int] = []
    for prev, cur in zip(ts, ts[1:]):
        expected = prev + step
        while expected < cur:
            missing.append(expected)
            expected += step
    return missing


def backfill_missing_points(session: requests.Session, job: EndpointJob, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    missing_points = find_missing_time_points(rows, job.interval)
    if not missing_points:
        return rows

    print(f"  gap scan: found {len(missing_points)} missing time points")
    existing_keys = {unique_row_key(r) for r in rows}
    recovered = 0

    for missing_time in missing_points:
        params = dict(job.base_params)
        params["limit"] = "1"
        params["end_time"] = str(missing_time)
        try:
            one_rows, _ = request_once(session, job, params)
        except Exception:
            time.sleep(REQUEST_INTERVAL_SEC)
            continue

        for row in one_rows:
            if row.get("time_ms", "") == str(missing_time):
                key = unique_row_key(row)
                if key not in existing_keys:
                    existing_keys.add(key)
                    rows.append(row)
                    recovered += 1
        time.sleep(REQUEST_INTERVAL_SEC)

    print(f"  gap backfill: recovered {recovered} rows")
    return rows


def write_rows_to_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        rows = [{"time_ms": ""}]

    columns = {"time_ms"}
    for row in rows:
        columns.update(row.keys())
    ordered_columns = ["time_ms"] + sorted(c for c in columns if c != "time_ms")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in ordered_columns})


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()

    success = 0
    failed = 0

    with requests.Session() as session:
        for idx, job in enumerate(jobs, start=1):
            print(f"[{idx}/{len(jobs)}] Collecting {job.slug} ({job.interval}) -> {job.path}")
            try:
                rows, final_url = collect_history_with_end_time_paging(session, job)
                print(f"  paging collected rows={len(rows)} last_url={final_url}")

                rows = backfill_missing_points(session, job, rows)

                dedup_map = {unique_row_key(r): r for r in rows}
                final_rows = sort_rows_by_time(list(dedup_map.values()))
                out_file = DATA_DIR / f"{job.slug}.csv"
                write_rows_to_csv(out_file, final_rows)
                print(f"  saved rows={len(final_rows)} file={out_file}")
                success += 1
            except Exception as e:
                print(f"  FAIL {e}")
                failed += 1
            finally:
                time.sleep(REQUEST_INTERVAL_SEC)

    print("\n============================================================")
    print(f"Done. success={success} failed={failed} total={len(jobs)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


FREQS: Tuple[str, ...] = ("1m", "5m", "15m", "1h")
INTERVAL_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}
EXCLUDED_PREFIXES: Tuple[str, ...] = (
    "futures_top_long_short_account_ratio_history_btcusdt_binance",
    "futures_whale_index_history_btcusdt_binance",
    "feat_ret",
    "feat_vol",
)
EXCLUDED_EXACT_COLUMNS: Tuple[str, ...] = ("feat_missing_ratio",)


def _resolve_root(root_dir: Optional[Path]) -> Path:
    if root_dir is not None:
        return Path(root_dir).resolve()
    return Path(__file__).resolve().parents[1]


def load_cleaned_minute_whale_csv(
    path: Optional[Path | str] = None,
    *,
    root_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load `cleaned_1m_with_whale.csv` (1m OHLCV + whale fields).

    Uses `price_time_ms` if present, else `time_ms`, as the bar key; index is named `time_ms`.
    Parses `time_utc`; coerces OHLCV to numeric.

    Typical columns include: time_utc, close, high, low, open, volume, whale_alert_count,
    whale_pos_notional_abs, whale_pos_lev_wavg, … (see Data_Pipeline output).
    """
    root = _resolve_root(root_dir)
    p = Path(path) if path is not None else root / "data" / "cleaned" / "cleaned_1m_with_whale.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, low_memory=False)
    time_col = "price_time_ms" if "price_time_ms" in df.columns else "time_ms"
    if time_col not in df.columns:
        raise ValueError(f"Expected price_time_ms or time_ms in {p}")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).drop_duplicates(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)
    df.index = df.index.astype("int64")
    df.index.name = "time_ms"

    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    else:
        df["time_utc"] = pd.to_datetime(df.index, unit="ms", utc=True)

    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _load_aligned_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "time_ms" not in df.columns:
        return None

    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    df = df.dropna(subset=["time_ms"]).copy()
    if df.empty:
        return None

    df["time_ms"] = df["time_ms"].astype("int64")
    df = df.drop_duplicates(subset=["time_ms"]).sort_values("time_ms")
    df = df.set_index("time_ms")
    df.index.name = "time_ms"

    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
    else:
        df["time_utc"] = pd.to_datetime(df.index, unit="ms", utc=True)

    return df


def _is_excluded_feature(
    col: str,
    prefixes: Iterable[str],
    exact_columns: Iterable[str],
) -> bool:
    if col in set(exact_columns):
        return True
    return any(col.startswith(prefix) for prefix in prefixes)


def _latest_continuous_segment(
    table: pd.DataFrame,
    freq: str,
    *,
    blackout_gap_ms: int = 86_400_000,
) -> pd.DataFrame:
    if table.empty:
        return table

    expected_step = INTERVAL_TO_MS.get(freq)
    if expected_step is None or len(table.index) <= 1:
        return table

    idx = pd.Index(table.index.astype("int64"))
    diffs = idx.to_series().diff().fillna(expected_step)
    # Treat only long outages as blackout boundaries.
    gap_threshold = max(expected_step, blackout_gap_ms)
    breaks = diffs > gap_threshold
    segment_id = breaks.cumsum()
    latest_id = int(segment_id.iloc[-1])
    return table.loc[segment_id == latest_id].copy()


def clean_one_table(
    aligned_table: pd.DataFrame,
    freq: str,
    excluded_prefixes: Tuple[str, ...] = EXCLUDED_PREFIXES,
    excluded_exact_columns: Tuple[str, ...] = EXCLUDED_EXACT_COLUMNS,
    blackout_gap_ms: int = 86_400_000,
) -> Tuple[Optional[pd.DataFrame], str]:
    t = aligned_table.copy()
    if t.empty:
        return None, "skipped: empty aligned table"

    # Drop excluded feature families from final cleaned table.
    excluded_cols = [
        c
        for c in t.columns
        if _is_excluded_feature(c, excluded_prefixes, excluded_exact_columns)
    ]
    if excluded_cols:
        t = t.drop(columns=excluded_cols)

    feature_cols = [c for c in t.columns if c != "time_utc"]
    if not feature_cols:
        return None, "skipped: no feature columns after exclusions"

    # Remove timestamps where all retained features are missing first.
    active = t[t[feature_cols].notna().any(axis=1)].copy()
    if active.empty:
        return None, "skipped: no active rows after exclusions"

    active = _latest_continuous_segment(active, freq, blackout_gap_ms=blackout_gap_ms)
    if active.empty:
        return None, "skipped: empty recent continuous segment"

    t = active

    full_mask = t[feature_cols].notna().all(axis=1)
    if not full_mask.any():
        return None, "skipped: no fully-covered timestamp in recent continuous segment"

    first_full_idx = int(full_mask[full_mask].index[0])
    sub = t.loc[t.index >= first_full_idx].copy()
    sub_clean = sub.dropna(subset=feature_cols).sort_index()
    if sub_clean.empty:
        return None, "skipped: empty after dropna"

    start_utc = sub_clean["time_utc"].iloc[0]
    end_utc = sub_clean["time_utc"].iloc[-1]
    msg = f"cleaned shape={sub_clean.shape}, start={start_utc}, end={end_utc}, excluded_cols={len(excluded_cols)}"
    return sub_clean, msg


def build_cleaned_tables(
    root_dir: Optional[Path] = None,
    freqs: Tuple[str, ...] = FREQS,
    blackout_gap_ms: int = 86_400_000,
) -> Dict[str, pd.DataFrame]:
    root = _resolve_root(root_dir)
    aligned_dir = root / "data" / "aligned"
    cleaned_dir = root / "data" / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    cleaned_tables: Dict[str, pd.DataFrame] = {}
    for freq in freqs:
        aligned_path = aligned_dir / f"aligned_{freq}.csv"
        table = _load_aligned_table(aligned_path)
        if table is None:
            print(f"[{freq}] skipped: aligned table not found or invalid ({aligned_path})")
            continue

        cleaned, status = clean_one_table(
            table,
            freq=freq,
            excluded_prefixes=EXCLUDED_PREFIXES,
            excluded_exact_columns=EXCLUDED_EXACT_COLUMNS,
            blackout_gap_ms=blackout_gap_ms,
        )
        if cleaned is None:
            print(f"[{freq}] {status}")
            continue

        out_path = cleaned_dir / f"cleaned_{freq}.csv"
        cleaned.to_csv(out_path)
        cleaned_tables[freq] = cleaned
        print(f"[{freq}] {status}")
        print(f"[{freq}] saved: {out_path}")

    print(f"\nDone. Cleaned tables are in: {cleaned_dir}")
    return cleaned_tables


def main() -> None:
    build_cleaned_tables()


if __name__ == "__main__":
    main()

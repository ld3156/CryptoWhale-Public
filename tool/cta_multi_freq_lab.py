"""
Multi-frequency CTA helpers: resample 1m OHLCV → lower bars, unified signal library (core + research extras).
Used by `tool/run_cta_signal_sweep.py` and notebooks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tool.core_cta_baseline import (
    build_core_feature_formulas,
    load_cleaned_1m_baseline,
)
from tool.newmath import apply_formulas, numeric_to_float32
from tool.technical_indicators import add_technical_indicators


def freq_str_for_bar_minutes(bar_minutes: int) -> str:
    if bar_minutes <= 1:
        return "1m"
    if bar_minutes == 5:
        return "5m"
    if bar_minutes == 15:
        return "15m"
    if bar_minutes == 30:
        return "30m"
    if bar_minutes == 60:
        return "1h"
    raise ValueError(f"Unsupported bar_minutes={bar_minutes}")


def z_window_scaled_from_1m_bars(z_window_1m: int, bar_minutes: int) -> int:
    """Match ~same calendar span as z_window_1m 1m bars (e.g. 480 → 32 at 15m)."""
    if bar_minutes <= 1:
        return max(16, int(z_window_1m))
    w = int(round(z_window_1m / float(bar_minutes)))
    return max(16, w)


def resample_ohlcv_1m(df_1m: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """
    Aggregate 1m OHLCV (time_ms index) to bar_minutes OHLCV. Right-labeled bar end timestamp.
    """
    if bar_minutes <= 1:
        return df_1m.copy()

    d = df_1m.sort_index()
    dt_ix = pd.to_datetime(d.index, unit="ms", utc=True)
    d = d.copy()
    d.index = dt_ix
    rule = f"{bar_minutes}min"
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [c for c in agg_map if c in d.columns]
    if not cols:
        raise ValueError("Need OHLCV columns for resample")
    out = d[cols].resample(rule, label="right", closed="right").agg({c: agg_map[c] for c in cols})
    out = out.dropna(subset=["close"])
    out["time_utc"] = out.index
    t = pd.DatetimeIndex(out.index)
    ms = (t.astype(np.int64) // 10**6).astype(np.int64)
    out.index = ms
    out.index.name = "time_ms"
    return out


def prepare_cta_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Technical indicators + core baseline formula features (same as Feature_Engineering)."""
    out = add_technical_indicators(df)
    formulas = build_core_feature_formulas()
    out = apply_formulas(out, formulas)
    return numeric_to_float32(out, exclude=("time_utc",))


def build_unified_signal_library() -> Dict[str, Dict[str, str]]:
    """
    Core `build_core_signal_library` plus research-style signals that worked well on 15m:
    momentum beta vs longer horizon, tail risk (kurtosis), skew reversal.
    """
    from tool.core_cta_baseline import build_core_signal_library

    lib = build_core_signal_library()

    # mom_beta / tail_risk / skew_reversal: also in `build_core_signal_library` (keep unified = core + empty for now)
    return lib


def add_inverted_signal_columns(
    df: pd.DataFrame,
    signal_cols: list[str],
) -> Tuple[pd.DataFrame, list[str]]:
    """Append negated columns `name__inv` for direction search."""
    inv = {f"{s}__inv": -df[s] for s in signal_cols}
    out = pd.concat([df, pd.DataFrame(inv, index=df.index)], axis=1)
    extra = list(inv.keys())
    return out, signal_cols + extra


def load_1m_baseline(root: Path | None = None) -> pd.DataFrame:
    return load_cleaned_1m_baseline(root)

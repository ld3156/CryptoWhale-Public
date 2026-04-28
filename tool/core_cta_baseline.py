"""
Core CTA baseline: OHLCV → technical indicators → formula features (no whale columns).

Typical pipeline (see Feature_Engineering):
`load_cleaned_1m_baseline` → `add_technical_indicators` (RSI/MA/EMA/MACD/BOLL/TR/ATR) →
`build_core_feature_formulas` → `build_core_signal_library`.

Train/test split: metrics for selection use bars strictly before TRAIN_END_EXCLUSIVE_UTC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# UTC: training uses [first_bar, TRAIN_END_EXCLUSIVE)
TRAIN_END_EXCLUSIVE_UTC = pd.Timestamp("2026-03-04", tz="UTC")


def baseline_1m_csv(root: Path | None = None) -> Path:
    r = root or Path(__file__).resolve().parents[1]
    return r / "data" / "cleaned" / "cleaned_1m.csv"


def load_cleaned_1m_baseline(root: Path | None = None) -> pd.DataFrame:
    """
    Load minute OHLCV from cleaned_1m.csv. Index: time_ms (int64). No whale truncation.
    """
    p = baseline_1m_csv(root)
    df = pd.read_csv(p, low_memory=False)
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    df = df.dropna(subset=["time_ms"]).drop_duplicates(subset=["time_ms"]).sort_values("time_ms")
    df = df.set_index("time_ms")
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


def train_test_masks_from_time_utc(
    df: pd.DataFrame,
    *,
    train_end_exclusive: pd.Timestamp | None = None,
    time_col: str = "time_utc",
) -> tuple[pd.Series, pd.Series]:
    """
    Boolean masks aligned to df.index. Train: time < train_end_exclusive. Test: >=.
    """
    end = train_end_exclusive if train_end_exclusive is not None else TRAIN_END_EXCLUSIVE_UTC
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    train_mask = t < end
    test_mask = t >= end
    return train_mask, test_mask


def build_core_feature_formulas() -> Dict[str, str]:
    """
    Extra OHLCV-derived features (minute bar units). Requires add_technical_indicators + ret_1.
    """
    return {
        "ret_1": "ts_returns(close, 1)",
        "ret_5": "ts_returns(close, 5)",
        "ret_15": "ts_returns(close, 15)",
        "ret_30": "ts_returns(close, 30)",
        "ret_60": "ts_returns(close, 60)",
        "ret_120": "ts_returns(close, 120)",
        "ret_240": "ts_returns(close, 240)",
        "ret_720": "ts_returns(close, 720)",
        "ret_1440": "ts_returns(close, 1440)",
        "ret_2880": "ts_returns(close, 2880)",
        "ret_120ex15": "ret_120 - ret_15",
        "range_pct": "safe_div(high - low, close)",
        "clv": "safe_div(close - low, high - low + 1e-12)",
        "vol_log": "log(volume + 1.0)",
        "vol_z_60": "ts_zscore(volume, 60)",
        "vol_z_240": "ts_zscore(volume, 240)",
        "vol_chg": "ts_returns(volume, 1)",
        "ret_vol_60_1440": "safe_div(ts_std(ret_1, 60), ts_std(ret_1, 1440) + 1e-12)",
        "rsi_z": "ts_zscore(rsi, 60)",
        "rsi_chg": "rsi - last(rsi)",
        "candle_box": "safe_div((close - open), last(close))",
        "ma_above": "ifcond(close > ma, 1, -1)",
        "ema_above": "ifcond(close > ema, 1, -1)",
        "ma_slope_30": "ma - ts_delay(ma, 30)",
        "ma_curv_15": "ma - 2*ts_delay(ma, 15) + ts_delay(ma, 30)",
        "boll_pos": "safe_div(close - boll_lb, boll_ub - boll_lb)",
        "band_width": "safe_div((boll_ub - boll_lb), boll_mb)",
        "tr": "safe_div(ts_max(high, 2) - ts_min(low, 2), close)",
        "atr_20": "ts_mean(tr, 20)",
    }


def build_core_signal_library() -> Dict[str, Dict[str, str]]:
    """
    Expanded CTA signal library (technical + volume/price only).
    Naming: groups mirror cta_signal_research.py for reuse of tooling.
    """
    lib: Dict[str, Dict[str, str]] = {
        "trend": {},
        "momentum": {},
        "mean_revert": {},
        "volatility": {},
        "breakout": {},
        "volume": {},
        "adaptive": {},
    }

    for fast, slow in [(8, 48), (16, 96), (30, 180), (60, 360), (120, 720), (240, 1440)]:
        lib["trend"][f"ema_spread_{fast}_{slow}"] = (
            f"safe_div(ts_ema(close, {fast}) - ts_ema(close, {slow}), ts_std(close, {slow}))"
        )
        lib["trend"][f"ma_spread_{fast}_{slow}"] = (
            f"safe_div(ts_mean(close, {fast}) - ts_mean(close, {slow}), ts_std(close, {slow}))"
        )

    for slow in [120, 240, 360, 720, 1440]:
        lib["trend"][f"price_ma_dist_{slow}"] = f"safe_div(close - ts_mean(close, {slow}), ts_std(close, {slow}))"

    for short, long in [(5, 60), (15, 120), (30, 240), (60, 360), (120, 720), (240, 1440), (720, 2880)]:
        lib["momentum"][f"mom_diff_{short}_{long}"] = f"ts_returns(close, {short}) - ts_returns(close, {long})"
        lib["momentum"][f"mom_ratio_{short}_{long}"] = (
            f"safe_div(ts_returns(close, {short}), ts_std(ret_1, {long}) + 1e-12)"
        )

    for long in [60, 120, 240, 720, 1440]:
        lib["momentum"][f"mom_sign_{long}"] = f"ifcond(ts_returns(close, {long}) > 0, 1, -1)"

    for w in [48, 96, 168, 240, 480]:
        lib["momentum"][f"macd_hist_z_{w}"] = f"ts_zscore(macd_hist, {w})"
        lib["momentum"][f"rsi_trend_z_{w}"] = f"ts_zscore(rsi - 50, {w})"

    # Short-horizon reversal (minute-level documented effect; test-only validation)
    for w in [5, 15, 30, 60]:
        lib["mean_revert"][f"rev_ret_{w}"] = f"-ts_zscore(ts_returns(close, 1), {w})"

    for w in [48, 96, 168, 240]:
        lib["mean_revert"][f"rsi_revert_{w}"] = f"-ts_zscore(rsi, {w})"
        lib["mean_revert"][f"boll_revert_{w}"] = f"-ts_zscore(boll_pos - 0.5, {w})"
        lib["mean_revert"][f"clv_revert_{w}"] = f"-ts_zscore(clv - 0.5, {w})"

    for w in [24, 48, 96, 168, 240]:
        lib["volatility"][f"vol_break_{w}"] = f"ts_zscore(ts_std(ret_1, {w}), {w * 3})"
        lib["volatility"][f"vol_compress_{w}"] = f"-ts_zscore(band_width, {w * 3})"
        lib["volatility"][f"rv_ratio_z_{w}"] = f"ts_zscore(ret_vol_60_1440, {w * 4})"

    for w in [20, 40, 60, 120, 240, 480, 960, 1440]:
        lib["breakout"][f"donchian_up_{w}"] = f"safe_div(close - ts_max(high, {w}), ts_mean(atr_20, {w}) + 1e-12)"
        lib["breakout"][f"donchian_dn_{w}"] = f"safe_div(close - ts_min(low, {w}), ts_mean(atr_20, {w}) + 1e-12)"
        lib["breakout"][f"quantile_hi_{w}"] = f"safe_div(close - ts_quantile(close, {w}, 0.8), ts_std(close, {w}) + 1e-12)"

    for w in [30, 60, 120, 240]:
        lib["volume"][f"vol_pressure_{w}"] = f"ts_zscore(ts_corr(ret_1, vol_log, {w}), {w})"
        lib["volume"][f"vol_z_mom_{w}"] = f"ts_zscore(vol_z_60, {w})"

    lib["volume"]["vol_spike_60"] = "vol_z_60"
    lib["volume"]["range_vol_interact"] = "ts_zscore(range_pct * vol_z_60, 120)"

    lib["adaptive"]["trend_low_vol_gate"] = (
        "ifcond(ts_std(ret_1, 96) < ts_std(ret_1, 480), "
        "safe_div(ts_ema(close, 120) - ts_ema(close, 720), ts_std(close, 720)), 0)"
    )
    lib["adaptive"]["mom_vol_target"] = "safe_div(ts_returns(close, 360), ts_std(ret_1, 240) + 1e-12)"

    # Rolling beta of short-horizon vs long-horizon returns (15m research favorites; bar units = current TF)
    for short, long in [(15, 120), (30, 240), (60, 360), (120, 720), (240, 1440)]:
        lib["momentum"][f"mom_beta_{short}_{long}"] = (
            f"ts_beta(ts_returns(close, {short}), ts_returns(close, {long}), {long})"
        )

    for w in [96, 168, 240, 360]:
        lib["volatility"][f"tail_risk_{w}"] = f"ts_kurt(ret_1, {w})"
        lib["volatility"][f"skew_reversal_{w}"] = f"-ts_skew(ret_1, {w})"

    return lib

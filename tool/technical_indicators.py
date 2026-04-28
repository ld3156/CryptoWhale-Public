from __future__ import annotations

"""
Classic OHLCV technical indicators — vectorized (parallel-friendly) batch computation.

All series align on the input DataFrame index; intended for 1m (or any) bar tables
where RSI / EMA / MA / MACD / Bollinger are not pre-supplied.
"""

import numpy as np
import pandas as pd

from tool.newmath import numeric_to_float32


def _round_series(s: pd.Series, ndigits: int) -> pd.Series:
    """Round floats; NaN unchanged. Cuts spurious 1e-15-style noise from chained ops."""
    if ndigits < 0:
        return s
    return s.round(ndigits)


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder-style smoothing (via EWM alpha=1/period)."""
    c = pd.to_numeric(close, errors="coerce")
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def sma(series: pd.Series, period: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .ewm(span=span, adjust=False, min_periods=span)
        .mean()
    )


def macd_components(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    c = pd.to_numeric(close, errors="coerce")
    ema_fast = ema(c, fast)
    ema_slow = ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    c = pd.to_numeric(close, errors="coerce")
    mid = sma(c, period)
    sd = c.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return upper, mid, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def add_technical_indicators(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    rsi_period: int = 14,
    ma_period: int = 20,
    ema_period: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    boll_period: int = 20,
    boll_std: float = 2.0,
    prefix: str = "",
    round_indicators: bool = True,
    rsi_decimals: int = 4,
    price_scale_decimals: int = 6,
    macd_decimals: int = 6,
    use_float32: bool = True,
) -> pd.DataFrame:
    """
    Append columns: rsi, ma, ema, macd, macd_signal, macd_hist, boll_ub, boll_mb, boll_lb, tr, atr_20.

    Column names match the short aliases used in Feature_Engineering / CTA notebooks.
    Optional `prefix` prepends to each new column name (e.g. prefix=\"btc_\").

    By default, numeric columns are rounded to reduce float noise when persisting or chaining formulas.
    Set `round_indicators=False` to skip. Tune `*_decimals` if you need more precision.

    When `use_float32` is True (default), the full frame is passed through `numeric_to_float32`
    so OHLCV, whale, and indicator columns share float32 storage.
    """
    out = df.copy()
    close = out[close_col]

    rsi_col = f"{prefix}rsi" if prefix else "rsi"
    ma_col = f"{prefix}ma" if prefix else "ma"
    ema_col = f"{prefix}ema" if prefix else "ema"
    macd_col = f"{prefix}macd" if prefix else "macd"
    sig_col = f"{prefix}macd_signal" if prefix else "macd_signal"
    hist_col = f"{prefix}macd_hist" if prefix else "macd_hist"
    ub_col = f"{prefix}boll_ub" if prefix else "boll_ub"
    mb_col = f"{prefix}boll_mb" if prefix else "boll_mb"
    lb_col = f"{prefix}boll_lb" if prefix else "boll_lb"

    m_line, s_line, hist = macd_components(close, macd_fast, macd_slow, macd_signal_period)
    ub, mb, lb = bollinger_bands(close, boll_period, boll_std)

    ti: dict[str, pd.Series] = {
        rsi_col: rsi_wilder(close, rsi_period),
        ma_col: sma(close, ma_period),
        ema_col: ema(close, ema_period),
        macd_col: m_line,
        sig_col: s_line,
        hist_col: hist,
        ub_col: ub,
        mb_col: mb,
        lb_col: lb,
    }

    tr_name = f"{prefix}tr" if prefix else "tr"
    atr_name = f"{prefix}atr_20" if prefix else "atr_20"
    if high_col in out.columns and low_col in out.columns:
        tr = true_range(out[high_col], out[low_col], close)
        ti[tr_name] = tr
        ti[atr_name] = tr.rolling(20, min_periods=20).mean()

    ti_df = pd.DataFrame(ti, index=out.index)
    overlap = [c for c in ti_df.columns if c in out.columns]
    if overlap:
        out = out.drop(columns=overlap, errors="ignore")
    out = pd.concat([out, ti_df], axis=1)
    out = out.copy()

    if round_indicators:
        out[rsi_col] = _round_series(out[rsi_col], rsi_decimals)
        for col in (ma_col, ema_col, ub_col, mb_col, lb_col):
            out[col] = _round_series(out[col], price_scale_decimals)
        for col in (macd_col, sig_col, hist_col):
            out[col] = _round_series(out[col], macd_decimals)
        if tr_name in out.columns and atr_name in out.columns:
            out[tr_name] = _round_series(out[tr_name], price_scale_decimals)
            out[atr_name] = _round_series(out[atr_name], price_scale_decimals)

    if use_float32:
        out = numeric_to_float32(out)

    return out

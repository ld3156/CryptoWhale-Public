"""
Ensemble allocation helpers: volatility targeting and inverse-volatility (risk-parity-style) mixing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tool.cta_signal_lab import BacktestConfig


def bars_per_year(cfg: BacktestConfig) -> float:
    return float(cfg.annual_bars.get(cfg.freq, 525_600))


def rolling_ann_vol(
    per_bar_pnl: pd.Series,
    *,
    window: int,
    bars_per_year: float,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling annualized volatility of per-bar PnL (e.g. pos * ret)."""
    mp = min_periods if min_periods is not None else max(int(window) // 4, 10)
    s = pd.to_numeric(per_bar_pnl, errors="coerce").fillna(0.0)
    sig = s.rolling(int(window), min_periods=int(mp)).std(ddof=0)
    return sig * np.sqrt(float(bars_per_year))


def vol_target_position(
    pos: pd.Series,
    ret_1: pd.Series,
    *,
    target_ann_vol: float,
    window: int,
    bars_per_year: float,
    max_leverage: float = 2.5,
    min_ann_vol: float = 0.03,
    clip_abs: float = 1.0,
) -> pd.Series:
    """
    Scale ``pos`` so rolling ann. vol of gross PnL ``pos * ret_1`` tracks ``target_ann_vol``.

    Uses leverage ``target_ann_vol / realized_vol`` (floored at ``min_ann_vol``, capped at ``max_leverage``),
    then clips position to ``[-clip_abs, clip_abs]`` (execution / margin cap).
    """
    pos = pd.to_numeric(pos, errors="coerce").fillna(0.0).reindex(ret_1.index).fillna(0.0)
    r = pd.to_numeric(ret_1, errors="coerce").fillna(0.0)
    pnl = pos * r
    vol = rolling_ann_vol(pnl, window=int(window), bars_per_year=bars_per_year)
    lev = float(target_ann_vol) / vol.clip(lower=float(min_ann_vol))
    lev = lev.clip(upper=float(max_leverage))
    lev = lev.ffill().bfill().fillna(1.0)
    out = pos * lev
    return out.clip(-float(clip_abs), float(clip_abs))


def inverse_vol_weights(
    pos_mat: pd.DataFrame,
    ret_1: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Row-wise inverse-volatility weights from rolling std of per-signal gross returns ``pos_i * ret``.

    Weights sum to 1 per row (convex combination of strategies). Naive risk-parity heuristic when
    correlations are ignored.
    """
    r = pd.to_numeric(ret_1, errors="coerce").fillna(0.0)
    gross = pos_mat.mul(r, axis=0)
    mp = min_periods if min_periods is not None else max(int(window) // 4, 10)
    vol = gross.rolling(int(window), min_periods=int(mp)).std(ddof=0)
    inv = 1.0 / (vol + float(eps))
    w = inv.div(inv.sum(axis=1), axis=0)
    n = pos_mat.shape[1]
    eq = 1.0 / float(n) if n else 1.0
    return w.fillna(eq)


def risk_parity_position(
    pos_mat: pd.DataFrame,
    ret_1: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    clip_abs: float = 1.0,
) -> pd.Series:
    """Combine columns with inverse-vol weights; clip combined position."""
    w = inverse_vol_weights(pos_mat, ret_1, window=window, min_periods=min_periods)
    comb = (pos_mat * w).sum(axis=1)
    return comb.clip(-float(clip_abs), float(clip_abs))

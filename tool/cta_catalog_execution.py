"""
Execute CTA catalog signals (possibly learned on 5m/15m/30m/1h) while holding a 1m OHLCV table.

Pipeline per catalog row:
  1m OHLCV → resample to `bar_minutes` → features + single signal → `signal_to_position`
  → causal shift → **forward-fill** position onto every 1m timestamp.

So you always trade on the 1m timeline; intra-bar the position is piecewise-constant between
lower-frequency bar closes (standard multi-timeframe execution).
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from tool.core_cta_baseline import build_core_signal_library
from tool.cta_multi_freq_lab import (
    freq_str_for_bar_minutes,
    prepare_cta_feature_frame,
    resample_ohlcv_1m,
)
from tool.cta_signal_lab import BacktestConfig, flatten_signal_library, signal_to_position
from tool.newmath import apply_formulas


def split_signal_inv(sig_col: str) -> Tuple[str, bool]:
    """Catalog names may end with ``__inv`` (negated raw signal)."""
    if sig_col.endswith("__inv"):
        return sig_col[: -len("__inv")], True
    return sig_col, False


def upsample_position_to_1m(pos_low: pd.Series, index_1m: pd.Index) -> pd.Series:
    """
    Map low-frequency positions to 1m index: sort-merge union, ffill, reindex to 1m.
    """
    pos_low = pos_low.astype("float64")
    if pos_low.empty:
        return pd.Series(0.0, index=index_1m)
    union = pos_low.index.union(index_1m)
    s = pos_low.reindex(union).sort_index().ffill()
    out = s.reindex(index_1m).ffill().fillna(0.0)
    return out


def _formula_for_base_column(base_sig: str, flat: Dict[str, str]) -> str:
    if base_sig not in flat:
        raise KeyError(
            f"Signal {base_sig!r} not in library. Re-run sweep after `build_core_signal_library` includes it."
        )
    return flat[base_sig]


def position_from_catalog_row(
    df_1m: pd.DataFrame,
    row: pd.Series,
    *,
    signal_library: Optional[Dict[str, Dict[str, str]]] = None,
    cfg: Optional[BacktestConfig] = None,
) -> pd.Series:
    """
    One catalog row → position series aligned to ``df_1m.index`` (1m bars).

    Expects ``row`` to have at least: signal, bar_minutes, z_window, deadband, smooth_span, position_mode.
    Uses ``freq`` from row if present, else derived from bar_minutes.
    """
    lib = signal_library if signal_library is not None else build_core_signal_library()
    flat = flatten_signal_library(lib)

    sig_col = str(row["signal"])
    base_col, inv = split_signal_inv(sig_col)
    formula = {base_col: _formula_for_base_column(base_col, flat)}

    bm = int(row["bar_minutes"])
    if bm <= 1:
        dfr = df_1m.copy()
    else:
        dfr = resample_ohlcv_1m(df_1m, bm)

    dfr = prepare_cta_feature_frame(dfr)
    dfr = apply_formulas(dfr, formula)
    raw = dfr[base_col].astype("float64")
    if inv:
        raw = -raw

    z_window = int(row["z_window"])
    deadband = float(row["deadband"])
    smooth_span = int(row["smooth_span"])
    mode = str(row["position_mode"])

    freq = str(row["freq"]) if "freq" in row and pd.notna(row["freq"]) else freq_str_for_bar_minutes(bm)
    if cfg is None:
        cfg = BacktestConfig(freq=freq)

    pos = signal_to_position(
        raw,
        clip=cfg.signal_clip,
        z_window=z_window,
        mode=mode,
        deadband=deadband,
        smooth_span=smooth_span,
        max_abs_position=cfg.max_abs_position,
        position_step=cfg.position_step,
    )
    pos = pos.shift(int(cfg.signal_shift)).fillna(0.0)

    return upsample_position_to_1m(pos, df_1m.index)


def positions_table_from_catalog(
    df_1m: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    top_n: Optional[int] = None,
    signal_library: Optional[Dict[str, Dict[str, str]]] = None,
    cfg: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """
    Build a matrix of positions (each column = one catalog row) on 1m index.
    If a row fails (missing formula, etc.), that column is all zeros and a warning is emitted.
    """
    cat = catalog.copy()
    if top_n is not None:
        cat = cat.head(int(top_n))

    cols: Dict[str, pd.Series] = {}
    for _, row in cat.iterrows():
        label = f"{row['signal']}|{row.get('freq', row['bar_minutes'])}"
        label = str(label).replace("/", "_")[:120]
        try:
            cols[label] = position_from_catalog_row(
                df_1m, row, signal_library=signal_library, cfg=cfg
            )
        except Exception as e:
            warnings.warn(f"{label}: {e}", stacklevel=2)
            cols[label] = pd.Series(0.0, index=df_1m.index)
    return pd.DataFrame(cols)


def ensemble_position_equal_weight(pos_df: pd.DataFrame) -> pd.Series:
    """Mean of component positions (ignoring non-numeric / error columns)."""
    num = pos_df.select_dtypes(include=[np.floating, np.integer, bool])
    if num.empty:
        return pd.Series(0.0, index=pos_df.index)
    return num.mean(axis=1).clip(-1.0, 1.0)


def net_return_1m_from_position(pos_1m: pd.Series, ret_1m: pd.Series) -> pd.Series:
    """Gross 1m return; multiply aligned position × next-bar return if you use shift elsewhere — here same index."""
    return pos_1m * ret_1m


def net_return_with_costs(
    pos_1m: pd.Series,
    ret_1m: pd.Series,
    cfg: BacktestConfig,
) -> pd.Series:
    """Fee + slippage on position changes (same convention as ``cta_signal_lab.evaluate_single_signal``)."""
    turn = pos_1m.diff().abs().fillna(0.0)
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    return pos_1m * ret_1m - turn * cost_rate


def equity_curve_from_1m_net(net: pd.Series) -> pd.Series:
    net = pd.to_numeric(net, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + net).cumprod()

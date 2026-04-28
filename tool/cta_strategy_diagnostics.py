"""
CTA strategy diagnostics: PnL attribution, position stats, trade markers for candlestick plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tool.cta_signal_lab import BacktestConfig
from tool.cta_catalog_execution import net_return_with_costs


def per_signal_net_series(
    pos_mat: pd.DataFrame,
    ret_1: pd.Series,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Fee-adjusted net return for each signal column (standalone, not ensemble)."""
    out: Dict[str, pd.Series] = {}
    for col in pos_mat.columns:
        out[str(col)] = net_return_with_costs(pos_mat[col], ret_1, cfg)
    return pd.DataFrame(out, index=pos_mat.index)


def ensemble_pnl_decomposition(
    pos_ens: pd.Series,
    ret_1: pd.Series,
    net_ens: pd.Series,
) -> pd.DataFrame:
    """
    Split ensemble PnL into long / short / flat gross contribution and fee drag.
    Rows: long_gross, short_gross, flat_period, implied_fee_drag (net - gross).
    """
    pos = pos_ens.reindex(net_ens.index).fillna(0.0)
    r = ret_1.reindex(net_ens.index).fillna(0.0)
    gross = pos * r
    eps = 1e-9
    long_g = gross.where(pos > eps, 0.0)
    short_g = gross.where(pos < -eps, 0.0)
    flat_g = gross.where((pos >= -eps) & (pos <= eps), 0.0)
    return pd.DataFrame(
        {
            "component": ["long_gross_sum", "short_gross_sum", "flat_gross_sum", "total_gross_sum", "total_net_sum", "fee_and_slip_drag"],
            "value": [
                float(long_g.sum()),
                float(short_g.sum()),
                float(flat_g.sum()),
                float(gross.sum()),
                float(net_ens.sum()),
                float(net_ens.sum() - gross.sum()),
            ],
        }
    )


def standalone_vs_ensemble_table(net_per_sig: pd.DataFrame, net_ens: pd.Series) -> pd.DataFrame:
    """Total cumulative log-return proxy: sum of per-period net for each column vs ensemble."""
    idx = net_ens.index
    s = net_per_sig.reindex(idx)
    rows = []
    for col in s.columns:
        nv = s[col].fillna(0.0)
        rows.append(
            {
                "signal": col[:80],
                "standalone_total_net": float(nv.sum()),
                "standalone_ann_approx": float(nv.mean() * 525_600),
            }
        )
    df = pd.DataFrame(rows).sort_values(by="standalone_total_net", ascending=False)
    df["ensemble_total_net"] = float(net_ens.fillna(0.0).sum())
    return df


@dataclass
class PositionStats:
    mean_abs_pos: float
    pct_time_long: float
    pct_time_short: float
    pct_time_flat: float
    turnover_mean: float
    n_position_changes: int
    avg_bars_between_changes: float


def position_management_stats(pos: pd.Series, cfg: BacktestConfig) -> PositionStats:
    pos = pd.to_numeric(pos, errors="coerce").fillna(0.0)
    eps = 1e-6
    n = len(pos)
    pct_long = float((pos > eps).sum() / n) if n else 0.0
    pct_short = float((pos < -eps).sum() / n) if n else 0.0
    pct_flat = float(1.0 - pct_long - pct_short)
    turn = pos.diff().abs().fillna(0.0)
    tmean = float(turn.mean())
    steps = turn > 1e-9
    nchg = int(steps.sum())
    avg_gap = float(n / max(nchg, 1))
    return PositionStats(
        mean_abs_pos=float(pos.abs().mean()),
        pct_time_long=pct_long,
        pct_time_short=pct_short,
        pct_time_flat=pct_flat,
        turnover_mean=tmean,
        n_position_changes=nchg,
        avg_bars_between_changes=avg_gap,
    )


def trade_markers_from_position(
    pos: pd.Series,
    *,
    eps: float = 0.02,
) -> Tuple[pd.Index, pd.Index, pd.Index, pd.Index]:
    """
    Causal markers (bar close):
    - long_entry: flat/short -> long (pos > eps)
    - long_exit: long -> flat/short
    - short_entry: flat/long -> short
    - short_cover: short -> flat/long
    """
    p = pd.to_numeric(pos, errors="coerce").fillna(0.0)
    prev = p.shift(1).fillna(0.0)
    le = (prev <= eps) & (p > eps)
    lx = (prev > eps) & (p <= eps)
    se = (prev >= -eps) & (p < -eps)
    sc = (prev < -eps) & (p >= -eps)
    return p.index[le], p.index[lx], p.index[se], p.index[sc]


def ohlc_for_plot(
    df: pd.DataFrame,
    *,
    resample_rule: Optional[str] = None,
    tail: Optional[int] = None,
) -> pd.DataFrame:
    """Datetime index + OHLC columns; optional resample and tail for visibility."""
    out = df[["open", "high", "low", "close"]].copy()
    out.index = pd.to_datetime(df["time_utc"], utc=True)
    if resample_rule:
        out = out.resample(resample_rule, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        out = out.dropna(subset=["close"])
    if tail is not None:
        out = out.tail(int(tail))
    return out


def plot_candles_with_trade_markers(
    ohlc: pd.DataFrame,
    buy_idx: pd.Index,
    sell_idx: pd.Index,
    short_idx: Optional[pd.Index] = None,
    cover_idx: Optional[pd.Index] = None,
    *,
    title: str = "Price + strategy markers",
    figsize: Tuple[float, float] = (14, 5),
):
    """
    Matplotlib candlesticks + scatter markers. `ohlc` datetime index.
    buy/sell/short/cover are **time_ms** or datetime index values matching underlying 1m markers;
    for resampled ohlc, pass indices aligned via `align_marker_index_to_ohlc` externally as booleans — simpler API below.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(ohlc))
    w = 0.6
    for i in range(len(ohlc)):
        o = float(ohlc["open"].iloc[i])
        h = float(ohlc["high"].iloc[i])
        l = float(ohlc["low"].iloc[i])
        c = float(ohlc["close"].iloc[i])
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color="#333", linewidth=0.6, zorder=1)
        body_b = min(o, c)
        body_t = max(o, c)
        height = max(body_t - body_b, 1e-12)
        ax.add_patch(Rectangle((i - w / 2, body_b), w, height, facecolor=col, edgecolor=col, zorder=2))

    def _scatter(ts_idx: pd.Index, color: str, marker: str, lab: str, yoff: float) -> None:
        if ts_idx is None or len(ts_idx) == 0:
            return
        pos_x: List[int] = []
        ys_list: List[float] = []
        for ix in ts_idx:
            ts = pd.to_datetime(ix, utc=True) if not isinstance(ix, pd.Timestamp) else ix
            hit = int(ohlc.index.get_indexer([ts], method="nearest")[0])
            if 0 <= hit < len(ohlc):
                pos_x.append(hit)
                ys_list.append(float(ohlc["low"].iloc[hit]) * (1.0 - yoff))
        if not pos_x:
            return
        ax.scatter(pos_x, ys_list, color=color, marker=marker, s=55, zorder=5, label=lab, edgecolors="white", linewidths=0.4)

    _scatter(buy_idx, "#16a34a", "^", "Long entry / cover short", 0.0012)
    _scatter(sell_idx, "#dc2626", "v", "Long exit / short entry", 0.0012)
    if short_idx is not None and len(short_idx):
        _scatter(short_idx, "#a855f7", "v", "Short entry", 0.0025)
    if cover_idx is not None and len(cover_idx):
        _scatter(cover_idx, "#2563eb", "^", "Cover short", 0.0025)

    tick_step = max(len(ohlc) // 8, 1)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([ohlc.index[i].strftime("%m-%d %H:%M") for i in range(0, len(ohlc), tick_step)], rotation=25, ha="right")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    return fig, ax


def plot_candles_with_marks_simple(
    df_1m: pd.DataFrame,
    pos_ens: pd.Series,
    *,
    resample_rule: str = "15min",
    tail_bars: int = 500,
    eps: float = 0.02,
    title: str = "OHLC + ensemble trade markers (resampled for visibility)",
):
    """
    Convenience: resample OHLC, recompute markers on 1m then map to resampled bars by time.
    """
    le, lx, se, sc = trade_markers_from_position(pos_ens, eps=eps)
    ohlc = ohlc_for_plot(df_1m, resample_rule=resample_rule, tail=tail_bars)
    # map marker times to show on resampled chart: use nearest bar index in ohlc
    idx_utc = ohlc.index

    def _nearest(ts_idx: pd.Index) -> pd.Index:
        if len(ts_idx) == 0:
            return ts_idx
        out: List[pd.Timestamp] = []
        for ix in ts_idx:
            ts = pd.to_datetime(ix, unit="ms", utc=True) if isinstance(ix, (int, np.integer)) else pd.to_datetime(ix, utc=True)
            j = int(idx_utc.get_indexer([ts], method="nearest")[0])
            if 0 <= j < len(idx_utc):
                out.append(idx_utc[j])
        return pd.Index(out)

    return plot_candles_with_trade_markers(
        ohlc,
        _nearest(le),
        _nearest(lx),
        _nearest(se),
        _nearest(sc),
        title=title,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RegimeCTAConfig:
    vol_window: int = 60
    fee_bps: float = 3.0
    slippage_bps: float = 1.0
    signal_shift: int = 1
    max_abs_position: float = 1.0
    position_step: float = 0.1


def to_utc_index(index_like: pd.Index) -> pd.Index:
    idx = pd.Index(index_like)
    if len(idx) == 0:
        return idx
    num = pd.to_numeric(pd.Series(idx), errors="coerce")
    if num.notna().all():
        vals = num.astype("int64")
        if vals.median() >= 100_000_000_000:
            return pd.to_datetime(vals, unit="ms", utc=True)
    if np.issubdtype(idx.dtype, np.datetime64):
        return pd.to_datetime(idx, utc=True)
    return idx


def build_expanding_vol_regime(price_ret: pd.Series, vol_window: int = 60) -> pd.Series:
    """
    Build volatility regimes (0/1/2) without lookahead:
    - vol_t = rolling std of returns over vol_window
    - thresholds at time t use expanding quantiles from history up to t-1
    """
    r = pd.to_numeric(price_ret, errors="coerce")
    vol = r.rolling(vol_window, min_periods=vol_window).std()

    q33 = vol.expanding(min_periods=vol_window).quantile(1.0 / 3.0).shift(1)
    q67 = vol.expanding(min_periods=vol_window).quantile(2.0 / 3.0).shift(1)

    regime = pd.Series(np.nan, index=vol.index, dtype="float64")
    m0 = vol <= q33
    m1 = (vol > q33) & (vol <= q67)
    m2 = vol > q67
    regime[m0] = 0.0
    regime[m1] = 1.0
    regime[m2] = 2.0
    return regime


def build_position_matrix_from_traces(traces: Dict[str, pd.DataFrame], signal_names: Iterable[str]) -> pd.DataFrame:
    frames = []
    for name in signal_names:
        tr = traces.get(name)
        if tr is None or tr.empty or "position" not in tr.columns:
            continue
        frames.append(tr[["position"]].rename(columns={"position": name}))
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return out.sort_index().fillna(0.0)


def build_regime_position(
    position_matrix: pd.DataFrame,
    regime: pd.Series,
    regime_signal_map: Dict[int, List[str]],
    *,
    max_abs_position: float = 1.0,
    position_step: float = 0.1,
) -> pd.Series:
    if position_matrix.empty:
        return pd.Series(dtype="float64")

    idx = position_matrix.index.union(regime.index)
    pm = position_matrix.reindex(idx).fillna(0.0)
    rg = pd.to_numeric(regime.reindex(idx), errors="coerce")

    out = pd.Series(0.0, index=idx, dtype="float64")
    for reg_value, sigs in regime_signal_map.items():
        valid_sigs = [s for s in sigs if s in pm.columns]
        if not valid_sigs:
            continue
        mask = rg == float(reg_value)
        out.loc[mask] = pm.loc[mask, valid_sigs].mean(axis=1)

    out = out.clip(-abs(max_abs_position), abs(max_abs_position))
    if position_step and position_step > 0:
        step = float(position_step)
        out = (np.round(out / step) * step).clip(-abs(max_abs_position), abs(max_abs_position))
    return out


def build_2d_regime_code(vol_regime: pd.Series, whale_regime: pd.Series) -> pd.Series:
    """
    Combine two 3-state regimes into one integer code:
      code = vol_regime * 3 + whale_regime
    Output range: 0..8
    """
    idx = vol_regime.index.union(whale_regime.index)
    v = pd.to_numeric(vol_regime.reindex(idx), errors="coerce")
    w = pd.to_numeric(whale_regime.reindex(idx), errors="coerce")
    out = pd.Series(np.nan, index=idx, dtype="float64")
    mask = v.notna() & w.notna()
    out.loc[mask] = (v.loc[mask].astype(int) * 3 + w.loc[mask].astype(int)).astype(float)
    return out


def apply_risk_layer_multiplier(
    position: pd.Series,
    regime_code: pd.Series,
    multiplier_map: Dict[int, float],
    *,
    max_abs_position: float = 1.0,
    position_step: float = 0.1,
) -> pd.Series:
    idx = position.index.union(regime_code.index)
    p = pd.to_numeric(position.reindex(idx), errors="coerce").fillna(0.0)
    r = pd.to_numeric(regime_code.reindex(idx), errors="coerce")
    mult = r.map(multiplier_map).fillna(1.0)
    out = p * mult
    out = out.clip(-abs(max_abs_position), abs(max_abs_position))
    if position_step and position_step > 0:
        step = float(position_step)
        out = (np.round(out / step) * step).clip(-abs(max_abs_position), abs(max_abs_position))
    return out


def backtest_regime_strategy(
    position: pd.Series,
    ret_series: pd.Series,
    regime: pd.Series,
    cfg: RegimeCTAConfig,
) -> pd.DataFrame:
    idx = position.index.union(ret_series.index).union(regime.index)
    pos = pd.to_numeric(position.reindex(idx), errors="coerce").fillna(0.0)
    ret = pd.to_numeric(ret_series.reindex(idx), errors="coerce").fillna(0.0)
    reg = pd.to_numeric(regime.reindex(idx), errors="coerce")

    exec_pos = pos.shift(cfg.signal_shift).fillna(0.0)
    gross = exec_pos * ret
    turnover = exec_pos.diff().abs().fillna(0.0)
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    cost = turnover * cost_rate
    net = gross - cost
    equity = (1.0 + net).cumprod()

    return pd.DataFrame(
        {
            "regime": reg,
            "position_raw": pos,
            "position_exec": exec_pos,
            "ret": ret,
            "gross_ret": gross,
            "cost_ret": cost,
            "net_ret": net,
            "turnover": turnover,
            "equity": equity,
        },
        index=idx,
    )


def summarize_regime_performance(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = bt.dropna(subset=["regime"]).copy()
    if t.empty:
        return pd.DataFrame(columns=["regime", "count", "mean_ret", "std_ret", "sharpe_like", "turnover_mean"])
    for reg, grp in t.groupby("regime"):
        mean_ret = float(grp["net_ret"].mean())
        std_ret = float(grp["net_ret"].std(ddof=0))
        sharpe_like = mean_ret / std_ret if std_ret > 0 else np.nan
        rows.append(
            {
                "regime": int(reg),
                "count": int(len(grp)),
                "mean_ret": mean_ret,
                "std_ret": std_ret,
                "sharpe_like": float(sharpe_like) if sharpe_like == sharpe_like else np.nan,
                "turnover_mean": float(grp["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime")


def plot_regime_cta_dashboard(bt: pd.DataFrame, regime_summary: pd.DataFrame, title_prefix: str = "Regime CTA") -> None:
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.2)

    # Equity
    ax_eq = fig.add_subplot(gs[0, :])
    x = to_utc_index(bt.index)
    ax_eq.plot(x, bt["equity"], color="#2563eb", linewidth=1.4, label="Equity")
    ax_eq.set_title(f"{title_prefix} - Equity Curve (net of cost)")
    ax_eq.grid(alpha=0.25, linestyle="--")
    ax_eq.legend()

    # Position
    ax_pos = fig.add_subplot(gs[1, 0])
    ax_pos.plot(x, bt["position_exec"], color="#0f766e", linewidth=1.0)
    ax_pos.set_title("Executed Position")
    ax_pos.set_ylim(-1.05, 1.05)
    ax_pos.grid(alpha=0.25, linestyle="--")

    # Regime bar (mean_ret / sharpe_like)
    ax_bar = fig.add_subplot(gs[1, 1])
    if not regime_summary.empty:
        labels = [f"regime_{int(r)}" for r in regime_summary["regime"].tolist()]
        xx = np.arange(len(labels))
        w = 0.35
        ax_bar.bar(xx - w / 2, regime_summary["mean_ret"], width=w, label="mean_ret")
        ax_bar.bar(xx + w / 2, regime_summary["sharpe_like"], width=w, label="sharpe_like")
        ax_bar.set_xticks(xx)
        ax_bar.set_xticklabels(labels)
        ax_bar.set_title("Regime Performance Snapshot")
        ax_bar.grid(alpha=0.2, axis="y", linestyle="--")
        ax_bar.legend()
    else:
        ax_bar.set_title("Regime Performance Snapshot (no data)")

    plt.tight_layout()
    plt.show()

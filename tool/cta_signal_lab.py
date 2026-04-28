from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tool.newmath import apply_formulas


@dataclass
class BacktestConfig:
    freq: str = "1h"
    fee_bps: float = 3.0
    slippage_bps: float = 1.0
    signal_shift: int = 1
    signal_clip: float = 2.5
    max_abs_position: float = 1.0
    position_step: float = 0.1
    annual_bars: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.annual_bars is None:
            self.annual_bars = {
                "1m": 525_600,
                "5m": 105_120,
                "15m": 35_040,
                "30m": 17_520,
                "1h": 8_760,
            }


def flatten_signal_library(signal_library: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    formulas: Dict[str, str] = {}
    for group_name, group_signals in signal_library.items():
        for signal_name, formula in group_signals.items():
            col = f"sig_{group_name}__{signal_name}"
            formulas[col] = formula
    return formulas


def build_signal_features(
    df: pd.DataFrame,
    signal_library: Dict[str, Dict[str, str]],
) -> Tuple[pd.DataFrame, List[str]]:
    formulas = flatten_signal_library(signal_library)
    out = apply_formulas(df, formulas)
    signal_cols = list(formulas.keys())
    return out, signal_cols


def robust_zscore(signal: pd.Series, window: int = 240) -> pd.Series:
    med = signal.rolling(window, min_periods=window).median()
    mad = (signal - med).abs().rolling(window, min_periods=window).median()
    denom = (1.4826 * mad).replace(0, np.nan)
    return (signal - med) / denom


def _apply_z_entry_deadband(
    z: pd.Series,
    *,
    deadband: float,
    deadband_long: Optional[float],
    deadband_short: Optional[float],
) -> pd.Series:
    """
    After rolling robust-z normalization, require |z| large enough to trade:
    - Long-side contribution only if z >= tl (tl = deadband_long or deadband).
    - Short-side only if z <= -ts (ts = deadband_short or deadband).
    Otherwise z is set to 0 (flat). Symmetric case: tl = ts = deadband → flat on (-deadband, deadband).
    """
    tl = float(deadband_long) if deadband_long is not None else float(deadband)
    ts = float(deadband_short) if deadband_short is not None else float(deadband)
    # If symmetric deadband is 0 but only one arm is set, mirror it so we do not treat ts==0 as "all z<=0 short".
    if deadband == 0.0:
        if deadband_long is not None and deadband_short is None:
            ts = tl
        elif deadband_short is not None and deadband_long is None:
            tl = ts
    if tl <= 0.0 and ts <= 0.0:
        return z
    return z.where((z >= tl) | (z <= -ts), 0.0)


def signal_to_position(
    signal: pd.Series,
    *,
    clip: float = 2.5,
    z_window: int = 240,
    mode: str = "tanh",
    deadband: float = 0.0,
    deadband_long: Optional[float] = None,
    deadband_short: Optional[float] = None,
    smooth_span: int = 0,
    max_abs_position: float = 1.0,
    position_step: float = 0.1,
) -> pd.Series:
    """
    Turn a raw formula signal column into a position series (causal).

    Pipeline:
    1) **Raw signal** — from `build_signal_features` / `build_core_signal_library` (mixed units per name).
    2) **Rolling robust z-score** — `robust_zscore(signal, z_window)` using median/MAD, then clip to ±`clip`.
       This puts heterogeneous signals on a comparable scale before sizing.
    3) **Entry deadband (threshold)** — in **z-space**, not raw signal units: unless z is clearly positive
       or clearly negative, force flat to cut minute-bar noise. Use `deadband` for symmetric
       ``|z| >= deadband``, or set `deadband_long` / `deadband_short` for asymmetric arms.
    4) **Sizing** — `tanh(z/clip)` or `sign(z)`; optional EWM smooth; clip; discrete `position_step`.
    """
    z = robust_zscore(signal, window=z_window).clip(-clip, clip)
    z = _apply_z_entry_deadband(
        z, deadband=deadband, deadband_long=deadband_long, deadband_short=deadband_short
    )
    if mode == "sign":
        pos = np.sign(z)
    else:
        pos = np.tanh(z / clip)
    pos = pd.Series(pos, index=signal.index, dtype="float64")
    if smooth_span and smooth_span > 1:
        pos = pos.ewm(span=smooth_span, adjust=False, min_periods=1).mean()
    # No-leverage guardrail.
    pos = pos.clip(-abs(max_abs_position), abs(max_abs_position))
    # Discrete execution ladder (e.g., 10% per step).
    if position_step and position_step > 0:
        step = float(position_step)
        pos = (np.round(pos / step) * step).clip(-abs(max_abs_position), abs(max_abs_position))
    return pos


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return np.nan
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())


def _annual_factor(freq: str, annual_bars: Dict[str, int]) -> float:
    return float(annual_bars.get(freq, 8_760))


def _to_utc_plot_index(index_like: pd.Index) -> pd.Index:
    """
    Convert ms-based index to UTC datetime index for plotting.
    Keeps original index when conversion is not applicable.
    """
    idx = pd.Index(index_like)
    if len(idx) == 0:
        return idx

    num = pd.to_numeric(pd.Series(idx), errors="coerce")
    if num.notna().all():
        vals = num.astype("int64")
        # Heuristic: unix epoch in milliseconds is typically >= 1e11.
        if vals.median() >= 100_000_000_000:
            return pd.to_datetime(vals, unit="ms", utc=True)

    # If index is already datetime-like, normalize to UTC.
    if np.issubdtype(idx.dtype, np.datetime64):
        return pd.to_datetime(idx, utc=True)

    return idx


def evaluate_single_signal(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str,
    cfg: BacktestConfig,
    z_window: int = 240,
    position_mode: str = "tanh",
    deadband: float = 0.0,
    deadband_long: Optional[float] = None,
    deadband_short: Optional[float] = None,
    smooth_span: int = 0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    t = df[[signal_col, ret_col]].copy()
    t = t.replace([np.inf, -np.inf], np.nan).dropna()
    if t.empty:
        empty_metrics = {
            "obs": 0,
            "ann_ret": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
            "max_dd": np.nan,
            "turnover": np.nan,
            "hit_rate": np.nan,
            "avg_trade_pnl": np.nan,
            "profit_factor": np.nan,
        }
        return empty_metrics, pd.DataFrame(index=t.index)

    pos = signal_to_position(
        t[signal_col],
        clip=cfg.signal_clip,
        z_window=z_window,
        mode=position_mode,
        deadband=deadband,
        deadband_long=deadband_long,
        deadband_short=deadband_short,
        smooth_span=smooth_span,
        max_abs_position=cfg.max_abs_position,
        position_step=cfg.position_step,
    ).shift(cfg.signal_shift)
    pos = pos.fillna(0.0)

    gross = pos * t[ret_col]
    turnover = pos.diff().abs().fillna(0.0)
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    cost = turnover * cost_rate
    net = gross - cost
    eq = (1.0 + net).cumprod()

    bars = _annual_factor(cfg.freq, cfg.annual_bars)
    ann_ret = float(net.mean() * bars)
    ann_vol = float(net.std(ddof=0) * np.sqrt(bars))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd = _max_drawdown(eq)
    calmar = ann_ret / abs(max_dd) if (max_dd is not None and max_dd < 0) else np.nan

    active = turnover > 0
    trade_pnl = net[active]
    gross_pos = gross[gross > 0].sum()
    gross_neg = -gross[gross < 0].sum()
    profit_factor = float(gross_pos / gross_neg) if gross_neg > 0 else np.nan

    result = pd.DataFrame(
        {
            "position": pos,
            "gross_ret": gross,
            "cost_ret": cost,
            "net_ret": net,
            "equity": eq,
            "turnover": turnover,
        },
        index=t.index,
    )
    metrics = {
        "obs": int(len(result)),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "calmar": float(calmar) if calmar == calmar else np.nan,
        "max_dd": float(max_dd) if max_dd == max_dd else np.nan,
        "turnover": float(turnover.mean()),
        "hit_rate": float((net > 0).mean()),
        "avg_trade_pnl": float(trade_pnl.mean()) if len(trade_pnl) > 0 else np.nan,
        "profit_factor": profit_factor,
    }
    return metrics, result


def _metrics_from_net_ret(net_ret: pd.Series, cfg: BacktestConfig) -> Dict[str, float]:
    """Point-in-time metrics for a net return series (single contiguous segment)."""
    net_ret = pd.to_numeric(net_ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if net_ret.empty:
        return {
            "obs": 0,
            "ann_ret": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
            "max_dd": np.nan,
            "turnover": np.nan,
            "hit_rate": np.nan,
            "profit_factor": np.nan,
        }
    bars = _annual_factor(cfg.freq, cfg.annual_bars)
    ann_ret = float(net_ret.mean() * bars)
    ann_vol = float(net_ret.std(ddof=0) * np.sqrt(bars))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = (1.0 + net_ret).cumprod()
    max_dd = _max_drawdown(eq)
    calmar = ann_ret / abs(max_dd) if (max_dd is not None and max_dd < 0) else np.nan
    gross = net_ret  # already net of cost in caller
    gross_pos = gross[gross > 0].sum()
    gross_neg = -gross[gross < 0].sum()
    profit_factor = float(gross_pos / gross_neg) if gross_neg > 0 else np.nan
    return {
        "obs": int(len(net_ret)),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "calmar": float(calmar) if calmar == calmar else np.nan,
        "max_dd": float(max_dd) if max_dd == max_dd else np.nan,
        "turnover": np.nan,
        "hit_rate": float((net_ret > 0).mean()),
        "profit_factor": profit_factor,
    }


def compute_strategy_metrics(
    net_ret: pd.Series,
    cfg: BacktestConfig,
    position: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Point-in-time metrics for a **net** return series (costs already deducted).
    Annualization uses ``cfg.freq`` / ``cfg.annual_bars``. Optional ``position`` adds mean turnover.
    """
    net_ret = pd.to_numeric(net_ret, errors="coerce").replace([np.inf, -np.inf], np.nan)
    idx = net_ret.dropna().index
    if len(idx) == 0:
        out = _metrics_from_net_ret(pd.Series(dtype=float), cfg)
        if position is not None:
            out["turnover"] = np.nan
        return out
    net_clean = net_ret.loc[idx]
    m = _metrics_from_net_ret(net_clean, cfg)
    if position is not None:
        pos = pd.to_numeric(position.reindex(idx), errors="coerce").fillna(0.0)
        turn = pos.diff().abs().fillna(0.0)
        m["turnover"] = float(turn.mean())
    return m


def evaluate_single_signal_train_test(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str,
    cfg: BacktestConfig,
    train_mask: pd.Series,
    test_mask: pd.Series,
    z_window: int = 240,
    position_mode: str = "tanh",
    deadband: float = 0.0,
    deadband_long: Optional[float] = None,
    deadband_short: Optional[float] = None,
    smooth_span: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """
    Same execution as evaluate_single_signal (causal rolling z-score on full sample),
    but Sharpe/returns are computed separately on train vs test index subsets.
    """
    t = df[[signal_col, ret_col]].copy()
    t = t.replace([np.inf, -np.inf], np.nan).dropna()
    if t.empty:
        empty = _metrics_from_net_ret(pd.Series(dtype=float), cfg)
        return empty, empty, pd.DataFrame(index=t.index)

    idx = t.index
    tr_m = pd.to_numeric(train_mask.reindex(idx), errors="coerce").fillna(False).astype(bool)
    te_m = pd.to_numeric(test_mask.reindex(idx), errors="coerce").fillna(False).astype(bool)

    pos = signal_to_position(
        t[signal_col],
        clip=cfg.signal_clip,
        z_window=z_window,
        mode=position_mode,
        deadband=deadband,
        deadband_long=deadband_long,
        deadband_short=deadband_short,
        smooth_span=smooth_span,
        max_abs_position=cfg.max_abs_position,
        position_step=cfg.position_step,
    ).shift(cfg.signal_shift)
    pos = pos.fillna(0.0)

    gross = pos * t[ret_col]
    turnover = pos.diff().abs().fillna(0.0)
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    cost = turnover * cost_rate
    net = gross - cost

    train_metrics = _metrics_from_net_ret(net[tr_m], cfg)
    train_metrics["turnover"] = float(turnover[tr_m].mean()) if tr_m.any() else np.nan

    test_metrics = _metrics_from_net_ret(net[te_m], cfg)
    test_metrics["turnover"] = float(turnover[te_m].mean()) if te_m.any() else np.nan

    result = pd.DataFrame(
        {
            "position": pos,
            "gross_ret": gross,
            "cost_ret": cost,
            "net_ret": net,
            "equity": (1.0 + net).cumprod(),
            "turnover": turnover,
            "is_train": tr_m,
            "is_test": te_m,
        },
        index=idx,
    )
    return train_metrics, test_metrics, result


def evaluate_signal_library_train_test(
    df: pd.DataFrame,
    signal_cols: Iterable[str],
    ret_col: str,
    train_mask: pd.Series,
    test_mask: pd.Series,
    cfg: BacktestConfig = None,
    z_window: int = 240,
    position_mode: str = "tanh",
    deadband: float = 0.0,
    deadband_long: Optional[float] = None,
    deadband_short: Optional[float] = None,
    smooth_span: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Per-signal train and test metrics; full-period traces for plotting / ensemble.
    Sort key: train Sharpe (desc).
    """
    if cfg is None:
        cfg = BacktestConfig()

    rows: List[Dict[str, float]] = []
    traces: Dict[str, pd.DataFrame] = {}

    for sig in signal_cols:
        m_tr, m_te, trace = evaluate_single_signal_train_test(
            df=df,
            signal_col=sig,
            ret_col=ret_col,
            cfg=cfg,
            train_mask=train_mask,
            test_mask=test_mask,
            z_window=z_window,
            position_mode=position_mode,
            deadband=deadband,
            deadband_long=deadband_long,
            deadband_short=deadband_short,
            smooth_span=smooth_span,
        )
        row: Dict[str, float] = {"signal": sig}
        for k, v in m_tr.items():
            row[f"train_{k}"] = v
        for k, v in m_te.items():
            row[f"test_{k}"] = v
        rows.append(row)
        traces[sig] = trace

    report = pd.DataFrame(rows).set_index("signal")
    if "train_sharpe" in report.columns:
        report = report.sort_values(by=["train_sharpe", "test_sharpe"], ascending=False)
    return report, traces


def evaluate_signal_library(
    df: pd.DataFrame,
    signal_cols: Iterable[str],
    ret_col: str = "ret_1",
    cfg: BacktestConfig = None,
    z_window: int = 240,
    position_mode: str = "tanh",
    deadband: float = 0.0,
    deadband_long: Optional[float] = None,
    deadband_short: Optional[float] = None,
    smooth_span: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if cfg is None:
        cfg = BacktestConfig()

    rows: List[Dict[str, float]] = []
    traces: Dict[str, pd.DataFrame] = {}

    for sig in signal_cols:
        metrics, trace = evaluate_single_signal(
            df=df,
            signal_col=sig,
            ret_col=ret_col,
            cfg=cfg,
            z_window=z_window,
            position_mode=position_mode,
            deadband=deadband,
            deadband_long=deadband_long,
            deadband_short=deadband_short,
            smooth_span=smooth_span,
        )
        metrics["signal"] = sig
        rows.append(metrics)
        traces[sig] = trace

    report = pd.DataFrame(rows).set_index("signal").sort_values(
        by=["sharpe", "calmar", "ann_ret"],
        ascending=False,
    )
    return report, traces


def rank_signals(report: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    cols = ["ann_ret", "ann_vol", "sharpe", "calmar", "max_dd", "turnover", "hit_rate", "profit_factor"]
    keep = [c for c in cols if c in report.columns]
    out = report[keep].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.head(top_n)


def _annualized_stats(net_ret: pd.Series, bars_per_year: float) -> Tuple[float, float, float]:
    if net_ret.empty:
        return np.nan, np.nan, np.nan
    ann_ret = float(net_ret.mean() * bars_per_year)
    ann_vol = float(net_ret.std(ddof=0) * np.sqrt(bars_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe


def add_robustness_columns(
    report: pd.DataFrame,
    traces: Dict[str, pd.DataFrame],
    cfg: BacktestConfig,
    n_splits: int = 6,
) -> pd.DataFrame:
    out = report.copy()
    bars = _annual_factor(cfg.freq, cfg.annual_bars)

    robust_rows = []
    for sig in out.index:
        tr = traces.get(sig)
        if tr is None or tr.empty or "net_ret" not in tr.columns:
            robust_rows.append(
                {
                    "signal": sig,
                    "wf_sharpe_min": np.nan,
                    "wf_sharpe_mean": np.nan,
                    "wf_sharpe_std": np.nan,
                    "robust_score": np.nan,
                }
            )
            continue

        net = tr["net_ret"].dropna()
        if len(net) < max(120, n_splits * 20):
            robust_rows.append(
                {
                    "signal": sig,
                    "wf_sharpe_min": np.nan,
                    "wf_sharpe_mean": np.nan,
                    "wf_sharpe_std": np.nan,
                    "robust_score": np.nan,
                }
            )
            continue

        chunk_size = len(net) // n_splits
        fold_sharpes = []
        for i in range(n_splits):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_splits - 1 else len(net)
            sub = net.iloc[start:end]
            _, _, s = _annualized_stats(sub, bars_per_year=bars)
            if s == s:
                fold_sharpes.append(float(s))

        if not fold_sharpes:
            wf_min = np.nan
            wf_mean = np.nan
            wf_std = np.nan
            robust = np.nan
        else:
            wf_min = float(np.min(fold_sharpes))
            wf_mean = float(np.mean(fold_sharpes))
            wf_std = float(np.std(fold_sharpes))
            if "sharpe" in out.columns:
                base_sharpe = float(out.loc[sig, "sharpe"])
            elif "train_sharpe" in out.columns:
                base_sharpe = float(out.loc[sig, "train_sharpe"])
            else:
                base_sharpe = wf_mean
            # Prefer high overall Sharpe, high worst-fold Sharpe, low dispersion.
            robust = base_sharpe + 0.45 * wf_min - 0.25 * wf_std

        robust_rows.append(
            {
                "signal": sig,
                "wf_sharpe_min": wf_min,
                "wf_sharpe_mean": wf_mean,
                "wf_sharpe_std": wf_std,
                "robust_score": robust,
            }
        )

    robust_df = pd.DataFrame(robust_rows).set_index("signal")
    out = out.join(robust_df, how="left")
    # Sort keys differ between evaluate_signal_library (sharpe/calmar) and
    # evaluate_signal_library_train_test (train_sharpe/train_calmar, etc.).
    sort_by = [c for c in ("robust_score", "sharpe", "calmar") if c in out.columns]
    if not sort_by:
        sort_by = [c for c in ("robust_score", "train_sharpe", "train_calmar") if c in out.columns]
    if sort_by:
        out = out.sort_values(by=sort_by, ascending=False)
    return out


def plot_signal_dashboard(
    report: pd.DataFrame,
    traces: Dict[str, pd.DataFrame],
    signal_matrix: pd.DataFrame,
    signal_cols: List[str],
    top_n: int = 6,
) -> None:
    top = report.head(top_n).index.tolist()
    if not top:
        print("No signals available for plotting.")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.28, wspace=0.2)

    ax_eq = fig.add_subplot(gs[0, :])
    for sig in top:
        tr = traces.get(sig)
        if tr is None or tr.empty:
            continue
        x = _to_utc_plot_index(tr.index)
        ax_eq.plot(x, tr["equity"], label=sig.replace("sig_", ""), linewidth=1.2)
    ax_eq.set_title("Top Signal Equity Curves (net of cost)")
    ax_eq.legend(loc="upper left", ncol=2, fontsize=8)
    ax_eq.grid(alpha=0.25, linestyle="--")

    ax_sharpe = fig.add_subplot(gs[1, 0])
    report.loc[top, "sharpe"].sort_values().plot(kind="barh", ax=ax_sharpe, color="#3b82f6")
    ax_sharpe.set_title("Top Signal Sharpe")
    ax_sharpe.grid(alpha=0.2, axis="x", linestyle="--")

    ax_calmar = fig.add_subplot(gs[1, 1])
    report.loc[top, "calmar"].sort_values().plot(kind="barh", ax=ax_calmar, color="#10b981")
    ax_calmar.set_title("Top Signal Calmar")
    ax_calmar.grid(alpha=0.2, axis="x", linestyle="--")

    # Bottom-left: regime stability snapshot
    ax_reg = fig.add_subplot(gs[2, 0])
    reg_top = top[: min(4, len(top))]
    reg_records = []
    for sig in reg_top:
        tr = traces.get(sig)
        if tr is None or tr.empty or "net_ret" not in tr.columns:
            continue
        vol = tr["net_ret"].rolling(96, min_periods=96).std()
        vol_bucket = pd.qcut(vol.rank(method="first"), q=3, labels=["low_vol", "mid_vol", "high_vol"])
        tmp = pd.DataFrame({"signal": sig, "net_ret": tr["net_ret"], "regime": vol_bucket})
        reg_records.append(tmp.dropna())

    if reg_records:
        reg = pd.concat(reg_records, axis=0)
        labels = ["low_vol", "mid_vol", "high_vol"]
        width = 0.22
        x = np.arange(len(reg_top))
        for i, lab in enumerate(labels):
            vals = []
            for sig in reg_top:
                sub = reg[(reg["signal"] == sig) & (reg["regime"] == lab)]["net_ret"]
                vals.append(float(sub.mean()) if len(sub) else np.nan)
            ax_reg.bar(x + (i - 1) * width, vals, width=width, label=lab)
        ax_reg.set_xticks(x)
        ax_reg.set_xticklabels([s.replace("sig_", "") for s in reg_top], rotation=35, ha="right")
        ax_reg.set_title("Regime Mean Return (top 4)")
        ax_reg.grid(alpha=0.2, axis="y", linestyle="--")
        ax_reg.legend(fontsize=7)
    else:
        ax_reg.set_title("Regime Mean Return (insufficient data)")
        ax_reg.axis("off")

    # Bottom-right: correlation heatmap
    ax_corr = fig.add_subplot(gs[2, 1])
    corr = signal_matrix[signal_cols].corr().fillna(0.0)
    show_cols = top[: min(10, len(top))]
    corr_show = corr.loc[show_cols, show_cols] if show_cols else corr.iloc[:10, :10]
    im = ax_corr.imshow(corr_show.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(corr_show.columns)))
    ax_corr.set_xticklabels([c.replace("sig_", "") for c in corr_show.columns], rotation=45, ha="right")
    ax_corr.set_yticks(range(len(corr_show.index)))
    ax_corr.set_yticklabels([c.replace("sig_", "") for c in corr_show.index])
    ax_corr.set_title("Signal Correlation Heatmap (top subset)")
    fig.colorbar(im, ax=ax_corr, fraction=0.02, pad=0.01)

    plt.show()


def plot_regime_boxplot(
    traces: Dict[str, pd.DataFrame],
    report: pd.DataFrame,
    top_n: int = 4,
) -> None:
    top = report.head(top_n).index.tolist()
    if not top:
        return

    records = []
    for sig in top:
        tr = traces.get(sig)
        if tr is None or tr.empty:
            continue
        vol = tr["net_ret"].rolling(96, min_periods=96).std()
        vol_bucket = pd.qcut(vol.rank(method="first"), q=3, labels=["low_vol", "mid_vol", "high_vol"])
        tmp = pd.DataFrame({"signal": sig, "net_ret": tr["net_ret"], "regime": vol_bucket})
        records.append(tmp.dropna())
    if not records:
        return

    reg = pd.concat(records, axis=0)
    labels = ["low_vol", "mid_vol", "high_vol"]
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.18
    x = np.arange(len(top))
    for i, lab in enumerate(labels):
        vals = []
        for sig in top:
            sub = reg[(reg["signal"] == sig) & (reg["regime"] == lab)]["net_ret"]
            vals.append(float(sub.mean()) if len(sub) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("sig_", "") for t in top], rotation=30, ha="right")
    ax.set_title("Regime Mean Return by Signal")
    ax.grid(alpha=0.2, axis="y", linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.show()


def build_ensemble_trace(
    traces: Dict[str, pd.DataFrame],
    selected_signals: List[str],
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    if not selected_signals:
        return pd.DataFrame()

    valid = [s for s in selected_signals if s in traces and not traces[s].empty]
    if not valid:
        return pd.DataFrame()

    if weights is None:
        w = {s: 1.0 / len(valid) for s in valid}
    else:
        total = sum(abs(weights.get(s, 0.0)) for s in valid)
        if total <= 0:
            w = {s: 1.0 / len(valid) for s in valid}
        else:
            w = {s: weights.get(s, 0.0) / total for s in valid}

    frame = None
    for s in valid:
        tr = traces[s][["net_ret"]].rename(columns={"net_ret": s})
        frame = tr if frame is None else frame.join(tr, how="outer")
    frame = frame.fillna(0.0)

    ens_ret = sum(frame[s] * w[s] for s in valid)
    out = pd.DataFrame({"ensemble_net_ret": ens_ret}, index=frame.index)
    out["ensemble_equity"] = (1.0 + out["ensemble_net_ret"]).cumprod()
    return out


def plot_ensemble_equity(ensemble_trace: pd.DataFrame, title: str = "Signal Ensemble Equity") -> None:
    if ensemble_trace.empty:
        print("Empty ensemble trace.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    x = _to_utc_plot_index(ensemble_trace.index)
    ax.plot(x, ensemble_trace["ensemble_equity"], color="#ef4444", linewidth=1.5)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.show()


def build_market_vol_regime(
    signal_matrix: pd.DataFrame,
    *,
    vol_source_col: str = "ret_1",
    window: int = 96,
    labels: Tuple[str, ...] = ("low_vol", "mid_vol", "high_vol"),
) -> pd.Series:
    """
    Build market-volatility regimes from underlying asset return volatility.
    Uses rolling std of `vol_source_col` (typically ret_1) and quantile buckets.
    """
    if vol_source_col not in signal_matrix.columns:
        raise ValueError(f"vol_source_col '{vol_source_col}' not found in signal_matrix.")

    base = pd.to_numeric(signal_matrix[vol_source_col], errors="coerce")
    vol = base.rolling(window, min_periods=window).std()
    ranked = vol.rank(method="first")
    regime = pd.qcut(ranked, q=len(labels), labels=list(labels))
    return regime


def evaluate_signals_by_market_vol_regime(
    traces: Dict[str, pd.DataFrame],
    regime: pd.Series,
    selected_signals: List[str],
) -> pd.DataFrame:
    """
    Evaluate each selected signal under market-volatility regimes.
    Returns per-signal, per-regime summary metrics.
    """
    rows = []
    for sig in selected_signals:
        tr = traces.get(sig)
        if tr is None or tr.empty:
            continue

        work = tr.copy()
        work["regime"] = regime.reindex(work.index)
        work = work.dropna(subset=["regime", "net_ret"])
        if work.empty:
            continue

        for reg, grp in work.groupby("regime"):
            mean_ret = float(grp["net_ret"].mean())
            std_ret = float(grp["net_ret"].std(ddof=0))
            sharpe_like = mean_ret / std_ret if std_ret > 0 else np.nan
            rows.append(
                {
                    "signal": sig,
                    "regime": str(reg),
                    "count": int(len(grp)),
                    "mean_ret": mean_ret,
                    "std_ret": std_ret,
                    "sharpe_like": float(sharpe_like) if sharpe_like == sharpe_like else np.nan,
                    "turnover_mean": float(grp["turnover"].mean()) if "turnover" in grp.columns else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["signal", "regime", "count", "mean_ret", "std_ret", "sharpe_like", "turnover_mean"]
        )
    out = pd.DataFrame(rows)
    regime_order = {"low_vol": 0, "mid_vol": 1, "high_vol": 2}
    out = out.sort_values(
        by=["signal", "regime"],
        key=lambda s: s.map(regime_order) if s.name == "regime" else s,
    )
    return out


def plot_market_vol_regime_signal_performance(
    regime_report: pd.DataFrame,
    *,
    metric: str = "mean_ret",
    title: str = "Signal Performance by Market Vol Regime",
) -> None:
    """
    Grouped bar chart for selected metric across low/mid/high vol regimes.
    """
    if regime_report.empty:
        print("Empty regime report.")
        return
    if metric not in regime_report.columns:
        raise ValueError(f"metric '{metric}' not found in regime_report.")

    pivot = regime_report.pivot(index="signal", columns="regime", values=metric)
    cols = [c for c in ["low_vol", "mid_vol", "high_vol"] if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(pivot.index))
    width = 0.22
    for i, col in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2.0) * width, pivot[col].values, width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sig_", "") for s in pivot.index], rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(alpha=0.2, axis="y", linestyle="--")
    ax.legend()
    plt.tight_layout()
    plt.show()


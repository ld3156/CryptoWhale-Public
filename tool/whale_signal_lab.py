"""
Helpers for `main/Whale_Single_Signal_Lab.ipynb` — whale 1m data, IS sweep, ensemble plots.

Keeps the notebook short; behavior matches the original inlined notebook code.
"""
from __future__ import annotations

import warnings
from itertools import product
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tool.cta_catalog_execution import (
    equity_curve_from_1m_net,
    net_return_with_costs,
    upsample_position_to_1m,
)
from tool.cta_multi_freq_lab import (
    add_inverted_signal_columns,
    freq_str_for_bar_minutes,
    z_window_scaled_from_1m_bars,
)
from tool.cta_signal_lab import (
    BacktestConfig,
    add_robustness_columns,
    build_signal_features,
    evaluate_signal_library_train_test,
    signal_to_position,
)
from tool.newmath import apply_formulas, numeric_to_float32

# --- Defaults (notebook may override) ---
OOS_TEST_FRAC = 0.2

COMPOSITE_W_TRAIN = 0.5
COMPOSITE_W_ROBUST = 0.4
COMPOSITE_W_WF_MIN = 0.1
COMPOSITE_W_TRIGGER_PENALTY = 0.08
TRIGGER_TARGET_IS = 24

# Hard filter: IS-only trigger events must exceed this count to enter ranking (user: >15 → >=16).
MIN_IS_TRIGGER_FOR_SWEEP = 15
MIN_OS_TRIGGER_FOR_SWEEP = 10
TOPK_OPT_DEFAULT = 30

BAR_MINUTES_LIST_DEFAULT: List[int] = [15, 30, 60]

EXEC_PROFILES_DEFAULT: List[dict[str, Any]] = [
    {"z1m": 360, "deadband": 0.28, "smooth": 6, "mode": "tanh"},
    {"z1m": 480, "deadband": 0.35, "smooth": 8, "mode": "tanh"},
    {"z1m": 720, "deadband": 0.42, "smooth": 10, "mode": "tanh"},
    {"z1m": 480, "deadband": 0.35, "smooth": 8, "mode": "sign"},
]

CFG_BASE_DEFAULT = BacktestConfig(
    fee_bps=3.0,
    slippage_bps=1.0,
    signal_shift=1,
    signal_clip=2.5,
    max_abs_position=1.0,
    position_step=0.1,
)

VOL_ROLL_WIN = 1440 * 5
BARS_PER_YEAR_1M = 525600
_MINP = max(48, VOL_ROLL_WIN // 5)


def train_test_masks_whale_last_pct(
    df: pd.DataFrame,
    *,
    test_frac: Optional[float] = None,
) -> tuple[pd.Series, pd.Series]:
    if test_frac is None:
        test_frac = OOS_TEST_FRAC
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be in (0, 1)")
    d = df.sort_index()
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 rows for train/test split")
    k_test = max(1, int(round(n * test_frac)))
    k_test = min(k_test, n - 1)
    train_ix = d.index[:-k_test]
    test_ix = d.index[-k_test:]
    train_mask = pd.Series(df.index.isin(train_ix), index=df.index, dtype=bool)
    test_mask = pd.Series(df.index.isin(test_ix), index=df.index, dtype=bool)
    return train_mask, test_mask


def resample_1m_with_whale(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """
    Aggregate 1m rows to `bar_minutes` bars: OHLCV + whale columns, then `ret_1` on resampled close.
    Index remains millisecond timestamps (bar end labels).
    """
    if bar_minutes <= 1:
        return df.copy()
    d = df.sort_index()
    dt_ix = pd.to_datetime(d.index, unit="ms", utc=True)
    d = d.copy()
    d.index = dt_ix
    rule = f"{bar_minutes}min"
    agg: dict[str, str] = {}
    for c in ("open", "high", "low", "close", "volume"):
        if c not in d.columns:
            continue
        if c == "open":
            agg[c] = "first"
        elif c == "high":
            agg[c] = "max"
        elif c == "low":
            agg[c] = "min"
        elif c == "close":
            agg[c] = "last"
        else:
            agg[c] = "sum"
    for c in d.columns:
        if c in agg or c == "ret_1":
            continue
        if c == "time_utc":
            agg[c] = "last"
            continue
        cl = c.lower()
        if "lev" in cl or "wavg" in cl:
            agg[c] = "mean"
        elif any(k in cl for k in ("count", "_n", "rows", "wallet_n", "user_n")):
            agg[c] = "sum"
        else:
            agg[c] = "sum"
    sub = d[list(agg.keys())]
    out = sub.resample(rule, label="right", closed="right").agg(agg)
    out = out.dropna(subset=["close"])
    out["time_utc"] = out.index
    ms = (pd.DatetimeIndex(out.index).astype(np.int64) // 10**6).astype(np.int64)
    out.index = ms
    out.index.name = "time_ms"
    out = apply_formulas(out, {"ret_1": "ts_returns(close, 1)"})
    return numeric_to_float32(out, exclude=("time_utc",))


def list_whale_numeric_columns(
    df: pd.DataFrame,
    extra_numeric: Sequence[str] = (),
) -> list[str]:
    cols: list[str] = []
    ex = set(extra_numeric)
    for c in df.columns:
        if c in ("ret_1", "time_utc"):
            continue
        if c in ex:
            cols.append(c)
            continue
        if not str(c).startswith("whale_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def trim_whale_window(df: pd.DataFrame, whale_base_cols: list[str]) -> pd.DataFrame:
    if not whale_base_cols:
        raise ValueError("no whale columns")
    _wm = df[whale_base_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    _has = _wm.notna().any(axis=1)
    if not _has.any():
        raise ValueError("all whale columns empty")
    w_ix = _has[_has].index
    return df.loc[int(w_ix[0]) : int(w_ix[-1])].copy()


def fill_whale_missing(
    df: pd.DataFrame,
    whale_base_cols: list[str],
    policy: str = "zero",
) -> None:
    if policy == "zero":
        for c in whale_base_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    elif policy == "ffill_then_zero":
        for c in whale_base_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.ffill().fillna(0.0)
    else:
        raise ValueError(policy)


def build_whale_alpha_library(base_cols: list[str]) -> dict[str, dict[str, str]]:
    lib: dict[str, dict[str, str]] = {
        "whale_flow": {
            "alert_signed_z60": "ts_zscore(whale_alert_notional_signed, 60)",
            "alert_signed_mean30_z60": "ts_zscore(ts_mean(whale_alert_notional_signed, 30), 60)",
            "alert_abs_intensity_z60": "ts_zscore(whale_alert_notional_abs, 60)",
            "open_pressure": "ts_zscore(safe_div(whale_alert_open_count, whale_alert_count + 1.0), 240)",
            "close_pressure": "ts_zscore(safe_div(whale_alert_close_count, whale_alert_count + 1.0), 240)",
        },
        "whale_position_book": {
            "pos_signed_z120": "ts_zscore(whale_pos_notional_signed, 120)",
            "pos_signed_slow_z240": "ts_zscore(ts_mean(whale_pos_notional_signed, 30), 240)",
            "pos_net_bias": "ts_zscore(safe_div(whale_pos_notional_signed, whale_pos_notional_abs + 1e-12), 240)",
            "lev_regime_z240": "ts_zscore(whale_pos_lev_wavg, 240)",
            "pos_pnl_z120": "ts_zscore(whale_pos_pnl_sum, 120)",
            "active_wallet_chg_z120": "ts_zscore(ts_delta(whale_pos_active_wallet_n, 1), 120)",
        },
        "whale_attention": {
            "alert_count_z120": "ts_zscore(whale_alert_count, 120)",
            "user_n_z120": "ts_zscore(whale_alert_user_n, 120)",
        },
        # Price momentum (bar returns) × whale flow / position — CTA-style mom + microstructure.
        "whale_momentum": {
            "mom_ret60_z": "ts_zscore(ts_returns(close, 60), 120)",
            "mom_ret30_z": "ts_zscore(ts_returns(close, 30), 120)",
            "mom_vol_adj_ret60": "safe_div(ts_returns(close, 60), ts_std(ret_1, 120) + 1e-12)",
            "mom_flow_align_ret30": "ts_zscore(whale_alert_notional_signed, 60) * ts_zscore(ts_returns(close, 30), 120)",
            "mom_signed_flow_plus_ret30": "ts_zscore(whale_alert_notional_signed, 120) + ts_zscore(ts_returns(close, 30), 120)",
            "mom_pos_signed_x_ret30": "ts_zscore(whale_pos_notional_signed, 120) * ts_zscore(ts_returns(close, 30), 120)",
        },
        # Mean-revert when alerts are hot and price sits at a rolling range extreme (peak/trough).
        "whale_reverting": {
            "rev_hot_alert_x_price_rank": "-ts_zscore(whale_alert_count, 120) * (ts_rank(close, 60) - 0.5)",
            "rev_hot_abs_flow_x_price_rank": "-ts_zscore(whale_alert_notional_abs, 120) * (ts_rank(close, 96) - 0.5)",
            "rev_user_hot_x_price_rank": "-ts_zscore(whale_alert_user_n, 120) * (ts_rank(close, 72) - 0.5)",
            "rev_alert_z_x_range_mid": "-ts_zscore(whale_alert_count, 120) * ts_zscore(ts_rank(close, 48) - 0.5, 120)",
        },
        # Regime: short-horizon volatility of whale features + “active bar” share.
        "whale_volatility": {
            "vol_lev_stdev_z": "ts_zscore(ts_std(whale_pos_lev_wavg, 30), 120)",
            "vol_flow_abs_stdev_z": "ts_zscore(ts_std(whale_alert_notional_abs, 30), 120)",
            "vol_open_count_stdev_z": "ts_zscore(ts_std(whale_alert_open_count, 30), 120)",
            "vol_signed_flow_stdev_z": "ts_zscore(ts_std(whale_alert_notional_signed, 30), 120)",
            "vol_pos_signed_stdev_z": "ts_zscore(ts_std(whale_pos_notional_signed, 30), 120)",
            "vol_active_minute_share_z": "ts_zscore(ts_mean(ifcond(whale_alert_count > 0, 1.0, 0.0), 120), 240)",
        },
    }
    scan: dict[str, str] = {}
    for c in base_cols:
        stem = "".join(ch if ch.isalnum() else "_" for ch in c)[:48]
        scan[f"raw_{stem}"] = c
        scan[f"dz60_{stem}"] = f"ts_zscore(ts_delta({c}, 1), 60)"
        scan[f"z120_{stem}"] = f"ts_zscore({c}, 120)"
        scan[f"mean30_z120_{stem}"] = f"ts_zscore(ts_mean({c}, 30), 120)"
    lib["whale_scan"] = scan
    return lib


def composite_in_sample_only(row: pd.Series) -> float:
    r = row.get("robust_score", np.nan)
    wf = row.get("wf_sharpe_min", np.nan)
    tr = row.get("train_sharpe", np.nan)
    r = float(r) if r == r else 0.0
    wf = float(wf) if wf == wf else 0.0
    tr = float(tr) if tr == tr else 0.0
    score = (
        COMPOSITE_W_TRAIN * np.clip(tr, -2.0, 8.0)
        + COMPOSITE_W_ROBUST * np.clip(r, -5.0, 10.0)
        + COMPOSITE_W_WF_MIN * np.clip(wf, -4.0, 6.0)
    )
    is_trigger_count = row.get("is_trigger_count", np.nan)
    is_trigger_count = float(is_trigger_count) if is_trigger_count == is_trigger_count else 0.0
    scarcity = np.clip(
        (float(TRIGGER_TARGET_IS) - is_trigger_count) / float(TRIGGER_TARGET_IS),
        0.0,
        1.0,
    )
    score -= COMPOSITE_W_TRIGGER_PENALTY * scarcity
    if wf < -0.8:
        score -= 0.12 * abs(wf)
    return float(score)


def composite_result_oriented(
    row: pd.Series,
    *,
    w_train: float,
    w_test: float,
    w_robust: float,
    w_wf: float,
    w_consistency: float,
    w_profitability: float,
    w_trigger_penalty: float,
    trigger_target_is: float,
    trigger_target_os: float,
    consistency_clip: float = 4.0,
) -> float:
    def _safe_num(v: Any) -> float:
        x = pd.to_numeric(v, errors="coerce")
        x = float(x) if x == x else 0.0
        return x

    train_sh = _safe_num(row.get("train_sharpe"))
    test_sh = _safe_num(row.get("test_sharpe"))
    robust = _safe_num(row.get("robust_score"))
    wf_min = _safe_num(row.get("wf_sharpe_min"))
    is_trig = _safe_num(row.get("is_trigger_count"))
    os_trig = _safe_num(row.get("os_trigger_count"))

    consistency = -abs(train_sh - test_sh)
    consistency = float(np.clip(consistency, -consistency_clip, 0.0))
    profitability = float(np.clip(min(train_sh, test_sh), -3.0, 5.0))

    scarcity_is = np.clip((float(trigger_target_is) - is_trig) / max(float(trigger_target_is), 1.0), 0.0, 1.0)
    scarcity_os = np.clip((float(trigger_target_os) - os_trig) / max(float(trigger_target_os), 1.0), 0.0, 1.0)
    trigger_penalty = float(scarcity_is + 0.7 * scarcity_os)

    score = (
        w_train * np.clip(train_sh, -3.0, 8.0)
        + w_test * np.clip(test_sh, -3.0, 8.0)
        + w_robust * np.clip(robust, -6.0, 10.0)
        + w_wf * np.clip(wf_min, -5.0, 6.0)
        + w_consistency * consistency
        + w_profitability * profitability
        - w_trigger_penalty * trigger_penalty
    )
    return float(score)


def _count_trigger_events(position: pd.Series, eps: float = 1e-12) -> int:
    pos = pd.to_numeric(position, errors="coerce").fillna(0.0).astype("float64")
    active = pos.abs() > float(eps)
    entry = active & (~active.shift(1, fill_value=False))
    return int(entry.sum())


def _traces_train_only(traces: dict) -> dict:
    out = {}
    for sig, tr in traces.items():
        if tr is None or tr.empty:
            continue
        if "is_train" in tr.columns:
            sub = tr.loc[tr["is_train"]].drop(columns=["is_train", "is_test"], errors="ignore")
        else:
            sub = tr
        out[sig] = sub
    return out


def _build_master_candidates(
    df_1m: pd.DataFrame,
    signal_library: dict[str, dict[str, str]],
    *,
    bar_minutes_list: Sequence[int],
    exec_profiles: Sequence[dict],
    cfg_base: BacktestConfig,
    dedupe_best_exec_per_signal_freq: bool = True,
) -> pd.DataFrame:
    all_rows: list[dict] = []
    for bm in bar_minutes_list:
        freq = freq_str_for_bar_minutes(bm)
        cfg = BacktestConfig(
            freq=freq,
            fee_bps=cfg_base.fee_bps,
            slippage_bps=cfg_base.slippage_bps,
            signal_shift=cfg_base.signal_shift,
            signal_clip=cfg_base.signal_clip,
            max_abs_position=cfg_base.max_abs_position,
            position_step=cfg_base.position_step,
        )
        dfr = df_1m.copy() if bm <= 1 else resample_1m_with_whale(df_1m, bm)
        sm, cols = build_signal_features(dfr.copy(), signal_library)
        sm, cols = add_inverted_signal_columns(sm, cols)
        train_mask, test_mask = train_test_masks_whale_last_pct(sm)
        for prof in exec_profiles:
            z_w = z_window_scaled_from_1m_bars(prof["z1m"], bm)
            rep, traces = evaluate_signal_library_train_test(
                df=sm,
                signal_cols=cols,
                ret_col="ret_1",
                train_mask=train_mask,
                test_mask=test_mask,
                cfg=cfg,
                z_window=z_w,
                position_mode=prof["mode"],
                deadband=prof["deadband"],
                smooth_span=prof["smooth"],
            )
            tr_train = _traces_train_only(traces)
            rep = add_robustness_columns(rep, tr_train, cfg=cfg, n_splits=6)
            for sig in rep.index:
                pos = (
                    signal_to_position(
                        sm[sig],
                        clip=cfg.signal_clip,
                        z_window=z_w,
                        mode=str(prof["mode"]),
                        deadband=float(prof["deadband"]),
                        smooth_span=int(prof["smooth"]),
                        max_abs_position=cfg.max_abs_position,
                        position_step=cfg.position_step,
                    )
                    .shift(cfg.signal_shift)
                    .fillna(0.0)
                )
                is_trigger_count = _count_trigger_events(pos.loc[train_mask])
                os_trigger_count = _count_trigger_events(pos.loc[test_mask])
                row = rep.loc[sig].to_dict()
                row["signal"] = sig
                row["bar_minutes"] = bm
                row["freq"] = freq
                row["z_window"] = z_w
                row["z_window_1m_equiv"] = prof["z1m"]
                row["deadband"] = prof["deadband"]
                row["smooth_span"] = prof["smooth"]
                row["position_mode"] = prof["mode"]
                row["is_trigger_count"] = is_trigger_count
                row["os_trigger_count"] = os_trigger_count
                row["trigger_count_total"] = is_trigger_count + os_trigger_count
                all_rows.append(row)
    master = pd.DataFrame(all_rows)
    if master.empty:
        return master
    mo = pd.to_numeric(master.get("train_obs"), errors="coerce")
    teo = pd.to_numeric(master.get("test_obs"), errors="coerce")
    master = master.loc[(mo.fillna(0) >= 120) & (teo.fillna(0) >= 120)].copy()
    trg_is = pd.to_numeric(master.get("is_trigger_count"), errors="coerce")
    master = master.loc[trg_is.fillna(0) >= float(MIN_IS_TRIGGER_FOR_SWEEP)].copy()
    trg_os = pd.to_numeric(master.get("os_trigger_count"), errors="coerce")
    master = master.loc[trg_os.fillna(0) >= float(MIN_OS_TRIGGER_FOR_SWEEP)].copy()
    if master.empty:
        return master
    master["train_test_gap_abs"] = (
        pd.to_numeric(master.get("train_sharpe"), errors="coerce")
        - pd.to_numeric(master.get("test_sharpe"), errors="coerce")
    ).abs()
    if dedupe_best_exec_per_signal_freq:
        master["composite_is"] = master.apply(composite_in_sample_only, axis=1)
        master = master.sort_values(
            by=["composite_is", "train_sharpe", "robust_score", "wf_sharpe_min"],
            ascending=False,
        )
        master = master.drop_duplicates(subset=["signal", "freq"], keep="first")
    return master


def _default_result_oriented_param_grid() -> list[dict[str, float]]:
    weight_grid = [
        # (train, test, robust, wf, consistency, profitability)
        (0.20, 0.45, 0.15, 0.05, 0.10, 0.05),
        (0.18, 0.48, 0.12, 0.05, 0.10, 0.07),
        (0.22, 0.40, 0.18, 0.05, 0.10, 0.05),
        (0.15, 0.50, 0.12, 0.06, 0.10, 0.07),
        (0.20, 0.42, 0.15, 0.08, 0.10, 0.05),
        (0.15, 0.45, 0.15, 0.05, 0.15, 0.05),
    ]
    trigger_penalty_grid = [0.04, 0.06, 0.08, 0.10]
    trigger_target_is_grid = [16.0, 24.0, 32.0]
    trigger_target_os_grid = [4.0, 8.0, 12.0]
    params: list[dict[str, float]] = []
    for wset, tp, tis, tos in product(
        weight_grid,
        trigger_penalty_grid,
        trigger_target_is_grid,
        trigger_target_os_grid,
    ):
        params.append(
            {
                "w_train": float(wset[0]),
                "w_test": float(wset[1]),
                "w_robust": float(wset[2]),
                "w_wf": float(wset[3]),
                "w_consistency": float(wset[4]),
                "w_profitability": float(wset[5]),
                "w_trigger_penalty": float(tp),
                "trigger_target_is": float(tis),
                "trigger_target_os": float(tos),
            }
        )
    return params


def _rank_with_result_oriented_composite(
    master: pd.DataFrame,
    *,
    params: dict[str, float],
    top_k_opt: int,
    dedupe_best_exec_per_signal_freq: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = master.copy()
    df["composite_result"] = df.apply(
        lambda row: composite_result_oriented(
            row,
            w_train=float(params["w_train"]),
            w_test=float(params["w_test"]),
            w_robust=float(params["w_robust"]),
            w_wf=float(params["w_wf"]),
            w_consistency=float(params["w_consistency"]),
            w_profitability=float(params["w_profitability"]),
            w_trigger_penalty=float(params["w_trigger_penalty"]),
            trigger_target_is=float(params["trigger_target_is"]),
            trigger_target_os=float(params["trigger_target_os"]),
        ),
        axis=1,
    )
    df = df.sort_values(
        by=["composite_result", "test_sharpe", "train_sharpe", "robust_score"],
        ascending=False,
    )
    if dedupe_best_exec_per_signal_freq:
        df = df.drop_duplicates(subset=["signal", "freq"], keep="first")
    top = df.head(max(1, int(top_k_opt))).copy()
    ts = pd.to_numeric(top.get("test_sharpe"), errors="coerce")
    tr = pd.to_numeric(top.get("train_sharpe"), errors="coerce")
    topk_weighted_sharpe = float(0.3 * ts.mean() + 0.7 * tr.mean()) if len(ts) else float("nan")
    objective = {
        "topk_weighted_sharpe_03test_07train": topk_weighted_sharpe,
        "topk_avg_test_sharpe": float(ts.mean()) if len(ts) else float("nan"),
        "topk_avg_train_sharpe": float(tr.mean()) if len(tr) else float("nan"),
        "topk_median_test_sharpe": float(ts.median()) if len(ts) else float("nan"),
        "topk_profitable_ratio_test": float((ts > 0).mean()) if len(ts) else float("nan"),
        "topk_avg_train_test_gap_abs": float((tr - ts).abs().mean()) if len(ts) else float("nan"),
    }
    return df, objective


def run_whale_signal_sweep(
    df_1m: pd.DataFrame,
    signal_library: dict[str, dict[str, str]],
    *,
    bar_minutes_list: Sequence[int] | None = None,
    exec_profiles: Sequence[dict] | None = None,
    cfg_base: BacktestConfig | None = None,
    dedupe_best_exec_per_signal_freq: bool = True,
    ranking_mode: str = "result_oriented",
    top_k_opt: int = TOPK_OPT_DEFAULT,
    tuning_param_grid: Sequence[dict[str, float]] | None = None,
) -> pd.DataFrame:
    bar_minutes_list = list(bar_minutes_list or BAR_MINUTES_LIST_DEFAULT)
    exec_profiles = list(exec_profiles or EXEC_PROFILES_DEFAULT)
    cfg_base = cfg_base or CFG_BASE_DEFAULT
    master = _build_master_candidates(
        df_1m,
        signal_library,
        bar_minutes_list=bar_minutes_list,
        exec_profiles=exec_profiles,
        cfg_base=cfg_base,
        dedupe_best_exec_per_signal_freq=dedupe_best_exec_per_signal_freq,
    )
    if master.empty:
        return master
    if ranking_mode == "in_sample":
        df = master.copy()
        df["composite_is"] = df.apply(composite_in_sample_only, axis=1)
        df = df.sort_values(
            by=["composite_is", "train_sharpe", "robust_score", "wf_sharpe_min"],
            ascending=False,
        )
        if dedupe_best_exec_per_signal_freq:
            df = df.drop_duplicates(subset=["signal", "freq"], keep="first")
        return df
    if ranking_mode != "result_oriented":
        raise ValueError("ranking_mode must be 'in_sample' or 'result_oriented'")

    grid = list(tuning_param_grid or _default_result_oriented_param_grid())
    if not grid:
        raise ValueError("tuning_param_grid must not be empty")
    best_df: pd.DataFrame | None = None
    best_params: dict[str, float] | None = None
    best_obj: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for params in grid:
        ranked_df, obj = _rank_with_result_oriented_composite(
            master,
            params=params,
            top_k_opt=top_k_opt,
            dedupe_best_exec_per_signal_freq=dedupe_best_exec_per_signal_freq,
        )
        key = (
            float(obj.get("topk_weighted_sharpe_03test_07train", -np.inf)),
            float(obj.get("topk_profitable_ratio_test", -np.inf)),
            -float(obj.get("topk_avg_train_test_gap_abs", np.inf)),
            float(obj.get("topk_median_test_sharpe", -np.inf)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_df = ranked_df
            best_params = dict(params)
            best_obj = dict(obj)
    if best_df is None:
        return master
    best_df = best_df.copy()
    best_df.attrs["ranking_mode"] = "result_oriented"
    best_df.attrs["top_k_opt"] = int(top_k_opt)
    best_df.attrs["best_params"] = best_params or {}
    best_df.attrs["tuning_objective"] = best_obj or {}
    return best_df


def _ensure_signal_on_frame(sm: pd.DataFrame, sig: str) -> None:
    if sig in sm.columns:
        return
    if sig.endswith("__inv"):
        base = sig[: -len("__inv")]
        if base in sm.columns:
            sm[sig] = -sm[base]


def position_low_freq_for_row(
    row: pd.Series,
    df1: pd.DataFrame,
    signal_library: dict[str, dict[str, str]],
    cfg_base: BacktestConfig,
) -> pd.Series:
    sig = row["signal"]
    bm = int(row["bar_minutes"])
    freq = row["freq"]
    dfr = df1.copy() if bm <= 1 else resample_1m_with_whale(df1, bm)
    sm, _ = build_signal_features(dfr.copy(), signal_library)
    _ensure_signal_on_frame(sm, sig)
    cfg = BacktestConfig(
        freq=freq,
        fee_bps=cfg_base.fee_bps,
        slippage_bps=cfg_base.slippage_bps,
        signal_shift=cfg_base.signal_shift,
        signal_clip=cfg_base.signal_clip,
        max_abs_position=cfg_base.max_abs_position,
        position_step=cfg_base.position_step,
    )
    z_w = int(row["z_window"])
    return (
        signal_to_position(
            sm[sig],
            clip=cfg.signal_clip,
            z_window=z_w,
            mode=str(row["position_mode"]),
            deadband=float(row["deadband"]),
            smooth_span=int(row["smooth_span"]),
            max_abs_position=cfg.max_abs_position,
            position_step=cfg.position_step,
        )
        .shift(cfg.signal_shift)
        .fillna(0.0)
    )


def build_position_matrix_1m(
    results_top: pd.DataFrame,
    df1: pd.DataFrame,
    signal_library: dict[str, dict[str, str]],
    cfg_base: BacktestConfig,
) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for j, (_, row) in enumerate(results_top.iterrows()):
        label = f"{j:02d}_{str(row['signal'])[:48]}|{row['freq']}"
        pos_low = position_low_freq_for_row(row, df1, signal_library, cfg_base)
        cols[label] = upsample_position_to_1m(pos_low, df1.index)
    return pd.DataFrame(cols, index=df1.index, dtype="float64")


def raw_signal_low_freq_for_row(
    row: pd.Series,
    df1: pd.DataFrame,
    signal_library: dict[str, dict[str, str]],
) -> pd.Series:
    """
    Raw formula signal on its native bar index (before ``signal_to_position``).
    Same resampling path as ``position_low_freq_for_row``.
    """
    sig = str(row["signal"])
    bm = int(row["bar_minutes"])
    dfr = df1.copy() if bm <= 1 else resample_1m_with_whale(df1, bm)
    sm, _ = build_signal_features(dfr.copy(), signal_library)
    _ensure_signal_on_frame(sm, sig)
    return pd.to_numeric(sm[sig], errors="coerce").astype("float64")


def build_raw_signal_matrix_1m(
    results_top: pd.DataFrame,
    df1: pd.DataFrame,
    signal_library: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """
    Stack raw alpha columns on the 1m index: each sweep row is evaluated on its ``bar_minutes``
    grid, then forward-filled onto every 1m timestamp (same convention as ``upsample_position_to_1m``).
    Use for correlation across mixed-frequency signals on a common timeline.
    """
    cols: dict[str, pd.Series] = {}
    for j, (_, row) in enumerate(results_top.iterrows()):
        label = f"{j:02d}_{str(row['signal'])[:48]}|{row['freq']}"
        s_low = raw_signal_low_freq_for_row(row, df1, signal_library)
        cols[label] = upsample_position_to_1m(s_low, df1.index)
    return pd.DataFrame(cols, index=df1.index, dtype="float64")


def vol_target_leverage(pos_ew: pd.Series, ret_1m: pd.Series, ann_vol: float) -> pd.Series:
    r = (pos_ew * ret_1m).astype("float64")
    sig_roll = r.rolling(VOL_ROLL_WIN, min_periods=_MINP).std().clip(lower=1e-12)
    target_bar = float(ann_vol) / float(np.sqrt(BARS_PER_YEAR_1M))
    lev = (target_bar / sig_roll).clip(0.15, 6.0)
    return lev.ffill().fillna(1.0)


def first_oos_time_utc(df1: pd.DataFrame) -> pd.Timestamp:
    _, te = train_test_masks_whale_last_pct(df1)
    first_ix = te[te].index[0]
    return pd.to_datetime(df1.loc[first_ix, "time_utc"], utc=True)


def plot_whale_ensemble_figures(
    df1: pd.DataFrame,
    results: pd.DataFrame,
    signal_library: dict[str, dict[str, str]],
    cfg_base: BacktestConfig,
    *,
    pool_sizes: Sequence[int] = (5, 10, 20, 30),
    vol_targets: Sequence[float] = (0.10, 0.20, 0.40),
) -> None:
    """
    Two figures (English): (1) equal weight, (2) vol target with one subplot per annual vol target.
    Four curves per panel where possible (Top-5/10/20/30 by composite_is). IS|OOS vertical line.
    """
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in divide",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in multiply",
    )

    pos_mats: dict[int, pd.DataFrame] = {}
    for k in pool_sizes:
        top = results.head(min(k, len(results)))
        if top.empty:
            continue
        pos_mats[int(k)] = build_position_matrix_1m(top, df1, signal_library, cfg_base)

    ret_1m = df1["ret_1"].astype("float64").reindex(df1.index)
    cfg_1m = BacktestConfig(
        freq="1m",
        fee_bps=cfg_base.fee_bps,
        slippage_bps=cfg_base.slippage_bps,
        signal_shift=0,
        signal_clip=cfg_base.signal_clip,
        max_abs_position=cfg_base.max_abs_position,
        position_step=cfg_base.position_step,
    )

    split_t = first_oos_time_utc(df1)

    def _vline(ax: plt.Axes, with_label: bool) -> None:
        ax.axvline(
            split_t,
            color="0.2",
            ls="--",
            lw=1.15,
            alpha=0.95,
            label="IS | OOS" if with_label else None,
            zorder=5,
        )

    pool_colors = {5: "C0", 10: "C1", 20: "C2", 30: "C3"}

    # --- Figure 1: Equal weight ---
    fig1, ax1 = plt.subplots(figsize=(14, 5.5))
    for k in pool_sizes:
        if k not in pos_mats:
            continue
        pm = pos_mats[k]
        r = ret_1m.reindex(pm.index)
        pos_ew = pm.mean(axis=1).clip(-1.0, 1.0)
        net = net_return_with_costs(pos_ew, r, cfg_1m)
        eq = equity_curve_from_1m_net(net)
        tdt = pd.to_datetime(df1.loc[eq.index, "time_utc"], utc=True)
        ax1.plot(tdt, eq.values, color=pool_colors.get(k, "C0"), lw=1.2, label=f"Top-{k}")
    _vline(ax1, True)
    ax1.set_title("Equal weight — IS-ranked pools (costs included)")
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Equity (start = 1)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Figure 2: Vol target (3 subplots) ---
    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (axv, ann) in enumerate(zip(axes3, vol_targets)):
        for k in pool_sizes:
            if k not in pos_mats:
                continue
            pm = pos_mats[k]
            r = ret_1m.reindex(pm.index)
            pos_ew = pm.mean(axis=1).clip(-1.0, 1.0)
            lev = vol_target_leverage(pos_ew, r, ann)
            pos_vt = (pos_ew * lev).clip(-1.0, 1.0)
            net = net_return_with_costs(pos_vt, r, cfg_1m)
            eq = equity_curve_from_1m_net(net)
            tdt = pd.to_datetime(df1.loc[eq.index, "time_utc"], utc=True)
            axv.plot(tdt, eq.values, color=pool_colors.get(k, "C0"), lw=1.1, label=f"Top-{k}")
        _vline(axv, i == 0)
        axv.set_ylabel("Equity")
        axv.set_title(f"Vol target ~{ann:.0%} ann (scaled EW), all pools")
        axv.legend(loc="best", fontsize=8, ncol=2)
        axv.grid(True, alpha=0.3)
    axes3[-1].set_xlabel("Time (UTC)")
    plt.tight_layout()
    plt.show()


CFG_BASE = CFG_BASE_DEFAULT
BAR_MINUTES_LIST = BAR_MINUTES_LIST_DEFAULT
EXEC_PROFILES = EXEC_PROFILES_DEFAULT

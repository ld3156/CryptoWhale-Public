from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.cta_signal_lab import (
    BacktestConfig,
    add_robustness_columns,
    build_signal_features,
    evaluate_signal_library,
)
from tool.newmath import apply_formulas



def load_base_table(freq: str = "15m") -> pd.DataFrame:
    p = ROOT / "data" / "cleaned" / f"cleaned_{freq}.csv"
    df = pd.read_csv(p)
    df = df.set_index("time_ms")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def add_aliases(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "close": "futures_price_history_btcusdt_binance__close",
        "open": "futures_price_history_btcusdt_binance__open",
        "high": "futures_price_history_btcusdt_binance__high",
        "low": "futures_price_history_btcusdt_binance__low",
        "rsi": "futures_rsi_history_btcusdt_binance__rsi_value",
        "ema": "futures_ema_history_btcusdt_binance__ema_value",
        "ma": "futures_ma_history_btcusdt_binance__ma_value",
        "macd": "futures_macd_history_btcusdt_binance__macd_value",
        "macd_signal": "futures_macd_history_btcusdt_binance__signal",
        "macd_hist": "futures_macd_history_btcusdt_binance__histogram",
        "boll_ub": "futures_boll_history_btcusdt_binance__ub_value",
        "boll_mb": "futures_boll_history_btcusdt_binance__mb_value",
        "boll_lb": "futures_boll_history_btcusdt_binance__lb_value",
        "volume": "futures_price_history_btcusdt_binance__volume_usd",
    }
    out = df.copy()
    for short, raw in alias_map.items():
        if raw in out.columns:
            out[short] = pd.to_numeric(out[raw], errors="coerce")
    return out


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    formulas = {
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
        "rsi_z": "ts_zscore(rsi, 60)",
        "rsi_chg": "rsi - last(rsi)",
        "candle_box": "safe_div((close - open), last(close))",
        "ma_slope_30": "ma - ts_delay(ma, 30)",
        "ma_curv_15": "ma - 2*ts_delay(ma, 15) + ts_delay(ma, 30)",
        "boll_pos": "safe_div(close - boll_lb, boll_ub - boll_lb)",
        "band_width": "safe_div((boll_ub - boll_lb), boll_mb)",
        "tr": "safe_div(ts_max(high, 2) - ts_min(low, 2), close)",
        "atr_20": "ts_mean(tr, 20)",
    }
    return apply_formulas(df, formulas)


def build_candidate_library() -> Dict[str, Dict[str, str]]:
    lib: Dict[str, Dict[str, str]] = {
        "trend": {},
        "momentum": {},
        "mean_revert": {},
        "volatility": {},
        "breakout": {},
        "adaptive": {},
    }

    # Trend / moving-average family
    for fast, slow in [(8, 48), (12, 72), (20, 120), (30, 180), (60, 360), (120, 720), (240, 1440)]:
        lib["trend"][f"ema_spread_{fast}_{slow}"] = (
            f"safe_div(ts_ema(close, {fast}) - ts_ema(close, {slow}), ts_std(close, {slow}))"
        )
        lib["trend"][f"ma_spread_{fast}_{slow}"] = (
            f"safe_div(ts_mean(close, {fast}) - ts_mean(close, {slow}), ts_std(close, {slow}))"
        )
        lib["trend"][f"price_ma_dist_{slow}"] = f"safe_div(close - ts_mean(close, {slow}), ts_std(close, {slow}))"

    # Momentum horizons (TSMOM style)
    for short, long in [(5, 30), (15, 120), (30, 240), (60, 360), (120, 720), (240, 1440), (720, 2880)]:
        lib["momentum"][f"mom_diff_{short}_{long}"] = f"ts_returns(close, {short}) - ts_returns(close, {long})"
        lib["momentum"][f"mom_beta_{short}_{long}"] = f"ts_beta(ts_returns(close, {short}), ts_returns(close, {long}), {long})"
        lib["momentum"][f"mom_sign_{long}"] = f"ifcond(ts_returns(close, {long}) > 0, 1, -1)"

    for w in [24, 48, 96, 168, 240]:
        lib["momentum"][f"macd_impulse_z_{w}"] = f"ts_zscore(macd - macd_signal, {w})"
        lib["momentum"][f"rsi_trend_z_{w}"] = f"ts_zscore(rsi - 50, {w})"

    # Mean reversion
    for w in [48, 96, 168, 240]:
        lib["mean_revert"][f"rsi_revert_{w}"] = f"-ts_zscore(rsi, {w})"
        lib["mean_revert"][f"price_revert_{w}"] = f"-ts_zscore(close, {w})"
        lib["mean_revert"][f"boll_revert_{w}"] = f"-ts_zscore(boll_pos - 0.5, {w})"
        lib["mean_revert"][f"candle_revert_{w}"] = f"-ts_zscore(candle_box, {w})"

    # Volatility / risk
    for w in [24, 48, 96, 168, 240]:
        lib["volatility"][f"vol_break_{w}"] = f"ts_zscore(ts_std(ret_1, {w}), {w * 3})"
        lib["volatility"][f"vol_compress_{w}"] = f"-ts_zscore(band_width, {w * 3})"
        lib["volatility"][f"tail_risk_{w}"] = f"ts_kurt(ret_1, {w})"
        lib["volatility"][f"skew_reversal_{w}"] = f"-ts_skew(ret_1, {w})"

    # Breakout / Donchian-like normalized by ATR
    for w in [20, 40, 60, 120, 180, 240, 480, 720, 960, 1440]:
        lib["breakout"][f"donchian_up_{w}"] = f"safe_div(close - ts_max(high, {w}), ts_mean(atr_20, {w}))"
        lib["breakout"][f"donchian_dn_{w}"] = f"safe_div(close - ts_min(low, {w}), ts_mean(atr_20, {w}))"
        lib["breakout"][f"quantile_break_{w}"] = f"safe_div(close - ts_quantile(close, {w}, 0.8), ts_std(close, {w}))"

    # Adaptive / regime-gated variants (CTA-style risk filter)
    lib["adaptive"]["trend_regime_gate_1"] = (
        "ifcond(ts_std(ret_1, 96) < ts_std(ret_1, 480), "
        "safe_div(ts_ema(close, 120) - ts_ema(close, 720), ts_std(close, 720)), 0)"
    )
    lib["adaptive"]["trend_regime_gate_2"] = (
        "ifcond(ts_std(ret_1, 48) < ts_std(ret_1, 240), "
        "safe_div(ts_mean(close, 120) - ts_mean(close, 720), ts_std(close, 720)), 0)"
    )
    lib["adaptive"]["breakout_regime_gate_1"] = (
        "ifcond(ts_std(ret_1, 96) < ts_std(ret_1, 480), "
        "safe_div(close - ts_min(low, 720), ts_mean(atr_20, 720)), 0)"
    )
    lib["adaptive"]["mom_vol_target_1"] = (
        "safe_div(ifcond(ts_returns(close, 720) > 0, 1, -1), ts_std(ret_1, 240))"
    )
    lib["adaptive"]["mom_vol_target_2"] = (
        "safe_div(ts_returns(close, 360), ts_std(ret_1, 240))"
    )

    return lib


def dedup_top_signals(report: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    out_rows = []
    used_keys = set()
    for sig, row in report.iterrows():
        base = sig.split("__")[-1]
        key = "_".join(base.split("_")[:2])
        if key in used_keys:
            continue
        out_rows.append((sig, row))
        used_keys.add(key)
        if len(out_rows) >= top_n:
            break
    if not out_rows:
        return report.head(0)
    idx = [x[0] for x in out_rows]
    return report.loc[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="CTA signal research runner")
    parser.add_argument("--freq", default="15m", choices=["1m", "5m", "15m", "1h"])
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    args = parser.parse_args()

    freq = args.freq
    df = load_base_table(freq=freq)
    df = add_aliases(df)
    df = add_core_features(df)

    signal_library = build_candidate_library()
    signal_matrix, signal_cols = build_signal_features(df.copy(), signal_library)
    # Add mirrored direction for each signal, then pick direction by performance.
    inv_cols = {f"{s}__inv": -signal_matrix[s] for s in list(signal_cols)}
    signal_matrix = pd.concat([signal_matrix, pd.DataFrame(inv_cols, index=signal_matrix.index)], axis=1)
    signal_cols = signal_cols + list(inv_cols.keys())
    print(f"[Info] candidate signals: {len(signal_cols)}")

    cfg = BacktestConfig(
        freq=freq,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        signal_shift=1,
        signal_clip=2.5,
    )

    exec_grid: List[Tuple[str, int, str, float, int]] = [
        ("ret_1", 240, "tanh", 0.00, 0),
        ("ret_1", 360, "tanh", 0.15, 4),
        ("ret_1", 480, "tanh", 0.20, 6),
        ("ret_1", 720, "tanh", 0.30, 12),
        ("ret_1", 960, "tanh", 0.45, 18),
        ("ret_1", 1440, "tanh", 0.60, 24),
        ("ret_1", 720, "sign", 0.45, 12),
        ("ret_1", 1440, "sign", 0.60, 24),
    ]

    all_reports = []
    for ret_col, z_window, mode, deadband, smooth_span in exec_grid:
        report, traces = evaluate_signal_library(
            df=signal_matrix,
            signal_cols=signal_cols,
            ret_col=ret_col,
            cfg=cfg,
            z_window=z_window,
            position_mode=mode,
            deadband=deadband,
            smooth_span=smooth_span,
        )
        report = add_robustness_columns(report, traces, cfg=cfg, n_splits=6)
        report["ret_col"] = ret_col
        report["z_window"] = z_window
        report["position_mode"] = mode
        report["deadband"] = deadband
        report["smooth_span"] = smooth_span
        all_reports.append(report.reset_index())

    merged = pd.concat(all_reports, axis=0, ignore_index=True)
    merged = merged.sort_values(by=["robust_score", "sharpe"], ascending=False)
    # keep best execution setup for each signal
    merged = merged.drop_duplicates(subset=["signal"], keep="first")
    report = merged.set_index("signal").sort_values(by=["robust_score", "sharpe"], ascending=False)

    top20 = dedup_top_signals(report, top_n=20)
    out_csv = ROOT / "data" / f"signal_research_top20_{freq}.csv"
    top20.to_csv(out_csv)
    print(f"[Info] top20 saved: {out_csv}")
    print(top20[["ann_ret", "ann_vol", "sharpe", "calmar", "wf_sharpe_min", "wf_sharpe_std", "robust_score"]])


if __name__ == "__main__":
    main()

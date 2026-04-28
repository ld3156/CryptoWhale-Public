"""
Sweep multi-frequency CTA signals (1m → resampled bars), rank by robust_score + OOS stability.

Run from repo root:
  python tool/run_cta_signal_sweep.py

Outputs:
  data/cta_signal_catalog_top30.csv
  tool/cta_signal_catalog.py   (snippet with TOP30 rows for notebooks)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.core_cta_baseline import train_test_masks_from_time_utc
from tool.core_cta_baseline import build_core_signal_library
from tool.cta_multi_freq_lab import (
    add_inverted_signal_columns,
    freq_str_for_bar_minutes,
    load_1m_baseline,
    prepare_cta_feature_frame,
    resample_ohlcv_1m,
    z_window_scaled_from_1m_bars,
)
from tool.cta_signal_lab import (
    BacktestConfig,
    add_robustness_columns,
    build_signal_features,
    evaluate_signal_library_train_test,
)
from tool.newmath import numeric_to_float32


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


def composite_rank_row(row: pd.Series) -> float:
    """
    Emphasize OOS Sharpe + train robustness (avoid train-only overfit).
    """
    r = row.get("robust_score", np.nan)
    te = row.get("test_sharpe", np.nan)
    wf = row.get("wf_sharpe_min", np.nan)
    tr = row.get("train_sharpe", np.nan)
    r = float(r) if r == r else 0.0
    te = float(te) if te == te else -99.0
    wf = float(wf) if wf == wf else -99.0
    tr = float(tr) if tr == tr else 0.0
    # Train/test sanity (caller filters tiny samples)
    score = 0.38 * r + 0.32 * np.clip(te, -4.0, 6.0) + 0.18 * np.clip(wf, -4.0, 6.0) + 0.12 * np.clip(tr, -2.0, 8.0)
    if te < -0.15:
        score -= 0.55 * abs(te)
    if wf < -0.8:
        score -= 0.25 * abs(wf)
    return float(score)


def main() -> None:
    print("[CTA sweep] Loading 1m baseline...")
    df1 = load_1m_baseline(ROOT)
    df1 = numeric_to_float32(df1, exclude=("time_utc",))

    signal_library = build_core_signal_library()
    print("[CTA sweep] Unified library groups:", {k: len(v) for k, v in signal_library.items()})

    bar_minutes_list = [5, 15, 30, 60]
    # z_window in **1m-bar units** (scaled per freq); deadband in robust-z space
    exec_profiles = [
        {"z1m": 360, "deadband": 0.28, "smooth": 6, "mode": "tanh"},
        {"z1m": 480, "deadband": 0.35, "smooth": 8, "mode": "tanh"},
        {"z1m": 720, "deadband": 0.42, "smooth": 10, "mode": "tanh"},
        {"z1m": 480, "deadband": 0.35, "smooth": 8, "mode": "sign"},
    ]

    cfg_base = BacktestConfig(
        fee_bps=3.0,
        slippage_bps=1.0,
        signal_shift=1,
        signal_clip=2.5,
        max_abs_position=1.0,
        position_step=0.1,
    )

    all_rows: list[dict] = []

    for bm in bar_minutes_list:
        freq = freq_str_for_bar_minutes(bm)
        print(f"[CTA sweep] --- bar={bm}m ({freq}) ---")
        if bm <= 1:
            dfr = df1
        else:
            dfr = resample_ohlcv_1m(df1, bm)
        dfr = prepare_cta_feature_frame(dfr)
        sm, cols = build_signal_features(dfr.copy(), signal_library)
        sm, cols = add_inverted_signal_columns(sm, cols)
        train_mask, test_mask = train_test_masks_from_time_utc(sm)

        for prof in exec_profiles:
            z_w = z_window_scaled_from_1m_bars(prof["z1m"], bm)
            cfg = BacktestConfig(
                freq=freq,
                fee_bps=cfg_base.fee_bps,
                slippage_bps=cfg_base.slippage_bps,
                signal_shift=cfg_base.signal_shift,
                signal_clip=cfg_base.signal_clip,
                max_abs_position=cfg_base.max_abs_position,
                position_step=cfg_base.position_step,
            )
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
                row = rep.loc[sig].to_dict()
                row["signal"] = sig
                row["bar_minutes"] = bm
                row["freq"] = freq
                row["z_window"] = z_w
                row["z_window_1m_equiv"] = prof["z1m"]
                row["deadband"] = prof["deadband"]
                row["smooth_span"] = prof["smooth"]
                row["position_mode"] = prof["mode"]
                all_rows.append(row)

    master = pd.DataFrame(all_rows)
    if master.empty:
        print("[CTA sweep] No rows; abort.")
        return

    # Drop unusable slices (no train or tiny test → misleading metrics)
    mo = pd.to_numeric(master.get("train_obs"), errors="coerce")
    teo = pd.to_numeric(master.get("test_obs"), errors="coerce")
    master = master.loc[(mo.fillna(0) >= 120) & (teo.fillna(0) >= 120)].copy()
    master["composite"] = master.apply(composite_rank_row, axis=1)

    master = master.sort_values(by=["composite", "test_sharpe", "robust_score"], ascending=False)
    # Same named signal at same frequency: keep best execution profile
    master = master.drop_duplicates(subset=["signal", "freq"], keep="first")
    ts = pd.to_numeric(master["test_sharpe"], errors="coerce")
    trn = pd.to_numeric(master["train_sharpe"], errors="coerce")
    # Prefer positive train + positive OOS (fee-aware CTA); relax if pool is thin
    prefer = master.loc[(trn > 0.05) & (ts > 0.08)].copy()
    if len(prefer) < 22:
        prefer = master.loc[(trn > -0.25) & (ts > 0.05)].copy()
    if len(prefer) < 12:
        prefer = master.loc[ts > 0.02].copy()
    prefer = prefer.sort_values(
        by=["test_sharpe", "robust_score", "train_sharpe", "wf_sharpe_min"],
        ascending=False,
    )
    top = prefer.head(30).copy()

    out_csv = ROOT / "data" / "cta_signal_catalog_top30.csv"
    top.to_csv(out_csv, index=False)
    print(f"[CTA sweep] Wrote {len(top)} rows -> {out_csv}")

    # Python snippet for offline import
    py_out = ROOT / "tool" / "cta_signal_catalog.py"
    preview_cols = [c for c in ("signal", "freq", "train_sharpe", "test_sharpe", "robust_score", "wf_sharpe_min", "composite") if c in top.columns]
    lines = [
        '"""Auto-generated curated CTA signals (see tool/run_cta_signal_sweep.py)."""',
        "from __future__ import annotations",
        "",
        "import pandas as pd",
        "from pathlib import Path",
        "",
        "ROOT = Path(__file__).resolve().parents[1]",
        'CATALOG_CSV = ROOT / "data" / "cta_signal_catalog_top30.csv"',
        "",
        "def load_catalog() -> pd.DataFrame:",
        "    return pd.read_csv(CATALOG_CSV)",
        "",
    ]
    py_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[CTA sweep] Wrote {py_out}")


if __name__ == "__main__":
    main()

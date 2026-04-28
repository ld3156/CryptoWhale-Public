from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _to_minute_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.floor("min")


def build_whale_alert_minute_features(
    alerts_csv: Path | str,
    *,
    symbol: str = "BTC",
    dedup: bool = True,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Build minute-level whale alert features.
    Expected columns include:
      user, symbol, position_size, position_value_usd, position_action, create_time
    """
    p = Path(alerts_csv)
    usecols = [
        "user",
        "symbol",
        "position_size",
        "position_value_usd",
        "position_action",
        "create_time",
    ]

    chunks = []
    seen = set()
    reader = pd.read_csv(p, usecols=usecols, chunksize=chunksize)
    for ch in reader:
        ch = ch[ch["symbol"].astype(str).str.upper() == symbol.upper()].copy()
        if ch.empty:
            continue

        if dedup:
            # Cross-chunk dedup by stable composite key.
            key = (
                ch["user"].astype(str)
                + "|"
                + ch["symbol"].astype(str)
                + "|"
                + ch["position_size"].astype(str)
                + "|"
                + ch["position_value_usd"].astype(str)
                + "|"
                + ch["position_action"].astype(str)
                + "|"
                + ch["create_time"].astype(str)
            )
            keep_mask = ~key.isin(seen)
            seen.update(key[keep_mask].tolist())
            ch = ch.loc[keep_mask].copy()
            if ch.empty:
                continue

        ch["minute_utc"] = _to_minute_utc(ch["create_time"])
        ch = ch.dropna(subset=["minute_utc"])
        if ch.empty:
            continue

        ch["position_size"] = pd.to_numeric(ch["position_size"], errors="coerce")
        ch["position_value_usd"] = pd.to_numeric(ch["position_value_usd"], errors="coerce")
        ch["position_action"] = pd.to_numeric(ch["position_action"], errors="coerce")

        a1 = (ch["position_action"] == 1).astype("int64")
        a2 = (ch["position_action"] == 2).astype("int64")
        signed_notional = np.sign(ch["position_size"].fillna(0.0)) * ch["position_value_usd"].abs().fillna(0.0)

        agg = (
            ch.assign(
                action_1_count=a1,
                action_2_count=a2,
                abs_notional=ch["position_value_usd"].abs().fillna(0.0),
                signed_notional=signed_notional,
            )
            .groupby("minute_utc", observed=True)
            .agg(
                whale_alert_count=("user", "size"),
                whale_alert_user_n=("user", "nunique"),
                whale_alert_action1=("action_1_count", "sum"),
                whale_alert_action2=("action_2_count", "sum"),
                whale_alert_notional_abs=("abs_notional", "sum"),
                whale_alert_notional_signed=("signed_notional", "sum"),
                whale_alert_pos_size_abs=("position_size", lambda s: float(np.nansum(np.abs(s.to_numpy(dtype=float))))),
            )
        )
        chunks.append(agg)

    if not chunks:
        return pd.DataFrame(index=pd.Index([], name="minute_utc"))
    out = pd.concat(chunks, axis=0).groupby(level=0).sum().sort_index()
    out.index.name = "minute_utc"
    return out


def build_whale_position_minute_features(
    positions_csv: Path | str,
    *,
    symbol: str = "BTC",
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Build minute-level whale position snapshot features using chunked processing.
    Expected columns include:
      user, symbol, position_size, leverage, position_value_usd, unrealized_pnl, update_time
    """
    p = Path(positions_csv)
    usecols = [
        "user",
        "symbol",
        "position_size",
        "leverage",
        "position_value_usd",
        "unrealized_pnl",
        "update_time",
    ]

    bucket = []
    reader = pd.read_csv(p, usecols=usecols, chunksize=chunksize)
    for ch in reader:
        ch = ch[ch["symbol"].astype(str).str.upper() == symbol.upper()].copy()
        if ch.empty:
            continue
        ch["minute_utc"] = _to_minute_utc(ch["update_time"])
        ch = ch.dropna(subset=["minute_utc"])
        if ch.empty:
            continue

        ch["position_size"] = pd.to_numeric(ch["position_size"], errors="coerce")
        ch["leverage"] = pd.to_numeric(ch["leverage"], errors="coerce")
        ch["position_value_usd"] = pd.to_numeric(ch["position_value_usd"], errors="coerce")
        ch["unrealized_pnl"] = pd.to_numeric(ch["unrealized_pnl"], errors="coerce")

        ch["abs_size"] = ch["position_size"].abs()
        ch["abs_notional"] = ch["position_value_usd"].abs()
        ch["signed_notional"] = np.sign(ch["position_size"].fillna(0.0)) * ch["position_value_usd"].abs().fillna(0.0)
        ch["lev_x_notional"] = ch["leverage"].fillna(0.0) * ch["abs_notional"].fillna(0.0)

        agg = (
            ch.groupby("minute_utc", observed=True)
            .agg(
                whale_pos_rows=("user", "size"),
                whale_pos_size_abs=("abs_size", "sum"),
                whale_pos_size_signed=("position_size", "sum"),
                whale_pos_notional_abs=("abs_notional", "sum"),
                whale_pos_notional_signed=("signed_notional", "sum"),
                whale_pos_pnl_sum=("unrealized_pnl", "sum"),
                whale_pos_lev_x_notional=("lev_x_notional", "sum"),
            )
            .sort_index()
        )
        bucket.append(agg)

    if not bucket:
        return pd.DataFrame(index=pd.Index([], name="minute_utc"))

    out = pd.concat(bucket, axis=0).groupby(level=0).sum().sort_index()
    # Notional-weighted leverage proxy.
    denom = out["whale_pos_notional_abs"].replace(0, np.nan)
    out["whale_pos_lev_wavg"] = out["whale_pos_lev_x_notional"] / denom
    out = out.drop(columns=["whale_pos_lev_x_notional"])
    out.index.name = "minute_utc"
    return out


def add_rolling_whale_features(
    whale_minute_df: pd.DataFrame,
    windows: Iterable[int] = (5, 15, 30, 60),
) -> pd.DataFrame:
    out = whale_minute_df.copy().sort_index()
    base_cols = out.columns.tolist()
    for w in windows:
        for c in base_cols:
            # Missing minutes imply no new whale flow in current data, treat as zero
            # so rolling features remain computable on sparse event series.
            s = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            out[f"{c}_mean_{w}"] = s.rolling(w, min_periods=w).mean()
            out[f"{c}_z_{w}"] = (s - s.rolling(w, min_periods=w).mean()) / s.rolling(w, min_periods=w).std(ddof=0)
    return out


def build_whale_feature_bundle(
    positions_csv: Path | str,
    alerts_csv: Path | str,
    *,
    symbol: str = "BTC",
    alert_chunksize: int = 200_000,
    position_chunksize: int = 200_000,
    rolling_windows: Iterable[int] = (60,),
) -> pd.DataFrame:
    alert = build_whale_alert_minute_features(
        alerts_csv,
        symbol=symbol,
        dedup=True,
        chunksize=alert_chunksize,
    )
    pos = build_whale_position_minute_features(
        positions_csv,
        symbol=symbol,
        chunksize=position_chunksize,
    )
    merged = alert.join(pos, how="outer").sort_index()
    merged = add_rolling_whale_features(merged, windows=rolling_windows)
    return merged


def build_whale_risk_score(
    whale_features: pd.DataFrame,
    indicator_cols: Iterable[str],
) -> pd.Series:
    cols = [c for c in indicator_cols if c in whale_features.columns]
    if not cols:
        return pd.Series(dtype="float64")
    s = whale_features[cols].apply(pd.to_numeric, errors="coerce")
    return s.mean(axis=1)


def build_expanding_tercile_regime(series: pd.Series, min_periods: int = 60) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q33 = s.expanding(min_periods=min_periods).quantile(1.0 / 3.0).shift(1)
    q67 = s.expanding(min_periods=min_periods).quantile(2.0 / 3.0).shift(1)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[s <= q33] = 0.0
    out[(s > q33) & (s <= q67)] = 1.0
    out[s > q67] = 2.0
    return out


def attach_whale_features_to_market_table(
    market_df: pd.DataFrame,
    whale_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align whale minute features to market table (index expected to be time_ms).
    """
    out = market_df.copy()

    # 1) Prefer time_ms index when available.
    idx_num = pd.to_numeric(pd.Series(out.index), errors="coerce")
    use_index_as_ms = idx_num.notna().all() and len(idx_num) > 0 and idx_num.median() >= 100_000_000_000
    if use_index_as_ms:
        minute_utc = pd.to_datetime(idx_num.astype("int64"), unit="ms", utc=True, errors="coerce").dt.floor("min")
    # 2) Fallback: use time_utc column if present.
    elif "time_utc" in out.columns:
        minute_utc = pd.to_datetime(out["time_utc"], utc=True, errors="coerce").dt.floor("min")
    # 3) Fallback: datetime-like index.
    elif np.issubdtype(out.index.dtype, np.datetime64):
        minute_utc = pd.to_datetime(out.index, utc=True, errors="coerce").floor("min")
    else:
        # No usable time key; keep NaT so caller can detect zero coverage.
        minute_utc = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")

    # Ensure aligned index before assignment (positional, not label reindex).
    minute_utc = pd.Series(pd.to_datetime(minute_utc, utc=True, errors="coerce").to_numpy(), index=out.index)

    # Keep timezone-aware UTC dtype (avoid .values, which may drop tz info).
    out["minute_utc"] = minute_utc

    wf = whale_feature_df.reset_index().copy()
    wf["minute_utc"] = pd.to_datetime(wf["minute_utc"], utc=True, errors="coerce").dt.floor("min")
    out["minute_utc"] = pd.to_datetime(out["minute_utc"], utc=True, errors="coerce").dt.floor("min")

    # Align whale feature timeline to market bar frequency (e.g. 15m).
    valid_time = out["minute_utc"].dropna().sort_values()
    step_minutes = 1
    if len(valid_time) >= 3:
        diffs = valid_time.diff().dropna().dt.total_seconds() / 60.0
        med = int(np.median(diffs)) if len(diffs) > 0 else 1
        step_minutes = max(1, med)

    if step_minutes > 1:
        freq = f"{step_minutes}min"
        wf["minute_utc"] = wf["minute_utc"].dt.floor(freq)
        num_cols = [c for c in wf.columns if c != "minute_utc"]
        # Sum-like aggregation is reasonable for flow/intensity features.
        wf = wf.groupby("minute_utc", observed=True)[num_cols].sum().reset_index()

    merged = out.merge(
        wf,
        on="minute_utc",
        how="left",
    )
    merged = merged.set_index(out.index)
    merged.index.name = out.index.name
    return merged


def quick_whale_data_report(positions_csv: Path | str, alerts_csv: Path | str) -> Dict[str, object]:
    alerts = pd.read_csv(alerts_csv, nrows=5000)
    pos = pd.read_csv(positions_csv, nrows=5000)
    report = {
        "alerts_columns": list(alerts.columns),
        "positions_columns": list(pos.columns),
        "alerts_head_rows": int(len(alerts)),
        "positions_head_rows": int(len(pos)),
        "alerts_action_values": sorted(pd.to_numeric(alerts.get("position_action"), errors="coerce").dropna().unique().tolist())
        if "position_action" in alerts.columns
        else [],
    }
    return report

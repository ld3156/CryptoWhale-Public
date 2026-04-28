from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def numeric_to_float32(
    df: pd.DataFrame,
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Cast every numeric column (integer or float) to float32. Leaves datetime, timedelta, bool, object unchanged.
    Use after load / feature blocks to halve RAM vs float64 for research-scale frames.
    """
    out = df.copy()
    excl = set(exclude)
    for col in out.columns:
        if col in excl:
            continue
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_timedelta64_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce").astype(np.float32)
    return out


def _to_series(x):
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def ts_delay(x, n: int = 1) -> pd.Series:
    """Lag by n periods."""
    x = _to_series(x)
    return x.shift(n)


def last(x) -> pd.Series:
    """
    Previous available value before current timestamp.
    Equivalent to:
      - continuous series: ts_delay(x, 1)
      - series with missing values: last valid observation before now
    """
    x = _to_series(x)
    return x.ffill().shift(1)


def ts_delta(x, n: int = 1) -> pd.Series:
    """Difference with lag n."""
    x = _to_series(x)
    return x - x.shift(n)


def ts_returns(x, n: int = 1) -> pd.Series:
    """Percent return over n periods."""
    x = _to_series(x)
    return x.pct_change(n)


def ts_mean(x, n: int) -> pd.Series:
    x = _to_series(x)
    return x.rolling(n, min_periods=n).mean()


def ts_std(x, n: int) -> pd.Series:
    x = _to_series(x)
    return x.rolling(n, min_periods=n).std()


def ts_sum(x, n: int) -> pd.Series:
    x = _to_series(x)
    return x.rolling(n, min_periods=n).sum()


def ts_min(x, n: int) -> pd.Series:
    x = _to_series(x)
    return x.rolling(n, min_periods=n).min()


def ts_max(x, n: int) -> pd.Series:
    x = _to_series(x)
    return x.rolling(n, min_periods=n).max()


def ts_rank(x, n: int) -> pd.Series:
    """Rank of latest value within rolling window (0..1)."""
    x = _to_series(x)

    def _rank_last(arr):
        s = pd.Series(arr)
        return s.rank(pct=True).iloc[-1]

    return x.rolling(n, min_periods=n).apply(_rank_last, raw=False)


def ts_zscore(x, n: int) -> pd.Series:
    x = _to_series(x)
    m = ts_mean(x, n)
    s = ts_std(x, n)
    return (x - m) / s.replace(0, np.nan)


def ts_corr(x, y, n: int) -> pd.Series:
    x = _to_series(x)
    y = _to_series(y)
    return x.rolling(n, min_periods=n).corr(y)


def ts_cov(x, y, n: int) -> pd.Series:
    x = _to_series(x)
    y = _to_series(y)
    return x.rolling(n, min_periods=n).cov(y)


def ts_decay(x, n: int) -> pd.Series:
    """
    Linear decay weighted moving average.
    Weights: 1..n (recent points get larger weight).
    """
    x = _to_series(x)
    w = np.arange(1, n + 1, dtype=float)
    w = w / w.sum()
    return x.rolling(n, min_periods=n).apply(lambda a: float(np.dot(a, w)), raw=True)


def ts_ema(x, n: int) -> pd.Series:
    """Exponential moving average with span n."""
    x = _to_series(x)
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def ts_quantile(x, n: int, q: float = 0.5) -> pd.Series:
    """Rolling quantile over n periods."""
    x = _to_series(x)
    return x.rolling(n, min_periods=n).quantile(q)


def ts_skew(x, n: int) -> pd.Series:
    """Rolling skewness over n periods."""
    x = _to_series(x)
    return x.rolling(n, min_periods=n).skew()


def ts_kurt(x, n: int) -> pd.Series:
    """Rolling kurtosis over n periods."""
    x = _to_series(x)
    return x.rolling(n, min_periods=n).kurt()


def ts_beta(x, y, n: int) -> pd.Series:
    """Rolling beta = cov(x, y) / var(y)."""
    x = _to_series(x)
    y = _to_series(y)
    cov = x.rolling(n, min_periods=n).cov(y)
    var = y.rolling(n, min_periods=n).var()
    return cov / var.replace(0, np.nan)


def rank(x) -> pd.Series:
    """Cross-time rank for one series (0..1)."""
    x = _to_series(x)
    return x.rank(pct=True)


def scale(x, a: float = 1.0) -> pd.Series:
    """Scale series so sum(abs(x)) == a."""
    x = _to_series(x)
    denom = x.abs().sum()
    if denom == 0 or np.isnan(denom):
        return x * np.nan
    return x * (a / denom)


def signed_power(x, a: float) -> pd.Series:
    x = _to_series(x)
    return np.sign(x) * (np.abs(x) ** a)


def clip(x, low=None, high=None) -> pd.Series:
    x = _to_series(x)
    return x.clip(lower=low, upper=high)


def safe_div(x, y) -> pd.Series:
    x = _to_series(x)
    y = _to_series(y)
    return x / y.replace(0, np.nan)


def ifcond(cond, x, y) -> pd.Series:
    """
    Conditional operator:
      ifcond(cond, x, y)
    cond can be a boolean Series/formula (e.g. rsi > 80).
    """
    cond_s = _to_series(cond).fillna(False).astype(bool)
    idx = cond_s.index

    # Correct scalar broadcasting to condition index.
    x_s = pd.Series(x, index=idx) if np.isscalar(x) else _to_series(x).reindex(idx)
    y_s = pd.Series(y, index=idx) if np.isscalar(y) else _to_series(y).reindex(idx)

    return pd.Series(np.where(cond_s, x_s, y_s), index=idx)


def winsorize(x, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    x = _to_series(x)
    lo = x.quantile(lower_q)
    hi = x.quantile(upper_q)
    return x.clip(lo, hi)


def formula_env(df: pd.DataFrame) -> Dict[str, object]:
    """
    Build eval environment:
    - all df columns are variables
    - all functions above are available
    """
    env: Dict[str, object] = {c: df[c] for c in df.columns}
    env.update(
        {
            "np": np,
            "pd": pd,
            "ts_delay": ts_delay,
            "last": last,
            "ts_delta": ts_delta,
            "ts_returns": ts_returns,
            "ts_mean": ts_mean,
            "ts_std": ts_std,
            "ts_sum": ts_sum,
            "ts_min": ts_min,
            "ts_max": ts_max,
            "ts_rank": ts_rank,
            "ts_zscore": ts_zscore,
            "ts_corr": ts_corr,
            "ts_cov": ts_cov,
            "ts_decay": ts_decay,
            "ts_ema": ts_ema,
            "ts_quantile": ts_quantile,
            "ts_skew": ts_skew,
            "ts_kurt": ts_kurt,
            "ts_beta": ts_beta,
            "rank": rank,
            "scale": scale,
            "signed_power": signed_power,
            "clip": clip,
            "safe_div": safe_div,
            "ifcond": ifcond,
            "winsorize": winsorize,
            "abs": np.abs,
            "log": np.log,
            "exp": np.exp,
            "sqrt": np.sqrt,
        }
    )
    return env


def apply_formulas(
    df: pd.DataFrame,
    formulas: Dict[str, str],
    *,
    output_float32: bool = True,
) -> pd.DataFrame:
    """
    formulas: {"new_feature_name": "formula string"}
    Example:
      {"feat1": "ts_decay(rsi, 3)", "feat2": "ts_zscore(close, 60)"}

    When output_float32 is True (default), all numeric columns are cast to float32 after evaluation
    (pandas rolling ops often promote to float64 mid-pipeline).

    New columns are accumulated in memory and joined in **one** ``pd.concat`` to avoid DataFrame
    fragmentation and repeated ``PerformanceWarning`` from inserting many columns one-by-one.
    """
    base = df.copy()
    if not formulas:
        return numeric_to_float32(base) if output_float32 else base

    env = formula_env(base)
    computed: Dict[str, pd.Series] = {}
    for name, expr in formulas.items():
        computed[name] = eval(expr, {"__builtins__": {}}, env)
        env[name] = computed[name]

    add = pd.DataFrame(computed, index=base.index)
    overlap = [c for c in add.columns if c in base.columns]
    if overlap:
        base = base.drop(columns=overlap, errors="ignore")
    out = pd.concat([base, add], axis=1)
    out = out.copy()
    if output_float32:
        out = numeric_to_float32(out)
    return out


def list_feature_columns(df: pd.DataFrame, exclude: Iterable[str] = ("time_utc",)) -> list[str]:
    return [c for c in df.columns if c not in set(exclude)]


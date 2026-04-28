from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class CoverageConfig:
    data_dir: Path
    frequencies: Tuple[str, ...] = ("1m", "5m", "15m", "1h")
    figsize: Tuple[int, int] = (18, 10)


class CoverageVisualizer:
    """Plot CSV time coverage as horizontal lines by file."""

    def __init__(self, config: CoverageConfig):
        self.config = config
        self.interval_to_ms = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
        }

    def _detect_freq(self, stem: str):
        for f in self.config.frequencies:
            if stem.endswith(f"_{f}"):
                return f
        return None

    def _load_segments(self):
        csv_files = sorted([p for p in self.config.data_dir.glob("*.csv") if not p.name.startswith("_")])
        rows = []
        all_times = []

        for fp in csv_files:
            freq = self._detect_freq(fp.stem)
            if not freq:
                continue

            try:
                df = pd.read_csv(fp, usecols=["time_ms"])
            except Exception:
                continue

            ts = pd.to_numeric(df["time_ms"], errors="coerce").dropna().astype("int64")
            ts = ts[ts > 0].drop_duplicates().sort_values()
            if ts.empty:
                continue

            utc = pd.to_datetime(ts, unit="ms", utc=True)
            all_times.append(utc)
            step = self.interval_to_ms.get(freq)

            segments = []
            if step and len(ts) >= 2:
                vals = ts.to_numpy()
                start_idx = 0
                for i in range(1, len(vals)):
                    if vals[i] - vals[i - 1] > step:
                        seg_start = pd.to_datetime(vals[start_idx], unit="ms", utc=True)
                        seg_end = pd.to_datetime(vals[i - 1], unit="ms", utc=True)
                        segments.append((seg_start, seg_end))
                        start_idx = i
                seg_start = pd.to_datetime(vals[start_idx], unit="ms", utc=True)
                seg_end = pd.to_datetime(vals[-1], unit="ms", utc=True)
                segments.append((seg_start, seg_end))
            else:
                segments.append((utc.iloc[0], utc.iloc[-1]))

            rows.append({"name": fp.stem, "segments": segments, "freq": freq})

        return rows, all_times

    def plot(self):
        rows, all_times = self._load_segments()
        if not rows:
            print("No valid CSV coverage data found.")
            return

        global_start = min(s.min() for s in all_times)
        global_end = max(s.max() for s in all_times)

        fig_h = max(6, 0.35 * len(rows) + 2)
        fig, ax = plt.subplots(figsize=(self.config.figsize[0], fig_h))

        for i, item in enumerate(rows):
            ax.hlines(i, global_start, global_end, color="lightgray", linewidth=1.5, alpha=0.8)
            for seg_start, seg_end in item["segments"]:
                ax.hlines(i, seg_start, seg_end, color="tab:blue", linewidth=3)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["name"] for r in rows])
        ax.set_xlim(global_start, global_end)
        ax.set_xlabel("UTC Timeline")
        ax.set_ylabel("CSV Files")
        ax.set_title("Time Coverage by CSV (UTC)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_feature_coverage(
        self,
        df: pd.DataFrame,
        title: str = "Feature Coverage in Table (UTC)",
        exclude_cols: Tuple[str, ...] = ("time_utc",),
    ):
        """
        Plot per-feature time coverage for one aligned/feature table.
        Assumptions:
        - index is time_ms (preferred), or a 'time_ms' column exists.
        - each feature may have NaN gaps; coverage is shown in blue segments.
        """
        t = df.copy()
        if t.index.name != "time_ms":
            if "time_ms" in t.columns:
                t = t.set_index("time_ms")
            else:
                raise ValueError("DataFrame must have time_ms index or time_ms column.")

        idx_ms = pd.to_numeric(pd.Series(t.index), errors="coerce").dropna().astype("int64")
        if idx_ms.empty:
            print("No valid time_ms found in table.")
            return

        time_utc = pd.to_datetime(idx_ms, unit="ms", utc=True)
        global_start = time_utc.min()
        global_end = time_utc.max()

        feature_cols = [c for c in t.columns if c not in set(exclude_cols)]
        if not feature_cols:
            print("No feature columns to plot.")
            return

        rows = []
        for c in feature_cols:
            s = pd.to_numeric(t[c], errors="coerce")
            valid = s.notna()
            if valid.sum() == 0:
                continue

            valid_times = pd.to_datetime(pd.to_numeric(pd.Series(t.index[valid]), errors="coerce"), unit="ms", utc=True)
            if valid_times.empty:
                continue

            # Segment contiguous non-NaN runs by checking index continuity.
            ms_vals = pd.to_numeric(pd.Series(t.index[valid]), errors="coerce").dropna().astype("int64").to_numpy()
            segs = []
            start = 0
            if len(ms_vals) == 1:
                ts = pd.to_datetime(ms_vals[0], unit="ms", utc=True)
                segs.append((ts, ts))
            else:
                # infer step from median positive diff
                diffs = np.diff(ms_vals)
                pos_diffs = diffs[diffs > 0]
                step = int(np.median(pos_diffs)) if len(pos_diffs) else 0
                for i in range(1, len(ms_vals)):
                    if step > 0 and (ms_vals[i] - ms_vals[i - 1]) > step:
                        segs.append(
                            (
                                pd.to_datetime(ms_vals[start], unit="ms", utc=True),
                                pd.to_datetime(ms_vals[i - 1], unit="ms", utc=True),
                            )
                        )
                        start = i
                segs.append(
                    (
                        pd.to_datetime(ms_vals[start], unit="ms", utc=True),
                        pd.to_datetime(ms_vals[-1], unit="ms", utc=True),
                    )
                )
            rows.append({"name": c, "segments": segs})

        if not rows:
            print("No non-empty feature columns to plot.")
            return

        fig_h = max(6, 0.32 * len(rows) + 2)
        fig, ax = plt.subplots(figsize=(self.config.figsize[0], fig_h))
        for i, item in enumerate(rows):
            ax.hlines(i, global_start, global_end, color="lightgray", linewidth=1.2, alpha=0.8)
            for seg_start, seg_end in item["segments"]:
                ax.hlines(i, seg_start, seg_end, color="tab:blue", linewidth=2.6)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["name"] for r in rows])
        ax.set_xlim(global_start, global_end)
        ax.set_xlabel("UTC Timeline")
        ax.set_ylabel("Features")
        ax.set_title(title)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()


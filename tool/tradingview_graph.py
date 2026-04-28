import warnings
from dataclasses import dataclass
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


@dataclass
class GraphColumns:
    close: str
    ema: str
    ma: str
    boll_lb: str
    boll_mb: str
    boll_ub: str
    macd: str
    macd_signal: str
    macd_hist: str
    rsi: str
    open_col: str
    high_col: str
    low_col: str


class TradingViewGrapher:
    """Reusable TradingView-style plotter for aligned 1m tables."""

    def __init__(self, max_points_full: int = 2500, recent_minutes: int = 180):
        self.max_points_full = max_points_full
        self.recent_minutes = recent_minutes
        warnings.filterwarnings("ignore")

    def default_columns(self) -> GraphColumns:
        return GraphColumns(
            close="futures_price_history_btcusdt_binance__close",
            ema="futures_ema_history_btcusdt_binance__ema_value",
            ma="futures_ma_history_btcusdt_binance__ma_value",
            boll_lb="futures_boll_history_btcusdt_binance__lb_value",
            boll_mb="futures_boll_history_btcusdt_binance__mb_value",
            boll_ub="futures_boll_history_btcusdt_binance__ub_value",
            macd="futures_macd_history_btcusdt_binance__macd_value",
            macd_signal="futures_macd_history_btcusdt_binance__signal",
            macd_hist="futures_macd_history_btcusdt_binance__histogram",
            rsi="futures_rsi_history_btcusdt_binance__rsi_value",
            open_col="futures_price_history_btcusdt_binance__open",
            high_col="futures_price_history_btcusdt_binance__high",
            low_col="futures_price_history_btcusdt_binance__low",
        )

    def _prepare(self, df_1m: pd.DataFrame, col: GraphColumns) -> pd.DataFrame:
        d = df_1m.copy()
        if d.index.name != "time_ms":
            if "time_ms" in d.columns:
                d = d.set_index("time_ms")
        if "time_utc" in d.columns:
            d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
        else:
            d["time_utc"] = pd.to_datetime(d.index.astype("int64"), unit="ms", utc=True)

        required = [
            col.close, col.ema, col.ma, col.boll_lb, col.boll_mb, col.boll_ub,
            col.macd, col.macd_signal, col.macd_hist, col.rsi,
        ]
        missing = [c for c in required if c not in d.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for c in required + [col.open_col, col.high_col, col.low_col]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        mask_full = d[required].notna().all(axis=1)
        if not mask_full.any():
            raise ValueError("No timestamp has full feature coverage for close/EMA/MA/BOLL/MACD/RSI.")

        start_idx = mask_full[mask_full].index[0]
        sub = d.loc[d.index >= start_idx].dropna(subset=required).copy()
        if len(sub) > self.max_points_full:
            sub = sub.tail(self.max_points_full).copy()

        print("First full-coverage timestamp (UTC):", pd.to_datetime(int(start_idx), unit="ms", utc=True))
        print("Plot rows (full-view):", len(sub))
        return sub

    def _style_axes(self, ax_price, ax_macd, ax_rsi):
        ax_rsi.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax_price.tick_params(axis="x", labelbottom=False)
        ax_macd.tick_params(axis="x", labelbottom=False)

        for ax in (ax_price, ax_macd, ax_rsi):
            ax.tick_params(colors="#c9d1d9")
            for spine in ax.spines.values():
                spine.set_color("#30363d")

    def _plot_indicators(self, ax_price, ax_macd, ax_rsi, x, close, ema, ma, lb, mb, ub, macd, macd_signal, macd_hist, rsi):
        ax_price.plot(x, close, color="#e6edf3", linewidth=1.2, label="Close")
        ax_price.plot(x, ema, color="#ff9800", linewidth=1.0, label="EMA")
        ax_price.plot(x, ma, color="#4dd0e1", linewidth=1.0, label="MA")
        ax_price.plot(x, ub, color="#ab47bc", linewidth=0.9, alpha=0.9, label="BOLL UB")
        ax_price.plot(x, mb, color="#7e57c2", linewidth=0.9, alpha=0.9, label="BOLL MB")
        ax_price.plot(x, lb, color="#5e35b1", linewidth=0.9, alpha=0.9, label="BOLL LB")
        ax_price.fill_between(x, lb, ub, color="#7e57c2", alpha=0.10)
        ax_price.set_ylabel("Price (USD)")
        ax_price.grid(True, linestyle="--", alpha=0.18)
        ax_price.legend(loc="upper left", ncol=3, frameon=False)

        bar_colors = np.where(macd_hist >= 0, "#26a69a", "#ef5350")
        ax_macd.bar(x, macd_hist, width=0.0008, color=bar_colors, alpha=0.85, label="Histogram")
        ax_macd.plot(x, macd, color="#4fc3f7", linewidth=1.0, label="MACD")
        ax_macd.plot(x, macd_signal, color="#ffca28", linewidth=1.0, label="Signal")
        ax_macd.axhline(0, color="#9e9e9e", linewidth=0.8, alpha=0.7)
        ax_macd.set_ylabel("MACD")
        ax_macd.grid(True, linestyle="--", alpha=0.18)
        ax_macd.legend(loc="upper left", ncol=3, frameon=False)

        ax_rsi.plot(x, rsi, color="#66bb6a", linewidth=1.1, label="RSI")
        ax_rsi.axhline(70, color="#ef5350", linestyle="--", linewidth=0.9, alpha=0.8)
        ax_rsi.axhline(30, color="#26a69a", linestyle="--", linewidth=0.9, alpha=0.8)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.grid(True, linestyle="--", alpha=0.18)
        ax_rsi.legend(loc="upper left", frameon=False)

    def plot_full(self, sub: pd.DataFrame, col: GraphColumns):
        x = sub["time_utc"]
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 10), facecolor="#0f1117")
        fig.suptitle("1m Strategy View: Price + EMA/MA/BOLL + MACD + RSI", fontsize=14)
        gs = GridSpec(3, 1, height_ratios=[3.8, 1.7, 1.7], hspace=0.12)
        ax_price = fig.add_subplot(gs[0])
        ax_macd = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)

        self._plot_indicators(
            ax_price, ax_macd, ax_rsi, x,
            sub[col.close], sub[col.ema], sub[col.ma],
            sub[col.boll_lb], sub[col.boll_mb], sub[col.boll_ub],
            sub[col.macd], sub[col.macd_signal], sub[col.macd_hist], sub[col.rsi]
        )
        self._style_axes(ax_price, ax_macd, ax_rsi)
        fig.subplots_adjust(hspace=0.14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def plot_recent_candles(self, sub: pd.DataFrame, col: GraphColumns):
        recent = sub.tail(self.recent_minutes).copy()
        if len(recent) < 2:
            print("Not enough points for recent candlestick view.")
            return

        x = recent["time_utc"]
        close = recent[col.close]
        ema = recent[col.ema]
        ma = recent[col.ma]
        lb = recent[col.boll_lb]
        mb = recent[col.boll_mb]
        ub = recent[col.boll_ub]
        macd = recent[col.macd]
        macd_signal = recent[col.macd_signal]
        macd_hist = recent[col.macd_hist]
        rsi = recent[col.rsi]

        o = close.shift(1).fillna(close)
        h = close
        l = close
        if col.open_col in recent.columns:
            o = recent[col.open_col]
        if col.high_col in recent.columns:
            h = recent[col.high_col]
        if col.low_col in recent.columns:
            l = recent[col.low_col]

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 10), facecolor="#0f1117")
        fig.suptitle(
            f"1m Strategy View (Recent {self.recent_minutes} Minutes): Candles + EMA/MA/BOLL + MACD + RSI",
            fontsize=14,
        )
        gs = GridSpec(3, 1, height_ratios=[3.8, 1.7, 1.7], hspace=0.12)
        ax_price = fig.add_subplot(gs[0])
        ax_macd = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)

        candle_width = 0.60 / (24 * 60)
        ax_price.vlines(x, l, h, color="#b0bec5", linewidth=0.8, alpha=0.9)
        up = close >= o
        down = ~up
        ax_price.bar(x[up], (close[up] - o[up]), bottom=o[up], width=candle_width,
                     color="#26a69a", edgecolor="#26a69a", alpha=0.9, align="center")
        ax_price.bar(x[down], (close[down] - o[down]), bottom=o[down], width=candle_width,
                     color="#ef5350", edgecolor="#ef5350", alpha=0.9, align="center")

        self._plot_indicators(
            ax_price, ax_macd, ax_rsi, x,
            close, ema, ma, lb, mb, ub, macd, macd_signal, macd_hist, rsi
        )
        self._style_axes(ax_price, ax_macd, ax_rsi)
        fig.subplots_adjust(hspace=0.14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def run(self, df_1m: pd.DataFrame, columns: GraphColumns = None):
        col = columns or self.default_columns()
        prepared = self._prepare(df_1m, col)
        self.plot_full(prepared, col)
        self.plot_recent_candles(prepared, col)


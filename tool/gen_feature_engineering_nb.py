"""Regenerate main/Feature_Engineering.ipynb — CTA catalog baseline only. Run: python tool/gen_feature_engineering_nb.py"""
from __future__ import annotations

import json
from pathlib import Path


def src(lines: list[str]) -> list[str]:
    return [line + "\n" for line in lines]


def main() -> None:
    cells: list[dict] = []

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(
                [
                    "# CTA 基准策略（目录 Top30）",
                    "",
                    "**唯一主线**：使用 `data/cta_signal_catalog_top30.csv` 中的信号（可全用或只取前 N 条）。"
                    "主数据为 **1m OHLCV**；每个信号在内部按 `bar_minutes` 重采样、计算仓位，再 **前向填充到每分钟**。",
                    "",
                    "- **样本内（IS）**：`time_utc < TRAIN_END_EXCLUSIVE_UTC`",
                    "- **样本外（OOS）**：`>= TRAIN_END_EXCLUSIVE_UTC`",
                    "- **费后净值**：`position * ret_1 - turnover * (fee_bps+slippage)/1e4`，与 `cta_signal_lab` 一致。",
                    "",
                    "更新目录：在项目根运行 `python tool/run_cta_signal_sweep.py`。",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "from __future__ import annotations",
                    "",
                    "from pathlib import Path",
                    "import sys",
                    "",
                    "import matplotlib.pyplot as plt",
                    "import pandas as pd",
                    "from IPython.display import display",
                    "",
                    "ROOT = Path.cwd()",
                    'if not (ROOT / "tool").exists() and (ROOT.parent / "tool").exists():',
                    "    ROOT = ROOT.parent",
                    "if str(ROOT) not in sys.path:",
                    "    sys.path.insert(0, str(ROOT))",
                    "",
                    "from tool.core_cta_baseline import (",
                    "    TRAIN_END_EXCLUSIVE_UTC,",
                    "    load_cleaned_1m_baseline,",
                    "    train_test_masks_from_time_utc,",
                    ")",
                    "from tool.cta_multi_freq_lab import prepare_cta_feature_frame",
                    "from tool.cta_catalog_execution import (",
                    "    ensemble_position_equal_weight,",
                    "    net_return_with_costs,",
                    "    positions_table_from_catalog,",
                    ")",
                    "from tool.cta_signal_catalog import CATALOG_CSV, load_catalog",
                    "from tool.cta_signal_lab import BacktestConfig, compute_strategy_metrics",
                    "from tool.newmath import numeric_to_float32",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "# --- 配置 ---",
                    "CATALOG_TOP_N = 30  # 设为 None 使用 CSV 全部行；或改为 10 等",
                    "EXEC_CFG = BacktestConfig(",
                    '    freq="1m",',
                    "    fee_bps=3.0,",
                    "    slippage_bps=1.0,",
                    "    signal_shift=1,",
                    "    signal_clip=2.5,",
                    "    max_abs_position=1.0,",
                    "    position_step=0.1,",
                    ")",
                    "",
                    'print("ROOT:", ROOT)',
                    'print("Train end (OOS starts here):", TRAIN_END_EXCLUSIVE_UTC)',
                    'print("Catalog:", CATALOG_CSV.resolve())',
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "# 1m 表 + ret_1",
                    "df = load_cleaned_1m_baseline(ROOT)",
                    'df = numeric_to_float32(df, exclude=("time_utc",))',
                    "df = prepare_cta_feature_frame(df)",
                    'ret_1 = df["ret_1"].astype("float64")',
                    "",
                    "train_mask, test_mask = train_test_masks_from_time_utc(df)",
                    'print("rows:", len(df), "| IS:", int(train_mask.sum()), "| OOS:", int(test_mask.sum()))',
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "# 目录 → 1m 仓位矩阵 + 等权组合",
                    "cat = load_catalog()",
                    "if CATALOG_TOP_N is not None:",
                    "    cat = cat.head(int(CATALOG_TOP_N))",
                    "",
                    "pos_mat = positions_table_from_catalog(df, cat, cfg=EXEC_CFG)",
                    "pos_ens = ensemble_position_equal_weight(pos_mat)",
                    "net_ens = net_return_with_costs(pos_ens, ret_1, EXEC_CFG)",
                    "",
                    'print("Signals:", pos_mat.shape[1], "| pos_ens min/max:", float(pos_ens.min()), float(pos_ens.max()))',
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(["### 组合 — 样本内 / 样本外 / 全样本（费后）"]),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "idx = net_ens.index",
                    "tr = train_mask.reindex(idx).fillna(False).astype(bool)",
                    "te = test_mask.reindex(idx).fillna(False).astype(bool)",
                    "",
                    "def _m(name, net_ser, pos_ser):",
                    '    return {"segment": name, **compute_strategy_metrics(net_ser, EXEC_CFG, position=pos_ser)}',
                    "",
                    "rows_ens = [",
                    '    _m("IS (in-sample)", net_ens.loc[tr], pos_ens.loc[tr]),',
                    '    _m("OOS (out-of-sample)", net_ens.loc[te], pos_ens.loc[te]),',
                    '    _m("Full", net_ens, pos_ens),',
                    "]",
                    'm_ens = pd.DataFrame(rows_ens).set_index("segment")',
                    "_cols = [c for c in (\"ann_ret\", \"ann_vol\", \"sharpe\", \"calmar\", \"max_dd\", \"turnover\", \"hit_rate\", \"profit_factor\", \"obs\") if c in m_ens.columns]",
                    "display(m_ens[_cols].round(4))",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(["### 单信号 — 各段 Sharpe（费后，便于排查）"]),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "per_rows = []",
                    "for col in pos_mat.columns:",
                    "    pos = pos_mat[col]",
                    "    net = net_return_with_costs(pos, ret_1, EXEC_CFG)",
                    "    m_is = compute_strategy_metrics(net.loc[tr], EXEC_CFG, position=pos.loc[tr])",
                    "    m_os = compute_strategy_metrics(net.loc[te], EXEC_CFG, position=pos.loc[te])",
                    "    m_all = compute_strategy_metrics(net, EXEC_CFG, position=pos)",
                    "    per_rows.append({",
                    '        "key": col,',
                    '        "name": col[:100],',
                    '        "Full_sharpe": m_all.get("sharpe"),',
                    '        "IS_sharpe": m_is.get("sharpe"),',
                    '        "OOS_sharpe": m_os.get("sharpe"),',
                    '        "IS_ann_ret": m_is.get("ann_ret"),',
                    '        "OOS_ann_ret": m_os.get("ann_ret"),',
                    '        "IS_max_dd": m_is.get("max_dd"),',
                    '        "OOS_max_dd": m_os.get("max_dd"),',
                    "    })",
                    "",
                    'per_df = pd.DataFrame(per_rows).sort_values(by="OOS_sharpe", ascending=False, na_position="last")',
                    "display(per_df.round(4))",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(["### 净值曲线（竖线 = 训练截止）"]),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "eq = (1.0 + net_ens.fillna(0.0)).cumprod()",
                    't_plot = pd.to_datetime(df["time_utc"], utc=True)',
                    "",
                    "fig, ax = plt.subplots(figsize=(12, 4))",
                    'ax.plot(t_plot, eq.values, color="#1d4ed8", lw=1.0, label="ensemble")',
                    "ax.axvline(TRAIN_END_EXCLUSIVE_UTC, color=\"#b91c1c\", ls=\"--\", lw=1, label=\"IS | OOS\")",
                    'ax.set_title("CTA catalog equal-weight — cumulative equity (net of fees)")',
                    'ax.legend(loc="upper left")',
                    "ax.grid(alpha=0.25)",
                    "plt.tight_layout()",
                    "plt.show()",
                    "",
                    "import matplotlib.dates as mdates",
                    "",
                    "eq_is = (1.0 + net_ens.loc[tr].fillna(0.0)).cumprod()",
                    "eq_oos = (1.0 + net_ens.loc[te].fillna(0.0)).cumprod()",
                    "fig2, ax2 = plt.subplots(1, 2, figsize=(12, 3.5))",
                    'ax2[0].plot(pd.to_datetime(df.loc[tr, "time_utc"], utc=True), eq_is.values, color="#059669", lw=1)',
                    'ax2[0].set_title("In-sample equity")',
                    "ax2[0].grid(alpha=0.25)",
                    'ax2[1].plot(pd.to_datetime(df.loc[te, "time_utc"], utc=True), eq_oos.values, color="#b91c1c", lw=1)',
                    'ax2[1].set_title("Out-of-sample equity")',
                    "ax2[1].grid(alpha=0.25)",
                    "for _ax in ax2:",
                    "    _loc = mdates.AutoDateLocator(minticks=4, maxticks=6)",
                    "    _ax.xaxis.set_major_locator(_loc)",
                    "    _ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_loc))",
                    "fig2.autofmt_xdate(rotation=22, ha=\"right\")",
                    "plt.tight_layout()",
                    "plt.show()",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(
                [
                    "### Part 2 — 赚钱来源拆解",
                    "",
                    "- **多 / 空 / 平** 的 **毛收益** 与 **费后净收益**（组合层）",
                    "- **单信号 standalone** 费后净值累加（每条信号若单独交易；与等权组合不同，因组合有分散化与合成换手）",
                    "- **柱状图**：各信号全样本累计费后净收益贡献（standalone）",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "from tool.cta_strategy_diagnostics import (",
                    "    ensemble_pnl_decomposition,",
                    "    per_signal_net_series,",
                    "    standalone_vs_ensemble_table,",
                    ")",
                    "",
                    "net_per = per_signal_net_series(pos_mat, ret_1, EXEC_CFG)",
                    "decomp = ensemble_pnl_decomposition(pos_ens, ret_1, net_ens)",
                    'display(decomp.set_index("component"))',
                    "",
                    "stab = standalone_vs_ensemble_table(net_per, net_ens)",
                    "display(stab[[c for c in stab.columns if c != \"ensemble_total_net\"]].head(15))",
                    "",
                    "tot = net_per.sum().sort_values(ascending=False)",
                    "fig, ax = plt.subplots(figsize=(10, 4))",
                    "tot.head(15).iloc[::-1].plot(kind=\"barh\", ax=ax, color=\"#0ea5e9\")",
                    'ax.set_title("Top 15 signals — standalone cumulative net sum (fee-adj.)")',
                    "plt.tight_layout()",
                    "plt.show()",
                    "",
                    "top_cols = tot.head(8).index",
                    "fig2, ax2 = plt.subplots(figsize=(12, 4))",
                    "for col in top_cols:",
                    "    ax2.plot(net_per[col].cumsum().values, alpha=0.85, label=str(col)[:55])",
                    "ax2.legend(loc=\"upper left\", fontsize=7)",
                    'ax2.set_title("Cumulative standalone net (top 8 by total sum)")',
                    "ax2.grid(alpha=0.25)",
                    "plt.tight_layout()",
                    "plt.show()",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(
                [
                    "### Part 3 — 仓位管理",
                    "",
                    "- 仓位分布、随时间暴露、换手相关统计（组合 `pos_ens`）",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "from tool.cta_strategy_diagnostics import position_management_stats",
                    "",
                    "pm = position_management_stats(pos_ens, EXEC_CFG)",
                    "display(pd.DataFrame([pm.__dict__]).T.rename(columns={0: \"value\"}))",
                    "",
                    "fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))",
                    "axes[0].hist(pos_ens.values, bins=36, color=\"#64748b\", edgecolor=\"white\")",
                    'axes[0].set_title("Ensemble position histogram")',
                    "axes[1].plot(pd.to_datetime(df[\"time_utc\"], utc=True), pos_ens.values, lw=0.25, alpha=0.65, color=\"#334155\")",
                    'axes[1].set_title("Position vs time (1m)")',
                    "axes[1].axhline(0, color=\"#94a3b8\", lw=0.6)",
                    "plt.tight_layout()",
                    "plt.show()",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(
                [
                    "### Part 4 — 蜡烛图 + 买卖标记",
                    "",
                    "- **绿色上三角**：做多进场或平空（`pos` 从 ≤0 到 >eps）",
                    "- **红色下三角**：平多或开空",
                    "- **紫 / 蓝**：纯空头开平（若组合有空头暴露）",
                    "- 默认 **重采样为 15m** 并只画 **最后若干根**，否则 1m 过密；可改 `CHART_RESAMPLE` / `CHART_TAIL`",
                    "- 第二张图：**OOS Sharpe 最高**的单条目录信号（与组合对比）",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "from tool.cta_strategy_diagnostics import plot_candles_with_marks_simple",
                    "",
                    "CHART_RESAMPLE = \"15min\"",
                    "CHART_TAIL = 650",
                    "",
                    "fig_a, _ = plot_candles_with_marks_simple(",
                    "    df,",
                    "    pos_ens,",
                    "    resample_rule=CHART_RESAMPLE,",
                    "    tail_bars=CHART_TAIL,",
                    "    eps=0.02,",
                    '    title=f"Ensemble — last ~{CHART_TAIL} resampled bars ({CHART_RESAMPLE})",',
                    ")",
                    "plt.show()",
                    "",
                    "best_key = per_df.sort_values(\"OOS_sharpe\", ascending=False).iloc[0][\"key\"]",
                    "pos_best = pos_mat[best_key]",
                    "fig_b, _ = plot_candles_with_marks_simple(",
                    "    df,",
                    "    pos_best,",
                    "    resample_rule=CHART_RESAMPLE,",
                    "    tail_bars=CHART_TAIL,",
                    "    eps=0.02,",
                    '    title=f"Best OOS Sharpe signal: {str(best_key)[:70]} ...",',
                    ")",
                    "plt.show()",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": src(
                [
                    "### Part 5 — 波动率目标 vs 逆波动加权（风险平价近似）",
                    "",
                    "- **波动率目标**：在等权组合 `pos_ens` 上，用滚动 **毛收益波动** `rolling_std(pos_ens * ret_1)` 年化后，按 `目标波动 / 实现波动` 缩放仓位（带杠杆上下限），再 **clip 到 ±1**；费后收益用 `net_return_with_costs` 重算。",
                    "- **逆波动加权**：各信号 gross `pos_i * ret_1` 的滚动波动率取倒数，行内归一化为权重，再 `sum_i w_i * pos_i`（忽略相关性，属常用 **risk parity 近似**）。",
                    "- 可调：`VOL_WIN`（默认约 7 天 1m）、`TARGET_ANN_VOL`、`MAX_LEV`。",
                ]
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": src(
                [
                    "from tool.cta_ensemble_alloc import bars_per_year, risk_parity_position, vol_target_position",
                    "",
                    "BARS_Y = bars_per_year(EXEC_CFG)",
                    "VOL_WIN = 10_080  # ~7 天 × 1440 根 1m",
                    "TARGET_ANN_VOL = 0.12",
                    "MAX_LEV = 2.5",
                    "",
                    "pos_vt = vol_target_position(",
                    "    pos_ens,",
                    "    ret_1,",
                    "    target_ann_vol=TARGET_ANN_VOL,",
                    "    window=VOL_WIN,",
                    "    bars_per_year=BARS_Y,",
                    "    max_leverage=MAX_LEV,",
                    "    min_ann_vol=0.04,",
                    "    clip_abs=1.0,",
                    ")",
                    "net_vt = net_return_with_costs(pos_vt, ret_1, EXEC_CFG)",
                    "",
                    "pos_rp = risk_parity_position(pos_mat, ret_1, window=VOL_WIN, clip_abs=1.0)",
                    "net_rp = net_return_with_costs(pos_rp, ret_1, EXEC_CFG)",
                    "",
                    'print("pos_vt min/max:", float(pos_vt.min()), float(pos_vt.max()), "| pos_rp min/max:", float(pos_rp.min()), float(pos_rp.max()))',
                    "",
                    "variants = [",
                    '    ("equal_weight", net_ens, pos_ens),',
                    '    ("vol_target", net_vt, pos_vt),',
                    '    ("risk_parity_inv_vol", net_rp, pos_rp),',
                    "]",
                    "rows_alloc = []",
                    "for label, net_s, pos_s in variants:",
                    "    rows_alloc.append({\"variant\": label, \"segment\": \"IS\", **compute_strategy_metrics(net_s.loc[tr], EXEC_CFG, position=pos_s.loc[tr])})",
                    "    rows_alloc.append({\"variant\": label, \"segment\": \"OOS\", **compute_strategy_metrics(net_s.loc[te], EXEC_CFG, position=pos_s.loc[te])})",
                    "    rows_alloc.append({\"variant\": label, \"segment\": \"Full\", **compute_strategy_metrics(net_s, EXEC_CFG, position=pos_s)})",
                    "alloc_df = pd.DataFrame(rows_alloc)",
                    "_ac = [c for c in (\"variant\", \"segment\", \"ann_ret\", \"ann_vol\", \"sharpe\", \"calmar\", \"max_dd\", \"turnover\") if c in alloc_df.columns]",
                    "display(alloc_df[_ac].round(4))",
                    "",
                    "t_all = pd.to_datetime(df[\"time_utc\"], utc=True)",
                    "fig, ax = plt.subplots(figsize=(12, 4))",
                    "for label, net_s, _ in variants:",
                    "    eq = (1.0 + net_s.fillna(0.0)).cumprod()",
                    "    ax.plot(t_all, eq.values, lw=1.0, label=label)",
                    "ax.axvline(TRAIN_END_EXCLUSIVE_UTC, color=\"#b91c1c\", ls=\"--\", lw=1, alpha=0.8)",
                    'ax.set_title("Cumulative equity (net of fees) — equal vs vol-target vs inv-vol mix")',
                    'ax.legend(loc="upper left", fontsize=8)',
                    "ax.grid(alpha=0.25)",
                    "plt.tight_layout()",
                    "plt.show()",
                ]
            ),
        }
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "python3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = Path(__file__).resolve().parents[1] / "main" / "Feature_Engineering.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", out, "cells=", len(cells))


if __name__ == "__main__":
    main()

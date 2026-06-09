"""
eda.py
------
Exploratory Data Analysis module.

Produces 10 publication-quality dark-theme charts covering:
  1.  Sentiment distribution (trade count)
  2.  Daily profit trend over time
  3.  PnL distribution by sentiment (violin)
  4.  Average PnL per trade by sentiment
  5.  Win rate by sentiment
  6.  Position size (USD) by sentiment
  7.  Trade direction (Buy vs Sell) by sentiment
  8.  Long vs Short PnL comparison
  9.  Correlation heatmap
  10. Cumulative PnL over time

Also runs four statistical tests and prints a formatted summary:
  - Welch's T-Test   (Fear PnL vs Greed PnL)
  - One-Way ANOVA    (PnL across all 5 sentiment categories)
  - Chi-Square Test  (sentiment × win/loss)
  - Pearson Correlation table

All figures are saved to outputs/figures/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

from src.utils import (
    apply_dark_style, save_fig, sentiment_colors,
    SENTIMENT_ORDER, SENTIMENT_PALETTE, DARK_BG, PANEL_BG,
    TEXT_COLOR, GRID_COLOR, ACCENT
)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart — how many trades fall in each sentiment bucket."""
    apply_dark_style()

    counts = (
        df["Classification"]
        .value_counts()
        .reindex(SENTIMENT_ORDER)
        .dropna()
    )
    colors = sentiment_colors(counts.index.tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.6, edgecolor=GRID_COLOR)

    # Annotate bars with counts
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR
        )

    ax.set_title("Trade Count by Market Sentiment", pad=14)
    ax.set_xlabel("Sentiment Classification")
    ax.set_ylabel("Number of Trades")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "01_sentiment_distribution.png", output_dir)


def plot_daily_profit_trend(df: pd.DataFrame, output_dir: str) -> None:
    """Line chart — total daily PnL coloured by sentiment."""
    apply_dark_style()

    daily = (
        df.groupby(["Date", "Classification"])["closedPnL"]
        .sum()
        .reset_index()
    )
    # Overall daily aggregate for the line
    daily_total = df.groupby("Date")["closedPnL"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))

    # Background sentiment shading
    for _, row in daily.iterrows():
        ax.axvspan(
            row["Date"] - pd.Timedelta(hours=12),
            row["Date"] + pd.Timedelta(hours=12),
            alpha=0.08,
            color=SENTIMENT_PALETTE.get(row["Classification"], ACCENT),
        )

    ax.plot(
        daily_total["Date"], daily_total["closedPnL"],
        color=ACCENT, linewidth=1.5, label="Daily PnL"
    )
    ax.axhline(0, color="#ff4d4d", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("Daily Aggregate Closed PnL Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closed PnL (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "02_daily_profit_trend.png", output_dir)


def plot_pnl_by_sentiment_violin(df: pd.DataFrame, output_dir: str) -> None:
    """Violin + strip chart — full PnL distribution per sentiment."""
    apply_dark_style()

    # Clip extreme outliers for visual clarity (keep 1st–99th percentile)
    low, high = df["closedPnL"].quantile([0.01, 0.99])
    plot_df = df[df["closedPnL"].between(low, high)].copy()
    present = [s for s in SENTIMENT_ORDER if s in plot_df["Classification"].unique()]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.violinplot(
        data=plot_df, x="Classification", y="closedPnL",
        order=present, palette=SENTIMENT_PALETTE,
        inner="quartile", linewidth=1.2, ax=ax
    )
    ax.axhline(0, color="#ff4d4d", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_title("PnL Distribution by Market Sentiment (1st–99th Percentile)")
    ax.set_xlabel("Sentiment Classification")
    ax.set_ylabel("Closed PnL (USD)")
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "03_pnl_violin_by_sentiment.png", output_dir)


def plot_avg_pnl_by_sentiment(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart — mean closed PnL per trade for each sentiment."""
    apply_dark_style()

    avg = (
        df.groupby("Classification")["closedPnL"]
        .mean()
        .reindex(SENTIMENT_ORDER)
        .dropna()
    )
    colors = ["#2dc9a7" if v >= 0 else "#ef5675" for v in avg.values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(avg.index, avg.values, color=colors, width=0.6, edgecolor=GRID_COLOR)

    for bar, val in zip(bars, avg.values):
        ypos = val + (avg.abs().max() * 0.015) if val >= 0 else val - (avg.abs().max() * 0.05)
        ax.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"${val:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR
        )

    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    ax.set_title("Average Closed PnL per Trade by Sentiment")
    ax.set_xlabel("Sentiment Classification")
    ax.set_ylabel("Avg Closed PnL (USD)")
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "04_avg_pnl_by_sentiment.png", output_dir)


def plot_win_rate_by_sentiment(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart — win rate (%) per sentiment, with 50% reference line."""
    apply_dark_style()

    wr = (
        df.groupby("Classification")["win"]
        .mean()
        .reindex(SENTIMENT_ORDER)
        .dropna()
        .mul(100)
    )
    colors = sentiment_colors(wr.index.tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(wr.index, wr.values, color=colors, width=0.6, edgecolor=GRID_COLOR)

    for bar, val in zip(bars, wr.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR
        )

    ax.axhline(50, color="#ffa600", linewidth=1, linestyle="--", alpha=0.8, label="50% baseline")
    ax.set_ylim(0, min(wr.max() + 10, 100))
    ax.set_title("Win Rate by Market Sentiment")
    ax.set_xlabel("Sentiment Classification")
    ax.set_ylabel("Win Rate (%)")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "05_win_rate_by_sentiment.png", output_dir)


def plot_position_size_by_sentiment(df: pd.DataFrame, output_dir: str) -> None:
    """Box plot — position size (USD) by sentiment."""
    apply_dark_style()

    low, high = df["position_value"].quantile([0.01, 0.99])
    plot_df = df[df["position_value"].between(low, high)].copy()
    present = [s for s in SENTIMENT_ORDER if s in plot_df["Classification"].unique()]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=plot_df, x="Classification", y="position_value",
        order=present, palette=SENTIMENT_PALETTE,
        width=0.5, linewidth=1.2, fliersize=2, ax=ax
    )
    ax.set_title("Position Size (USD) by Market Sentiment")
    ax.set_xlabel("Sentiment Classification")
    ax.set_ylabel("Trade Size (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "06_position_size_by_sentiment.png", output_dir)


def plot_direction_by_sentiment(df: pd.DataFrame, output_dir: str) -> None:
    """Stacked 100% bar — proportion of Buy vs Sell trades per sentiment."""
    apply_dark_style()

    dir_counts = (
        df.groupby(["Classification", "direction"])
        .size()
        .unstack(fill_value=0)
    )
    dir_pct = dir_counts.div(dir_counts.sum(axis=1), axis=0) * 100
    dir_pct = dir_pct.reindex(SENTIMENT_ORDER).dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(dir_pct))

    dir_colors = {"Buy": "#2dc9a7", "Sell": "#ef5675", "Open Long": "#58a6ff",
                  "Open Short": "#ff764a", "Close Long": "#7adb78", "Close Short": "#ffa600"}

    for col in dir_pct.columns:
        vals = dir_pct[col].values
        color = dir_colors.get(col, ACCENT)
        ax.bar(dir_pct.index, vals, bottom=bottom, color=color, label=col, edgecolor=PANEL_BG, linewidth=0.5)
        # Annotate if wide enough
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white")
        bottom += vals

    ax.set_ylim(0, 100)
    ax.set_title("Trade Direction Mix by Market Sentiment")
    ax.set_xlabel("Sentiment Classification")
    ax.set_ylabel("Proportion (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "07_direction_by_sentiment.png", output_dir)


def plot_long_vs_short_pnl(df: pd.DataFrame, output_dir: str) -> None:
    """Side-by-side bar — avg PnL for Long vs Short trades per sentiment."""
    apply_dark_style()

    # Classify trades as Long or Short from the 'direction' column
    df2 = df.copy()
    df2["trade_type"] = df2["direction"].str.strip().str.lower().apply(
        lambda d: "Long" if "buy" in d or "long" in d else "Short"
    )

    avg_pnl = (
        df2.groupby(["Classification", "trade_type"])["closedPnL"]
        .mean()
        .unstack()
        .reindex(SENTIMENT_ORDER)
        .dropna(how="all")
    )

    x = np.arange(len(avg_pnl))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    if "Long" in avg_pnl.columns:
        ax.bar(x - width/2, avg_pnl["Long"], width, label="Long", color="#2dc9a7", edgecolor=PANEL_BG)
    if "Short" in avg_pnl.columns:
        ax.bar(x + width/2, avg_pnl["Short"], width, label="Short", color="#ef5675", edgecolor=PANEL_BG)

    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(avg_pnl.index, rotation=15, ha="right")
    ax.set_title("Avg PnL: Long vs Short Trades by Sentiment")
    ax.set_xlabel("Sentiment Classification")
    ax.set_ylabel("Avg Closed PnL (USD)")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "08_long_vs_short_pnl.png", output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Heatmap — Pearson correlation between key numeric features."""
    apply_dark_style()

    cols = ["closedPnL", "position_value", "sentiment_encoded", "win", "fee"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))      # show lower triangle only

    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # blue → red
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", linewidths=0.5, linecolor=DARK_BG,
        annot_kws={"size": 11}, ax=ax,
        vmin=-1, vmax=1, square=True,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    save_fig(fig, "09_correlation_heatmap.png", output_dir)


def plot_cumulative_pnl(df: pd.DataFrame, output_dir: str) -> None:
    """Line chart — cumulative sum of closed PnL over time."""
    apply_dark_style()

    daily = df.groupby("Date")["closedPnL"].sum().sort_index()
    cumulative = daily.cumsum()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(
        cumulative.index, cumulative.values,
        where=(cumulative.values >= 0), color="#2dc9a7", alpha=0.3, label="Positive"
    )
    ax.fill_between(
        cumulative.index, cumulative.values,
        where=(cumulative.values < 0), color="#ef5675", alpha=0.3, label="Negative"
    )
    ax.plot(cumulative.index, cumulative.values, color=ACCENT, linewidth=1.8)
    ax.axhline(0, color="#ff4d4d", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative Closed PnL Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    save_fig(fig, "10_cumulative_pnl.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(df: pd.DataFrame) -> dict:
    """
    Perform four statistical tests and return a results dictionary.

    Tests:
      1. Welch's T-Test  — Fear vs Greed PnL means
      2. One-Way ANOVA   — PnL across all sentiment categories
      3. Chi-Square      — Sentiment category × Win/Loss contingency
      4. Pearson Corr    — Sentiment encoding ↔ closedPnL

    Returns a dict with keys: 'ttest', 'anova', 'chisq', 'correlation'
    """
    results = {}

    # ── 1. Welch's T-Test ─────────────────────────────────────────────────────
    fear_pnl  = df[df["Classification"] == "Fear"]["closedPnL"]
    greed_pnl = df[df["Classification"] == "Greed"]["closedPnL"]

    if len(fear_pnl) > 1 and len(greed_pnl) > 1:
        t_stat, p_val = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
        results["ttest"] = {
            "groups": ("Fear", "Greed"),
            "n_fear": len(fear_pnl),
            "n_greed": len(greed_pnl),
            "mean_fear":  fear_pnl.mean(),
            "mean_greed": greed_pnl.mean(),
            "t_statistic": t_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
        }

    # ── 2. One-Way ANOVA ──────────────────────────────────────────────────────
    groups = [
        df[df["Classification"] == s]["closedPnL"].values
        for s in SENTIMENT_ORDER
        if s in df["Classification"].unique()
    ]
    groups = [g for g in groups if len(g) > 1]

    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        results["anova"] = {
            "f_statistic": f_stat,
            "p_value": p_anova,
            "significant": p_anova < 0.05,
        }

    # ── 3. Chi-Square Test ────────────────────────────────────────────────────
    contingency = pd.crosstab(df["Classification"], df["win"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
    results["chisq"] = {
        "chi2_statistic": chi2,
        "p_value": p_chi2,
        "degrees_of_freedom": dof,
        "significant": p_chi2 < 0.05,
        "contingency_table": contingency,
    }

    # ── 4. Correlation table ──────────────────────────────────────────────────
    corr_cols = ["closedPnL", "position_value", "sentiment_encoded", "win"]
    results["correlation"] = df[corr_cols].corr()

    return results


def print_statistical_summary(results: dict) -> None:
    """Pretty-print the statistical test results to stdout."""
    SEP = "─" * 60

    print(f"\n{'═'*60}")
    print("  STATISTICAL ANALYSIS SUMMARY")
    print(f"{'═'*60}")

    # T-Test
    if "ttest" in results:
        r = results["ttest"]
        sig = "✓ Significant (p < 0.05)" if r["significant"] else "✗ Not significant"
        print(f"\n1. Welch's T-Test — Fear vs Greed PnL")
        print(SEP)
        print(f"   n(Fear)={r['n_fear']:,}   n(Greed)={r['n_greed']:,}")
        print(f"   Mean PnL  Fear : ${r['mean_fear']:.4f}")
        print(f"   Mean PnL  Greed: ${r['mean_greed']:.4f}")
        print(f"   T-Statistic : {r['t_statistic']:.4f}")
        print(f"   P-Value     : {r['p_value']:.6f}")
        print(f"   Result      : {sig}")

    # ANOVA
    if "anova" in results:
        r = results["anova"]
        sig = "✓ Significant (p < 0.05)" if r["significant"] else "✗ Not significant"
        print(f"\n2. One-Way ANOVA — PnL across all Sentiment Categories")
        print(SEP)
        print(f"   F-Statistic : {r['f_statistic']:.4f}")
        print(f"   P-Value     : {r['p_value']:.6f}")
        print(f"   Result      : {sig}")

    # Chi-Square
    if "chisq" in results:
        r = results["chisq"]
        sig = "✓ Significant (p < 0.05)" if r["significant"] else "✗ Not significant"
        print(f"\n3. Chi-Square Test — Sentiment × Win/Loss Contingency")
        print(SEP)
        print(f"   Chi² Statistic : {r['chi2_statistic']:.4f}")
        print(f"   Degrees of Freedom : {r['degrees_of_freedom']}")
        print(f"   P-Value     : {r['p_value']:.6f}")
        print(f"   Result      : {sig}")

    # Correlation
    if "correlation" in results:
        print(f"\n4. Pearson Correlation Matrix")
        print(SEP)
        print(results["correlation"].round(4).to_string())

    print(f"\n{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRADER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def trader_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-trader performance metrics and assign one of three segments:
      High Performer  — top tertile by total PnL
      Medium Performer
      Low Performer   — bottom tertile

    Returns the trader-level summary DataFrame.
    """
    summary = (
        df.groupby("account")
        .agg(
            total_pnl       = ("closedPnL",      "sum"),
            avg_pnl_per_trade=("closedPnL",      "mean"),
            win_rate        = ("win",             "mean"),
            total_trades    = ("closedPnL",       "count"),
            avg_position_sz = ("position_value",  "mean"),
            favourite_sentiment = ("Classification",
                                   lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )

    # Only segment if enough traders exist for 3 quantile bins
    if len(summary) >= 3:
        summary["segment"] = pd.qcut(
            summary["total_pnl"],
            q=3,
            labels=["Low Performer", "Medium Performer", "High Performer"],
            duplicates="drop"
        )
    else:
        summary["segment"] = "Insufficient Data"

    return summary


def print_segment_summary(seg_df: pd.DataFrame) -> None:
    """Print per-segment averages."""
    if "segment" not in seg_df.columns:
        return
    summary = (
        seg_df.groupby("segment")
        .agg(
            traders         = ("account", "count"),
            avg_total_pnl   = ("total_pnl", "mean"),
            avg_win_rate    = ("win_rate", "mean"),
            avg_position_sz = ("avg_position_sz", "mean"),
            avg_trades      = ("total_trades", "mean"),
        )
    )
    print("\n" + "═"*60)
    print("  TRADER SEGMENTATION SUMMARY")
    print("═"*60)
    print(summary.round(2).to_string())
    print("═"*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_eda(df: pd.DataFrame, output_dir: str = "outputs/figures") -> dict:
    """
    Run all 10 visualisations + statistical tests + trader segmentation.
    Returns the statistical results dict for downstream use.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n── Generating visualisations ──")

    plot_sentiment_distribution(df, output_dir)
    plot_daily_profit_trend(df, output_dir)
    plot_pnl_by_sentiment_violin(df, output_dir)
    plot_avg_pnl_by_sentiment(df, output_dir)
    plot_win_rate_by_sentiment(df, output_dir)
    plot_position_size_by_sentiment(df, output_dir)
    plot_direction_by_sentiment(df, output_dir)
    plot_long_vs_short_pnl(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_cumulative_pnl(df, output_dir)

    print("\n── Running statistical tests ──")
    stats_results = run_statistical_tests(df)
    print_statistical_summary(stats_results)

    print("── Trader segmentation ──")
    seg_df = trader_segmentation(df)
    print_segment_summary(seg_df)

    return stats_results

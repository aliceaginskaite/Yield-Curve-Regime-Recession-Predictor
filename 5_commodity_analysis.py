
"""
 5_commodity_analysis.py

 What this file does:
   - Loads regimes, spreads, and commodity prices
   - Computes daily log returns for Gold and WTI
   - For each regime: mean return, volatility, sharpe, max drawdown
   - Tests statistical significance of return differences
   - Plots bar charts and cumulative returns colored by regime
   - Saves regime performance table for dashboard

"""

# 1. Imports
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

print("=" * 60)
print("5_commodity_analysis.py - regimes vs commodities")
print("=" * 60)


# 2. Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir,"data")
plots_dir = os.path.join(base_dir,"plots")
os.makedirs(plots_dir, exist_ok=True)


# 3. Load data
print("\nLoading data...")

spreads_regime = pd.read_csv(
    os.path.join(data_dir, "spreads_with_regime.csv"),
    index_col="Date",
    parse_dates=True,
)

commodities = pd.read_csv(
    os.path.join(data_dir, "commodities.csv"),
    index_col="Date",
    parse_dates=True,
)

print(f"spreads_with_regime: {spreads_regime.shape[0]} rows")
print(f"commodities: {commodities.shape[0]} rows")


"""# 4. Log returns
# We use log returns (not simple returns) for two reasons:
#
# - Log returns are additive over time:
#     log(P3/P1) = log(P3/P2) + log(P2/P1)
#     This makes cumulative return calculation clean.
# - Log returns are more normally distributed than simple returns,
#     which makes statistical tests (t-test) more valid.
#
# Formula: r_t = log(P_t / P_{t-1})
# For small returns log return ≈ simple return, so interpretation is the same."""

commodities["gold_return"] = np.log(commodities["Gold"] / commodities["Gold"].shift(1))
commodities["wti_return"] = np.log(commodities["WTI"] / commodities["WTI"].shift(1))

# drop the first row with nan retirn
commodities.dropna(subset=["gold_return", "wti_return"], inplace=True)

print(f"\nReturn stats (full sample):")
for col, label in [("gold_return", "Gold"), ("wti_return", "WTI")]:
    mean = commodities[col].mean() * 252 * 100 
    vol = commodities[col].std()  * np.sqrt(252) * 100
    print(f"  {label}: annualized return={mean:.1f}%  vol={vol:.1f}%")


"""# 5. Merge regimes with commodity returns
# Treasuries and commodities trade on slightly different calendars. US bond market closes on some days 
# when equity/futures markets stay open
# 
#   Our approach - outer join + forward fill:
#   Keep all dates from both sources.
#   Fill missing regime with the last known regime (ffill).
#   If the bond market was closed Monday, the regime didn't change.
#   Fill missing commodity price with the last known price (ffill) -
#   standard practice for non-trading days.
#   Then drop rows where either is still NaN."""

regime_col = spreads_regime[["regime"]].dropna()

merged = regime_col.join(
    commodities[["Gold", "WTI", "gold_return", "wti_return"]],
    how="outer",
)

# forward fill regime and prices
merged["regime"] = merged["regime"].ffill()
merged["Gold"] = merged["Gold"].ffill()
merged["WTI"] = merged["WTI"].ffill()

# returns cannot be forward filled so we drop those rows
merged.dropna(subset=["gold_return", "wti_return", "regime"], inplace=True)

n_inner = len(regime_col.join(
    commodities[["Gold"]], how="inner").dropna())
print(f"Outer join: {len(merged)} days vs inner join would give: {n_inner} days"
      f"(recovered {len(merged) - n_inner} days)")

print(f"\nMerged dataset: {len(merged)} days")
print(f"Regime coverage:")
for r, n in merged["regime"].value_counts().items():
    print(f"{r:<10} {n} days ({n/len(merged)*100:.1f}%)")


"""# ── 6. PERFORMANCE METRICS PER REGIME ────────────────────────
# For each regime we compute a full performance table
#
# Metrics:
#   mean_daily_return - average log return per day (raw)
#   ann_return - annualized return (× 252 trading days)
#   ann_volatility - annualized volatility (std × sqrt(252))
#   sharpe - ann_return / ann_volatility (no risk-free rate) this is a simplified Sharpe, sufficient for comparison
#   max_drawdown - largest peak-to-trough loss within this regime
#   n_days - number of trading days in this regime"""

regimes_ordered = ["Steep", "Flat", "Humped", "Inverted"]

def max_drawdown(returns):
    """
    Compute maximum drawdown from a series of log returns.
    Drawdown = largest % drop from a rolling peak in cumulative wealth.
    Returns a negative number (e.g. -0.35 means -35% drawdown).
    """
    cum = np.exp(returns.cumsum())    
    rolling_peak = cum.cummax()   
    drawdown = (cum - rolling_peak) / rolling_peak
    return drawdown.min()          

results = {}

for commodity, ret_col in [("Gold", "gold_return"), ("WTI", "wti_return")]:
    results[commodity] = {}
    for regime in regimes_ordered:
        mask = merged["regime"] == regime
        ret_ser = merged.loc[mask, ret_col]

        if len(ret_ser) < 10:
            continue

        ann_ret = ret_ser.mean() * 252
        ann_vol = ret_ser.std()  * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        mdd = max_drawdown(ret_ser)
        n = len(ret_ser)

        results[commodity][regime] = {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "n_days": n,
        }

# print performance tables
for commodity in ["Gold", "WTI"]:
    print(f"\n{commodity} performance by yield curve regime")
    print(f"{'Regime':<12} {'Ann Ret':>9} {'Ann Vol':>9}"
          f"{'Sharpe':>8} {'Max DD':>9} {'Days':>6}")
    print("-" * 58)
    for regime in regimes_ordered:
        if regime not in results[commodity]:
            continue
        r = results[commodity][regime]
        print(f"  {regime:<10} "
              f"{r['ann_return']*100:>8.1f}% "
              f"{r['ann_vol']*100:>8.1f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['max_drawdown']*100:>8.1f}% "
              f"{r['n_days']:>6}")


"""# 7. Statistical significance
# Are the return differences between regimes statistically significant or random noise
#
# We use a two-sample t-test:
#   H0 (null hypothesis): returns in regime A == returns in regime B
#   If p-value < 0.05 → reject H0 → difference is significant
#
# We compare Inverted vs Steep for both commodities - the most extreme contrast."""

print("\nT-test: Inverted vs Steep returns")
for commodity, ret_col in [("Gold", "gold_return"), ("WTI", "wti_return")]:
    inv_returns = merged.loc[merged["regime"] == "INVERTED", ret_col]
    steep_returns = merged.loc[merged["regime"] == "STEEP",    ret_col]

    t_stat, p_val = stats.ttest_ind(inv_returns, steep_returns)
    sig = "+ significant" if p_val < 0.05 else "! not significant"
    print(f"{commodity}: t={t_stat:.2f} p={p_val:.4f} -> {sig}")


""" # 8: Plot bar charts + cumulative returns
# Four panels:
#   Top left - Gold annualized return by regime (bar chart)
#   Top right - WTI annualized return by regime (bar chart)
#   Bottom left - Gold cumulative return, colored by regime
#   Bottom right - WTI cumulative return, colored by regime"""

regime_colors = {
    "Steep": "#16A34A",
    "Flat": "#F59E0B",
    "Humped" : "#8B5CF6",
    "Inverted": "#EF4444",
}

print("\nBuilding charts...")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#FAFAFA")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

ax_bar_gold = fig.add_subplot(gs[0, 0])
ax_bar_wti = fig.add_subplot(gs[0, 1])
ax_cum_gold = fig.add_subplot(gs[1, 0])
ax_cum_wti = fig.add_subplot(gs[1, 1])

for ax in (ax_bar_gold, ax_bar_wti, ax_cum_gold, ax_cum_wti):
    ax.set_facecolor("#FAFAFA")
    ax.spines[["top", "right"]].set_visible(False)

# bar charts
for ax, commodity in [(ax_bar_gold, "Gold"), (ax_bar_wti, "WTI")]:
    ann_returns = [
        results[commodity][r]["ann_return"] * 100
        for r in regimes_ordered
        if r in results[commodity]
    ]
    colors = [regime_colors[r] for r in regimes_ordered if r in results[commodity]]
    x_pos = np.arange(len(ann_returns))

    bars = ax.bar(x_pos, ann_returns, color=colors, alpha=0.85, width=0.6, zorder=2)

    # value labels on top of each bar
    for bar, val in zip(bars, ann_returns):
        va = "bottom" if val >= 0 else "top"
        ypos = val + 0.5 if val >= 0 else val - 0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:.1f}%",
            ha="center", va=va, fontsize=9, fontweight="bold",
        )

    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [r for r in regimes_ordered if r in results[commodity]],
        fontsize=9,
    )
    ax.set_ylabel("Annualized return (%)", fontsize=10)
    ax.set_title(
        f"{commodity} - annualized return by yield curve regime",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

# cumulative return
for ax, commodity, ret_col in [
    (ax_cum_gold, "Gold", "gold_return"),
    (ax_cum_wti,  "WTI",  "wti_return"),
]:
    # cumulative log return
    cum_returns = merged[ret_col].cumsum()

    for regime in regimes_ordered:
        mask = merged["regime"] == regime
        subset = cum_returns[mask]

        # scatter for regime color
        ax.scatter(
            subset.index,
            subset.values * 100,
            color=regime_colors[regime],
            s=0.8,
            alpha=0.5,
            zorder=2,
        )

    # overlay a clean total line in dark
    ax.plot(
        cum_returns.index,
        cum_returns.values * 100,
        color="#1E293B",
        linewidth=0.8,
        alpha=0.4,
        zorder=1,
    )

    ax.set_ylabel("Cumulative log return (%)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title(
        f"{commodity} - cumulative return, colored by regime",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

# shared legend for regime colors
patches = [
    mpatches.Patch(color=c, label=r, alpha=0.85)
    for r, c in regime_colors.items()
]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=4,
    fontsize=10,
    framealpha=0.8,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Gold & WTI performance across yield curve regimes  (1990 - present)",
    fontsize=13, fontweight="bold",
)

plot_path = os.path.join(plots_dir, "4_commodity_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
# plt.show()
print(f"Chart saved -> {plot_path}")


# 9. Save results
rows = []
for commodity in ["Gold", "WTI"]:
    for regime in regimes_ordered:
        if regime not in results[commodity]:
            continue
        r = results[commodity][regime]
        rows.append({
            "commodity": commodity,
            "regime": regime,
            "ann_return" : round(r["ann_return"], 4),
            "ann_vol" : round(r["ann_vol"], 4),
            "sharpe": round(r["sharpe"], 3),
            "max_drawdown": round(r["max_drawdown"], 4),
            "n_days": r["n_days"],
        })

out_df = pd.DataFrame(rows)
out_path = os.path.join(data_dir, "commodity_regime_performance.csv")
out_df.to_csv(out_path, index=False)
print(f"\ncommodity_regime_performance.csv saved -> {out_path}")

print("\n" + "=" * 60)
print("Commodity analysis complete.")
print("=" * 60)


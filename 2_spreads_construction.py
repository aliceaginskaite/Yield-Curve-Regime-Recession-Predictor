"""
 2_spreads_construction.py
 
 What this file does:
   - Loads treasury yields saved in 01_data_collection.py
   - Builds three yield curve spreads: 2Y-10Y, 3M-10Y, 2Y-30Y
   - Adds rolling averages to smooth noise
   - Plots all three spreads with NBER recession shading
   - Saves the spread dataframe for the next scripts
"""

# 1. Imports 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

print("2_spreads_construction.py - building spreads")
print("=" * 60)


# 2. Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir,"data")
plots_dir = os.path.join(base_dir,"plots")
os.makedirs(plots_dir, exist_ok=True)


"""#  3. Load data
# Load treasuries - daily frequency, columns: 3M, 2Y, 10Y, 30Y
# Load recession - monthly frequency, column: recession (0 or 1)"""

print("\nLoading data from data/ folder...")

treasuries = pd.read_csv(
    os.path.join(data_dir, "treasuries.csv"),
    index_col="Date",
    parse_dates=True,
)

recession = pd.read_csv(
    os.path.join(data_dir, "recession.csv"),
    index_col="Date",
    parse_dates=True,
)

print(f"treasuries: {treasuries.shape[0]} rows, {treasuries.shape[1]} columns")
print(f"recession: {recession.shape[0]} rows")


"""#  4. Align date range
# Treasuries start 1990-01-02, recession starts 1990-01-01.
# We align to the common range so plots line up cleanly."""

start = max(treasuries.index.min(), recession.index.min())
end = min(treasuries.index.max(), recession.index.max())

treasuries = treasuries.loc[start:end]

print(f"\nAligned period: {start.date()} -> {end.date()}")


"""# 5. Building the three spreads
# A yield spread = long-term yield minus short-term yield.
# When spread >0 = curve is normal (steep or flat)
# When spread <0 = curve is inverted - historically precedes recessions
#
# We build three spreads:
#
# 2Y-10Y - classic benchmark spread.
# The Fed watches this closely. When the Fed hikes rates
# aggressively, 2Y rises faster than 10Y -> inversion.
# Best known predictor of recessions 12-18 months ahead.
#
# 3M-10Y - the Fed's preferred spread (per NY Fed research).
# 3M is almost a direct proxy for the Fed Funds Rate.
# This spread has the best statistical track record for
# predicting recessions at the 12-month horizon.
#
# 2Y-30Y - the long-end spread.
# Captures the slope across the full curve.
# 30Y is driven by long-term inflation expectations and
# demand from pension funds/insurance companies (ALM).
# Less sensitive to Fed actions, more to structural factors.
"""

spreads = pd.DataFrame(index=treasuries.index)

spreads["spread_2y10y"] = treasuries["10Y"] - treasuries["2Y"]
spreads["spread_3m10y"] = treasuries["10Y"] - treasuries["3M"]
spreads["spread_2y30y"] = treasuries["30Y"] - treasuries["2Y"]

print("\nSpreads built:")
print(f"spread_2y10y range: {spreads['spread_2y10y'].min():.2f} -> {spreads['spread_2y10y'].max():.2f} pp")
print(f"spread_3m10y range: {spreads['spread_3m10y'].min():.2f} -> {spreads['spread_3m10y'].max():.2f} pp")
print(f"spread_2y30y range: {spreads['spread_2y30y'].min():.2f} -> {spreads['spread_2y30y'].max():.2f} pp")


#  6. Rolling averages 
# Raw daily spreads are noisy (small market moves create spikes
# that don't reflect real regime changes).
# We add two rolling means to smooth the signal

# 30-day (medium smoothing), 90-day (strong smoothing)


for col in ["spread_2y10y", "spread_3m10y", "spread_2y30y"]:
    spreads[f"{col}_ma30"] = spreads[col].rolling(window=30,min_periods=15).mean()
    spreads[f"{col}_ma90"] = spreads[col].rolling(window=90,min_periods=45).mean()

# daily first differences
for col in ["spread_2y10y", "spread_3m10y", "spread_2y30y"]:
    spreads[f"{col}_diff"] = spreads[col].diff()

print(f"\nFinal spreads dataframe: {spreads.shape[0]} rows × {spreads.shape[1]} columns")
print(spreads[["spread_2y10y", "spread_3m10y", "spread_2y30y"]].tail(3).to_string())


"""# 7. Recession bands 
# For plotting, we need the start and end date of each recession period.
# NBER data is monthly (0/1). We find where it transitions 0→1 (start)
# and 1→0 (end) to get date pairs for shading the charts."""

# Resample recession to daily
recession_daily = recession.reindex(spreads.index, method="ffill")

# recession start and end dates
rec_values = recession_daily["recession"]
rec_starts = rec_values[(rec_values == 1) & (rec_values.shift(1) == 0)].index.tolist()
rec_ends = rec_values[(rec_values == 0) & (rec_values.shift(1) == 1)].index.tolist()

# If data ends during a recession, close the last band at the end
if len(rec_starts) > len(rec_ends):
    rec_ends.append(spreads.index[-1])

recession_bands = list(zip(rec_starts, rec_ends))
print(f"\nRecession periods found: {len(recession_bands)}")
for s, e in recession_bands:
    print(f"{s.date()} -> {e.date()}")


"""#  8. Plotting spreads
# One chart with three subplots stacked vertically.
# Each subplot shows:
# raw spread (thin, semi-transparent)
# 90-day rolling average (thick, solid)
# zero line (dashed, red) - inversion threshold
# grey shaded bands = NBER recessions"""

print("\nBuilding chart...")

# colors
color_2y10y = "#2563EB"   
color_3m10y = "#16A34A"   
color_2y30y = "#9333EA"   
color_rec = "#D1D5DB"  

spread_configs = [
    {
      "col": "spread_2y10y",
      "color": color_2y10y,
      "label": "2Y–10Y Spread",
      "note": "Classic Fed benchmark - inverted before every recession since 1980",
    },
    {
      "col": "spread_3m10y",
      "color": color_3m10y,
      "label": "3M–10Y Spread",
      "note": "NY Fed preferred - strongest statistical predictor at 12-month horizon",
    },
    {
      "col": "spread_2y30y",
      "color": color_2y30y,
      "label": "2Y–30Y Spread",
      "note": "Long-end slope - reflects structural inflation and pension demand",
    },
]

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
fig.patch.set_facecolor("#FAFAFA")

for ax, cfg in zip(axes, spread_configs):
    col = cfg["col"]
    color = cfg["color"]

    ax.set_facecolor("#FAFAFA")

    # shade recession bands
    for rec_start, rec_end in recession_bands:
        ax.axvspan(rec_start, rec_end, color=color_rec, alpha=0.6, zorder=0)

    # raw spread (thin and semi transparent)
    ax.plot(
        spreads.index,
        spreads[col],
        color=color,
        linewidth=0.6,
        alpha=0.35,
        zorder=1,
    )

    # 90 day rolling average main visible line
    ax.plot(
        spreads.index,
        spreads[f"{col}_ma90"],
        color=color,
        linewidth=1.8,
        alpha=0.95,
        zorder=2,
        label="90-day MA",
    )

    # zero line inversion threshold
    ax.axhline(0, color="#EF4444", linewidth=1.0, linestyle="--", alpha=0.8, zorder=3)

    # fill below zero in red to make inversions visually obvious
    ax.fill_between(
        spreads.index,
        spreads[f"{col}_ma90"],
        0,
        where=(spreads[f"{col}_ma90"] < 0),
        color="#EF4444",
        alpha=0.15,
        zorder=1,
    )

    # labels and formatting
    ax.set_ylabel("Spread (pp)", fontsize=10)
    ax.set_title(
        f"{cfg['label']}",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax.annotate(
        cfg["note"],
        xy=(0.01, 0.06),
        xycoords="axes fraction",
        fontsize=8.5,
        color="#6B7280",
    )

    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # legend
    raw_patch = mpatches.Patch(color=color,alpha=0.35, label="Daily spread")
    ma_patch = mpatches.Patch(color=color,alpha=0.95, label="90-day MA")
    zero_patch = mpatches.Patch(color="#EF4444",alpha=0.80, label="Zero (inversion threshold)")
    rec_patch = mpatches.Patch(color=color_rec,alpha=0.60, label="NBER recession")
    ax.legend(handles=[raw_patch, ma_patch, zero_patch, rec_patch],
              loc="upper right", fontsize=8, framealpha=0.7)

# shared x-axis label and title
axes[-1].set_xlabel("Date", fontsize=10)
fig.suptitle(
    "US Treasury Yield Curve Spreads (1990 – present)\n"
    "Red = inverted|Grey bands = NBER recessions",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plot_path = os.path.join(plots_dir, "1_spreads.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Chart saved -> {plot_path}")


# 9. Small summary stats 
# print a compact summary

print("\nSpread summary")
print(f"{'Spread':<15} {'Current':>9} {'Min':>8} {'Max':>8} {'Days<0':>8} {'%Inverted':>10}")
print("-" * 62)

for cfg in spread_configs:
    col = cfg["col"]
    series = spreads[col].dropna()
    current = series.iloc[-1]
    minimum = series.min()
    maximum = series.max()
    n_inv = (series < 0).sum()
    pct_inv = n_inv / len(series) * 100
    print(f"{cfg['label']:<15} {current:>9.2f} {minimum:>8.2f} {maximum:>8.2f} {n_inv:>8d} {pct_inv:>9.1f}%")


#  10. Save spreads
spreads_path = os.path.join(data_dir, "spreads.csv")
spreads.to_csv(spreads_path)
print(f"\nspreads.csv saved to {spreads_path}")

print("\n" + "=" * 60)
print("Spreads are built and saved.")
print("=" * 60)
"""
 6_composite_signal.py

 What this file does:
   - Loads spreads and AUC results from previous files
   - Builds a weighted composite signal from all three spreads
   - Weights are proportional to each spread's AUC at 12m horizon
   - Normalizes, smooths, and thresholds into a traffic light signal
   - Backtests: does the signal reduce false alarms vs single spreads?
   - Plots the composite signal with recession overlay
   - Saves the final signal dataframe for the dashboard

"""

# 1. Imports
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print("=" * 60)
print("6_composite_signal.py - building composite signal")
print("=" * 60)


# 2. Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)


# 3. Load data
print("\nLoading data...")

spreads = pd.read_csv(
    os.path.join(data_dir,"spreads.csv"),
    index_col="Date",
    parse_dates=True,
)

recession = pd.read_csv(
    os.path.join(data_dir, "recession.csv"),
    index_col="Date",
    parse_dates=True,
)

predictive_power = pd.read_csv(
    os.path.join(data_dir, "spread_predictive_power.csv"),
)

print(f"spreads: {spreads.shape[0]} rows")
print(f"recession: {recession.shape[0]} rows")
print(f"predictive_power: {len(predictive_power)} rows")
print("\nPredictive power summary:")
print(predictive_power.to_string(index=False))


"""# 4. Expanding window AUC weights
# One approach has look-ahead bias:
Full-sample AUC (1990-2026) used to weight signals at every point, including 1995.
The model would "know" 2008 and 2020 recessions when computing predictive power.
This artificially inflates backtest performance.

Better one - expanding window:
At each month T, AUC is computed using only data available up to T.
A 60-month warmup period is required before weights are trusted.
Prior to that, equal weighting (1/3 each) is used as fallback.

This is the honest, production-ready implementation.."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

spread_cols = ["spread_2y10y", "spread_3m10y", "spread_2y30y"]

# resample spreads to monthly for AUC computation 
spreads_m = spreads[spread_cols].resample("ME").mean()
spreads_m.index = spreads_m.index.to_period("M").to_timestamp()

recession_m = recession.copy()
recession_m.index = recession_m.index.to_period("M").to_timestamp()

combined_m = spreads_m.join(recession_m, how="inner").dropna()

horizon_months = 12  
min_warmup = 60  

print("\nComputing expanding-window AUC weights...")
print(f"Warmup period: {min_warmup} months | Forecast horizon: {horizon_months} months")

# for each month store the weight of each spread
expanding_weights = pd.DataFrame(
    index=combined_m.index,
    columns=spread_cols,
    dtype=float,
)

for i, date in enumerate(combined_m.index):
    # only data available up to t his date
    history = combined_m.loc[combined_m.index <= date]

    if len(history) < min_warmup + horizon_months:
        expanding_weights.loc[date] = 1.0 / 3.0
        continue

    # AUC for each spread
    auc_values = {}
    target = history["recession"].shift(-horizon_months).dropna()
    valid_idx = target.index

    for col in spread_cols:
        x = history.loc[valid_idx, col].values.reshape(-1, 1)
        y = target.values.astype(int)

        if len(np.unique(y)) < 2:
            auc_values[col] = 0.5 
            continue

        sc = StandardScaler()
        x_s = sc.fit_transform(x)
        mdl = LogisticRegression(random_state=42)
        mdl.fit(x_s, y)
        y_prob = mdl.predict_proba(x_s)[:, 1]
        auc_values[col] = roc_auc_score(y, y_prob)

    # normalize AUC values to weights that sum to 1
    total = sum(auc_values.values())
    for col in spread_cols:
        expanding_weights.loc[date, col] = auc_values[col] / total


expanding_weights_daily = expanding_weights.reindex(
    spreads.index, method="ffill"
).astype(float)

# print the final weights for inspection
final_weights = expanding_weights.iloc[-1]
print("\nMost recent expanding-window weights:")
for col in spread_cols:
    print(f"{col:<16} weight={final_weights[col]:.3f} ({final_weights[col]*100:.1f}%)")

print("\nComparison - static (biased) weights from file 4:")
auc_12m = predictive_power.set_index("spread")["auc_12m"]
static_total = auc_12m.sum()
for col in spread_cols:
    w = auc_12m[col] / static_total
    print(f"  {col:<16}  weight={w:.3f}  ({w*100:.1f}%)")


"""# 5. Expanding window z-score normalization
# Wrong - look-ahead bias:
z = (x - full_sample_mean) / full_sample_std
Mean and std computed in 1995 would include data from 2008 and 2020.

Correct - expanding window z-score:
At each date T, normalize using mean and std of data available up to T.
pandas .expanding().mean() and .expanding().std() implement this cleanly.
A minimum of 252 days (1 year) is required before normalisation begins."""

signal_df = pd.DataFrame(index=spreads.index)

for col in spread_cols:
    series = spreads[col]

    # expanding mean and std
    exp_mean = series.expanding(min_periods=252).mean()
    exp_std  = series.expanding(min_periods=252).std()

    # z 
    z_col = f"{col}_z"
    signal_df[z_col] = (series - exp_mean) / exp_std

print(f"\nExpanding-window z-scores computed (no look-ahead bias)")
print(f"Warmup: 252 days before first valid z-score")


"""# 6. Weighted composite signal
# Composite = sum of (expanding_weight[t] × z-score[t]) for each spread
#
# Both the weights and the z-scores are now computed without
# looking at future data (so bias-free).
#
# The result is a single number per day:
#   Large positive - curve is very steep (healthy economy signal)
#   Near zero - flat curve (late cycle, caution)
#   Large negative - curve is inverted (recession warning)"""

signal_df["composite_raw"] = sum(
    expanding_weights_daily[col] * signal_df[f"{col}_z"]
    for col in spread_cols
)

# 90-day 
signal_df["composite"] = (
    signal_df["composite_raw"]
    .rolling(window=90, min_periods=45)
    .mean()
)

# also faster 30-day version
signal_df["composite_30d"] = (
    signal_df["composite_raw"]
    .rolling(window=30, min_periods=15)
    .mean()
)

signal_df.dropna(subset=["composite"], inplace=True)

print(f"\nComposite signal stats:")
print(f"min: {signal_df['composite'].min():.3f}")
print(f"max: {signal_df['composite'].max():.3f}")
print(f"mean: {signal_df['composite'].mean():.3f}")
print(f"current: {signal_df['composite'].iloc[-1]:.3f}")


"""# ── 7. TRAFFIC LIGHT THRESHOLDS ──────────────────────────────
# We convert the continuous composite signal into three zones:

GREEN (> threshold_green): healthy curve - no recession signal.
YELLOW (threshold_red < comp ≤ threshold_green): flattening/mild inversion (watch).
RED (≤ threshold_red): deep inversion - historically precedes recessions.

Thresholds: 33rd and 10th percentiles of the composite signal distribution.
RED captures the bottom 10% of historical readings.."""

threshold_green = signal_df["composite"].quantile(0.33)
threshold_red = signal_df["composite"].quantile(0.10)

print(f"\nTraffic light thresholds:")
print(f"GREEN: composite > {threshold_green:.3f} (above 33rd percentile)")
print(f"YELLOW: {threshold_red:.3f} to {threshold_green:.3f}")
print(f"RED: composite <= {threshold_red:.3f} (below 10th percentile)")

def traffic_light(value):
    if pd.isna(value):
        return "UNKNOWN"
    if value > threshold_green:
        return "GREEN"
    elif value <= threshold_red:
        return "RED"
    else:
        return "YELLOW"

signal_df["signal"] = signal_df["composite"].apply(traffic_light)

# current signal
current_signal = signal_df["signal"].iloc[-1]
current_value = signal_df["composite"].iloc[-1]
print(f"\nCurrent signal: {current_signal} (composite = {current_value:.3f})")

signal_dist = signal_df["signal"].value_counts()
print("\nSignal distribution:")
for sig, count in signal_dist.items():
    pct = count / len(signal_df) * 100
    print(f"  {sig:<8}  {count} days  ({pct:.1f}%)")


"""# 8. False alarm analysis
# Does the composite signal produce fewer false alarms than
# any single spread used alone?
#
# False alarm = signal goes RED but no recession follows within the next 18 months.
#
# We compare:
#  composite signal (RED = below 10th percentile)
#  each individual spread (RED = below zero, i.e. inverted)
#
# Lower false alarm rate = better signal quality."""

print("\nFalse alarm analysis (signal RED, no recession within 18 months")

# align recession to daily by forward-filling
recession_daily = recession.reindex(signal_df.index, method="ffill")

window_days = 390

def false_alarm_rate(red_mask, recession_series, window):
    """
    For each day flagged RED, check if a recession occurs
    within the next `window` trading days.
    Returns the false alarm rate (0 to 1).
    """
    false_alarms = 0
    true_signals = 0
    red_dates = red_mask[red_mask].index

    for date in red_dates:
        future_idx = recession_series.index
        future_idx = future_idx[(future_idx > date) &
                                (future_idx <= date + pd.Timedelta(days=window * 1.5))]
        future_rec = recession_series.loc[future_idx]
        if future_rec.empty or future_rec["recession"].max() == 0:
            false_alarms += 1
        else:
            true_signals += 1

    total = false_alarms + true_signals
    return false_alarms / total if total > 0 else np.nan

# composite signal false alarm rate
composite_red = signal_df["signal"] == "RED"
fa_composite = false_alarm_rate(composite_red, recession_daily, window_days)
print(f"Composite signal: {fa_composite:.1%} false alarm rate")

# individual spread false alarm rates (inverted = below zero)
for col in spread_cols:
    spread_red = spreads[col].reindex(signal_df.index) < 0
    fa_spread  = false_alarm_rate(spread_red, recession_daily, window_days)
    print(f"  {col:<16} : {fa_spread:.1%} false alarm rate")


# 9. Recession bands helper
rec_values = recession_daily["recession"]
rec_starts = rec_values[(rec_values == 1) & (rec_values.shift(1) == 0)].index.tolist()
rec_ends = rec_values[(rec_values == 0) & (rec_values.shift(1) == 1)].index.tolist()
if len(rec_starts) > len(rec_ends):
    rec_ends.append(signal_df.index[-1])
recession_bands = list(zip(rec_starts, rec_ends))


"""# 10. Plot Compose signal 
# Three panels stacked vertically:
#  Top - composite signal line with colored background zones
#  Middle - traffic light bar (GREEN/YELLOW/RED per day)
#  Bottom - 2Y-10Y spread for reference"""

signal_colors = {
    "GREEN": "#16A34A",
    "YELLOW": "#F59E0B",
    "RED": "#EF4444",
}

print("\nBuilding composite signal chart...")

fig, axes = plt.subplots(
    3, 1, figsize=(18, 12),
    gridspec_kw={"height_ratios": [4, 1, 2]},
    sharex=True,
)
fig.patch.set_facecolor("#FAFAFA")

ax_sig, ax_light, ax_spread = axes

for ax in axes:
    ax.set_facecolor("#FAFAFA")
    ax.spines[["top", "right"]].set_visible(False)

# top panel

# colored background zones
ax_sig.axhspan(threshold_green, signal_df["composite"].max() + 0.5,
               color="#16A34A", alpha=0.07, zorder=0)
ax_sig.axhspan(threshold_red, threshold_green,
               color="#F59E0B", alpha=0.07, zorder=0)
ax_sig.axhspan(signal_df["composite"].min() - 0.5, threshold_red,
               color="#EF4444", alpha=0.07, zorder=0)

# threshold lines
ax_sig.axhline(threshold_green, color="#16A34A", linewidth=0.8,
               linestyle="--", alpha=0.6)
ax_sig.axhline(threshold_red,   color="#EF4444", linewidth=0.8,
               linestyle="--", alpha=0.6)
ax_sig.axhline(0, color="#9CA3AF", linewidth=0.6, linestyle=":", alpha=0.5)

# recession shading
for rs, re in recession_bands:
    ax_sig.axvspan(rs, re, color="#D1D5DB", alpha=0.5, zorder=0)

# 30-day composite
ax_sig.plot(
    signal_df.index,
    signal_df["composite_30d"],
    color="#94A3B8",
    linewidth=0.8,
    alpha=0.5,
    zorder=1,
    label="30-day MA",
)

# 90-day
ax_sig.plot(
    signal_df.index,
    signal_df["composite"],
    color="#1E293B",
    linewidth=1.8,
    alpha=0.9,
    zorder=2,
    label="90-day MA (main signal)",
)

ax_sig.set_ylabel("Composite z-score", fontsize=10)
ax_sig.set_title(
    "Yield Curve Composite Signal - weighted combination of 2Y-10Y, 3M-10Y, 2Y-30Y",
    fontsize=12, fontweight="bold", loc="left", pad=8,
)

# annotations for zones
ax_sig.annotate("GREEN zone", xy=(0.01, 0.92), xycoords="axes fraction",
                color="#16A34A", fontsize=8.5, fontweight="bold")
ax_sig.annotate("YELLOW zone", xy=(0.01, 0.48), xycoords="axes fraction",
                color="#B45309", fontsize=8.5, fontweight="bold")
ax_sig.annotate("RED zone", xy=(0.01, 0.06), xycoords="axes fraction",
                color="#EF4444", fontsize=8.5, fontweight="bold")

ax_sig.legend(loc="upper right", fontsize=8.5, framealpha=0.7)
ax_sig.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)

# middle panel
for sig_name, color in signal_colors.items():
    mask = signal_df["signal"] == sig_name
    ax_light.fill_between(
        signal_df.index, 0, 1,
        where=mask,
        color=color,
        alpha=0.85,
        zorder=2,
    )

for rs, re in recession_bands:
    ax_light.axvspan(rs, re, color="#000000", alpha=0.10, zorder=3)

ax_light.set_yticks([])
ax_light.set_ylabel("Signal", fontsize=9)
ax_light.spines["left"].set_visible(False)

#bottom panel
spread_ref = spreads["spread_2y10y"].reindex(signal_df.index)

ax_spread.plot(
    signal_df.index,
    spread_ref,
    color="#2563EB",
    linewidth=1.0,
    alpha=0.7,
)
ax_spread.axhline(0, color="#EF4444", linewidth=0.8, linestyle="--", alpha=0.7)
ax_spread.fill_between(
    signal_df.index, spread_ref, 0,
    where=(spread_ref < 0),
    color="#EF4444", alpha=0.15,
)

for rs, re in recession_bands:
    ax_spread.axvspan(rs, re, color="#D1D5DB", alpha=0.5, zorder=0)

ax_spread.set_ylabel("2Y–10Y (pp)", fontsize=10)
ax_spread.set_xlabel("Date", fontsize=10)
ax_spread.set_title("2Y–10Y spread (reference)",
                    fontsize=10, loc="left", pad=4)
ax_spread.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(plots_dir, "5_composite_signal.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
# plt.show()
print(f"  Chart saved → {plot_path}")


# 11. Save composite signal
out_path = os.path.join(data_dir, "composite_signal.csv")
signal_df.to_csv(out_path)
print(f"\ncomposite_signal.csv saved -> {out_path}")

print("\n" + "=" * 60)
print("Composite signal built and saved.")
print("=" * 60)

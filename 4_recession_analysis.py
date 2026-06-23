"""
 recession_analysis.py

 What this file does:
   - Loads spreads and NBER recession data
   - Resamples both to monthly frequency for lag analysis
   - Computes cross-correlation: spread today vs recession in X months
   - Fits logistic regression to predict recession probability
   - Measures ROC-AUC for each spread at different forecast horizons
   - Plots cross-correlation curves and ROC curves
   - Saves a summary table: best lag and AUC per spread

"""

# 1. Imports
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 60)
print(" 4_recession_analysis.py - predictive power of spreads")
print("=" * 60)


# 2. Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)


# 3. Load data
print("\nLoading data...")

spreads = pd.read_csv(
    os.path.join(data_dir, "spreads.csv"),
    index_col="Date",
    parse_dates=True,
)

recession = pd.read_csv(
    os.path.join(data_dir, "recession.csv"),
    index_col="Date",
    parse_dates=True,
)

print(f"spreads: {spreads.shape[0]} rows (daily)")
print(f"recession: {recession.shape[0]} rows (monthly)")


"""# 4. Resample to montly
# Spreads are daily, recession indicator is monthly.
# For lag analysis we need both on the same frequency.
# We resample spreads to monthly using end-of-month mean.
#
# Why mean and not last value:
# Mean reduces the effect of a single noisy day at month-end.
# For a signal that moves slowly (yield spreads) this is acceptable."""

spread_cols = ["spread_2y10y", "spread_3m10y", "spread_2y30y"]

spreads_monthly = spreads[spread_cols].resample("ME").mean()

# align both series to the same date range
start = max(spreads_monthly.index.min(), recession.index.min())
end = min(spreads_monthly.index.max(), recession.index.max())

spreads_monthly = spreads_monthly.loc[start:end]
recession_m     = recession.loc[start:end]

# normalize both indices to month start so they align
spreads_monthly.index = spreads_monthly.index.to_period("M").to_timestamp()
recession_m.index     = recession_m.index.to_period("M").to_timestamp()

combined = spreads_monthly.join(recession_m, how="inner")
combined.dropna(inplace=True)

print(f"\nMonthly aligned dataset: {len(combined)} months")
print(f"Period: {combined.index[0].date()}  →  {combined.index[-1].date()}")
print(f"Recession months: {int(combined['recession'].sum())}  "
      f"({combined['recession'].mean()*100:.1f}% of sample)")


"""# ── 5. CROSS-CORRELATION ANALYSIS ────────────────────────────
# Core question: does spread today predict recession in X months?
#
# Method: for each lag L (0 to 24 months), we compute the
# correlation between spread[t] and recession[t + L].
#
# A negative correlation at lag L means: lower spread today -> higher recession probability in L months
#
# We expect the strongest (most negative) correlation around
# lag 12-18 months based on historical literature.
#
# We use Pearson correlation here. """

print("\nComputing cross-correlations (spread[t] vs recession[t+lag])...")

max_lag = 24   # months
lag_range  = range(0, max_lag + 1)

cross_corr = {col: [] for col in spread_cols}

for lag in lag_range:
    for col in spread_cols:
        # shift recession backward by lag so we compare
        # spread[t] with recession[t+lag]
        rec_shifted = combined["recession"].shift(-lag)

        valid = combined[col].dropna().index.intersection(
            rec_shifted.dropna().index
        )

        corr = combined.loc[valid, col].corr(rec_shifted.loc[valid])
        cross_corr[col].append(corr)

# find the lag with the most negative correlation per spread
print("\nBest predictive lag per spread")
best_lags = {}
for col in spread_cols:
    corrs = cross_corr[col]
    best_lag = int(np.argmin(corrs)) 
    best_corr = corrs[best_lag]
    best_lags[col] = best_lag
    print(f"{col:<16} best lag = {best_lag:>2} months"
          f"(correlation = {best_corr:.3f})")


"""# ── 6. Logistic regretion - recession ptobability
# Cross correlation tells us the best lag per spread.
#
# Logistic regression outputs a probability between 0 and 1.
# We use it to answer: given today's spread value, what is the probability of recession in L months?

# We test three forecast horizons: 6, 12, 18 months.
# For each horizon we fit one model per spread and measure ROC-AUC.

# ROC-AUC interpretation:
# 0.5 = no better than random (useless)
# 0.7 = decent predictive power
# 0.8+ = strong predictive power
# 1.0 = perfect (never happens)"""

horizons = [6, 12, 18]

# store AUC results
auc_results = {col: {} for col in spread_cols}

# store predicted probabilities for the most recent observation
current_probs = {col: {} for col in spread_cols}

print("\nROC-AUC by spread and forecast horizon")
print(f"{'Spread':<16} {'6m AUC':>8}{'12m AUC':>8} {'18m AUC':>8}")
print("-" * 48)

scaler_lr = StandardScaler()

for col in spread_cols:
    for horizon in horizons:
        # build target
        target = combined["recession"].shift(-horizon)

        # align and drop nan
        df_lr = pd.DataFrame({
            "x": combined[col],
            "y": target,
        }).dropna()

        x = df_lr[["x"]].values
        y = df_lr["y"].values.astype(int)

        # skip if only one class present
        if len(np.unique(y)) < 2:
            auc_results[col][horizon] = np.nan
            continue

        x_scaled = scaler_lr.fit_transform(x)

        model = LogisticRegression(random_state=42)
        model.fit(x_scaled, y)

        y_prob = model.predict_proba(x_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
        auc_results[col][horizon] = auc

        # current recession probability using latest spread value
        latest_x = scaler_lr.transform([[combined[col].iloc[-1]]])
        current_probs[col][horizon] = model.predict_proba(latest_x)[0][1]

    row = f"{col:<16}"
    for h in horizons:
        v = auc_results[col].get(h, np.nan)
        row += f"{v:>8.3f}"
    print(row)

# current recession probabilities
print("\nCurrent recession probability (based on latest spread values)")
print(f"Latest month in data: {combined.index[-1].date()}")
print(f"\n{'Spread':<16}  {'6m prob':>8}  {'12m prob':>8}  {'18m prob':>8}")
print("-" * 48)
for col in spread_cols:
    row = f"{col:<16}"
    for h in horizons:
        v = current_probs[col].get(h, np.nan)
        row += f"{v:>8.1%}"
    print(row)


"""# 7. Plot - cross-correlation curves
# Left panel: cross-correlation vs lag for all three spreads
# Right panel: ROC curves at 12-month horizon for all three spreads"""

print("\nBuilding charts...")

spread_colors = {
    "spread_2y10y": "#2563EB",
    "spread_3m10y": "#16A34A",
    "spread_2y30y": "#9333EA",
}

spread_labels = {
    "spread_2y10y": "2Y–10Y",
    "spread_3m10y": "3M–10Y",
    "spread_2y30y": "2Y–30Y",
}

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor("#FAFAFA")
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

ax_corr = fig.add_subplot(gs[0])
ax_roc = fig.add_subplot(gs[1])

for ax in (ax_corr, ax_roc):
    ax.set_facecolor("#FAFAFA")
    ax.spines[["top", "right"]].set_visible(False)

# left panel
for col in spread_cols:
    corrs = cross_corr[col]
    ax_corr.plot(
        list(lag_range),
        corrs,
        color=spread_colors[col],
        linewidth=2.0,
        label=spread_labels[col],
    )
    # best lag
    bl = best_lags[col]
    ax_corr.scatter(
        bl,corrs[bl],
        color=spread_colors[col],
        s=80, zorder=5,
    )
    ax_corr.annotate(
        f"{bl}m",
        xy=(bl,corrs[bl]),
        xytext=(bl + 0.5, corrs[bl] - 0.02),
        fontsize=8,
        color=spread_colors[col],
    )

ax_corr.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")
ax_corr.set_xlabel("Forecast horizon (months)", fontsize=10)
ax_corr.set_ylabel("Correlation with recession", fontsize=10)
ax_corr.set_title(
    "Cross-correlation: spread[t] vs recession[t + lag]",
    fontsize=11, fontweight="bold", loc="left",
)
ax_corr.legend(fontsize=9, framealpha=0.7)
ax_corr.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax_corr.set_xlim(0, max_lag)

# right panel
horizon_plot = 12

for col in spread_cols:
    target = combined["recession"].shift(-horizon_plot)
    df_roc = pd.DataFrame({"x": combined[col], "y": target}).dropna()
    x_s = scaler_lr.fit_transform(df_roc[["x"]].values)
    y_b = df_roc["y"].values.astype(int)

    if len(np.unique(y_b)) < 2:
        continue

    model_roc = LogisticRegression(random_state=42)
    model_roc.fit(x_s, y_b)
    y_prob_roc = model_roc.predict_proba(x_s)[:, 1]

    fpr, tpr, _ = roc_curve(y_b, y_prob_roc)
    auc_val = auc_results[col].get(horizon_plot, np.nan)

    ax_roc.plot(
        fpr, tpr,
        color=spread_colors[col],
        linewidth=2.0,
        label=f"{spread_labels[col]}  (AUC={auc_val:.3f})",
    )

# diagonal = random classifier
ax_roc.plot([0, 1], [0, 1], color="#D1D5DB", linewidth=1.0,
            linestyle="--", label="Random (AUC=0.500)")

ax_roc.set_xlabel("False positive rate", fontsize=10)
ax_roc.set_ylabel("True positive rate", fontsize=10)
ax_roc.set_title(
    f"ROC curves - recession prediction at {horizon_plot}-month horizon",
    fontsize=11, fontweight="bold", loc="left",
)
ax_roc.legend(fontsize=9, framealpha=0.7)
ax_roc.grid(linestyle=":", linewidth=0.5, alpha=0.5)
ax_roc.set_xlim(0, 1)
ax_roc.set_ylim(0, 1)

fig.suptitle(
    "Predictive power of yield curve spreads for US recessions",
    fontsize=13, fontweight="bold", y=1.02,
)

plt.tight_layout()
plot_path = os.path.join(plots_dir, "03_recession_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
# plt.show()
print(f"  Chart saved → {plot_path}")


"""# 8. Summary table
# Compact table - one row per spread, showing best lag and AUC
# at each horizon. This is the key output of this file goes directly into the project write up"""

print("\nSummary table")
print(f"\n{'Spread':<16} {'Best lag':>9} {'AUC 6m':>8} {'AUC 12m':>9} {'AUC 18m':>9}")

summary_rows = []
for col in spread_cols:
    bl = best_lags[col]
    a6 = auc_results[col].get(6,  np.nan)
    a12 = auc_results[col].get(12, np.nan)
    a18 = auc_results[col].get(18, np.nan)
    print(f"{spread_labels[col]:<14} {bl:>7} mo  "
          f"{a6:>8.3f}  {a12:>9.3f}  {a18:>9.3f}")
    summary_rows.append({
        "spread": col,
        "best_lag": bl,
        "auc_6m": round(a6,  3),
        "auc_12m": round(a12, 3),
        "auc_18m": round(a18, 3),
    })

# save summary as csv 
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(data_dir, "spread_predictive_power.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved → {summary_path}")

# save current probabilities
probs_rows = []
for col in spread_cols:
    for h in horizons:
        probs_rows.append({
            "spread": col,
            "horizon": h,
            "prob": current_probs[col].get(h, np.nan),
        })
probs_df = pd.DataFrame(probs_rows)
probs_path = os.path.join(data_dir, "current_recession_probs.csv")
probs_df.to_csv(probs_path, index=False)
print(f"Current probs saved -> {probs_path}")

print("\n" + "=" * 60)
print("Recession analysis complete.")
print("=" * 60)

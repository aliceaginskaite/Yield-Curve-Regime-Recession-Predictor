
"""
 7_dashboard_plotly.py

 What this file does:
   - Loads all outputs from files 1-6
   - Builds a single interactive HTML dashboard with Plotly
   - Six panels: spreads, regimes, composite signal,
     recession probabilities, commodity performance,
     and a summary stats table
   - Saves dashboard.html can be opened in any browser, no server needed

"""

# 1. Imports
import os
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

print("=" * 60)
print("7_dashboard_plotly.py - dashboard")
print("=" * 60)


# 2. Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)


# 3. Load all data
print("\nLoading all outputs...")

spreads = pd.read_csv(
    os.path.join(data_dir, "spreads.csv"),
    index_col="Date", parse_dates=True,
)

spreads_regime = pd.read_csv(
    os.path.join(data_dir, "spreads_with_regime.csv"),
    index_col="Date", parse_dates=True,
)

recession = pd.read_csv(
    os.path.join(data_dir, "recession.csv"),
    index_col="Date", parse_dates=True,
)

composite = pd.read_csv(
    os.path.join(data_dir, "composite_signal.csv"),
    index_col="Date", parse_dates=True,
)

commodity_perf = pd.read_csv(
    os.path.join(data_dir, "commodity_regime_performance.csv"),
)

predictive_power = pd.read_csv(
    os.path.join(data_dir, "spread_predictive_power.csv"),
)

current_probs = pd.read_csv(
    os.path.join(data_dir, "current_recession_probs.csv"),
)

print("All files loaded successfully")


# 4. Recession bands helper 
# Plotly uses "shapes" to draw shaded rectangles. 
# We build a list of shape dicts (one per recession period).
# These get passed into fig.update_layout(shapes=...).

def build_recession_shapes(recession_df, row_refs, y0=0, y1=1):
    """
    Build Plotly shape dicts for NBER recession shading.

    recession_df : monthly recession dataframe (column 'recession')
    row_refs : list of 'y' axis references e.g. ['y1', 'y3']
                   one shape per row so shading appears on multiple panels
    """
    # resample recession to daily by forward fill
    daily_idx = pd.date_range(
        recession_df.index.min(), recession_df.index.max(), freq="D"
    )
    rec_daily = recession_df.reindex(daily_idx, method="ffill")
    rec_vals = rec_daily["recession"]
    rec_starts = rec_vals[(rec_vals == 1) & (rec_vals.shift(1) == 0)].index.tolist()
    rec_ends = rec_vals[(rec_vals == 0) & (rec_vals.shift(1) == 1)].index.tolist()

    if len(rec_starts) > len(rec_ends):
        rec_ends.append(rec_daily.index[-1])

    shapes = []
    for rs, re in zip(rec_starts, rec_ends):
        for yref in row_refs:
            shapes.append(dict(
                type = "rect",
                xref = "x",
                yref = yref,
                x0 = rs.strftime("%Y-%m-%d"),
                x1 = re.strftime("%Y-%m-%d"),
                y0 = y0,
                y1 = y1,
                fillcolor = "rgba(200,200,200,0.35)",
                line = dict(width=0),
                layer = "below",
            ))
    return shapes


# 5. Color maps
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

regime_colors = {
    "Steep" : "#16A34A",
    "Flat" : "#F59E0B",
    "Humped" : "#8B5CF6",
    "Inverted": "#EF4444",
}

signal_colors = {
    "GREEN": "#16A34A",
    "YELLOW": "#F59E0B",
    "RED" : "#EF4444",
}


"""# 6. Build dashboard layout
# Six subplots arranged vertically:
# Row 1 - Three yield curve spreads (shared x-axis)
# Row 2 - Regime timeline (colored scatter)
# Row 3 - Composite signal with traffic light zones
# Row 4 - Recession probability at 12m horizon
# Row 5 - Gold annualized return by regime (bar)
# Row 6 - WTI annualized return by regime (bar)"""

print("\nBuilding interactive dashboard...")

fig = make_subplots(
    rows=6, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(
        "Yield Curve Spreads  (2Y-10Y · 3M-10Y · 2Y-30Y)",
        "Yield Curve Regime",
        "Composite Signal  (weighted z-score)",
        "Recession Probability at 12-Month Horizon  (%)",
        "Gold - Annualized Return by Regime  (%)",
        "WTI - Annualized Return by Regime  (%)",
    ),
    row_heights=[0.22, 0.08, 0.18, 0.15, 0.17, 0.17],
    specs=[[{"type": "scatter"}]] * 4 + [[{"type": "bar"}]] * 2,
)


# 7. Row 1: three spreads

for col in ["spread_2y10y", "spread_3m10y", "spread_2y30y"]:
    # raw spread
    fig.add_trace(go.Scatter(
        x=spreads.index,
        y=spreads[col],
        mode="lines",
        name=spread_labels[col] + " (raw)",
        line=dict(color=spread_colors[col], width=0.6),
        opacity=0.3,
        legendgroup=col,
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>" + spread_labels[col] + ": %{y:.2f} pp<extra></extra>",
    ), row=1, col=1)

    # 90-day MA 
    fig.add_trace(go.Scatter(
        x=spreads.index,
        y=spreads[f"{col}_ma90"],
        mode="lines",
        name=spread_labels[col],
        line=dict(color=spread_colors[col], width=2.0),
        legendgroup=col,
        hovertemplate="%{x|%Y-%m-%d}<br>" + spread_labels[col] + " MA90: %{y:.2f} pp<extra></extra>",
    ), row=1, col=1)

# zero line on spreads panel
fig.add_hline(y=0, line=dict(color="#EF4444", width=1, dash="dash"),
              opacity=0.7, row=1, col=1)


# 8. Row 2: regime timeline
# We plot each regime as a separate scatter with colored markers.
# Using markers means color changes instantly at transitions.

regime_data = spreads_regime.dropna(subset=["regime"])

for regime_name, color in regime_colors.items():
    mask = regime_data["regime"] == regime_name
    fig.add_trace(go.Scatter(
        x=regime_data.index[mask],
        y=[1] * mask.sum(),
        mode="markers",
        name=regime_name,
        marker=dict(color=color, size=3, symbol="square"),
        legendgroup="regime_" + regime_name,
        hovertemplate="%{x|%Y-%m-%d}<br>Regime: " + regime_name + "<extra></extra>",
    ), row=2, col=1)

fig.update_yaxes(showticklabels=False, row=2, col=1)


# 9. Row 3: composite signal

# background zone colors using shapes
fig.add_trace(go.Scatter(
    x=composite.index,
    y=composite["composite_30d"],
    mode="lines",
    name="Composite 30d MA",
    line=dict(color="#94A3B8", width=0.8),
    opacity=0.5,
    hovertemplate="%{x|%Y-%m-%d}<br>Composite 30d: %{y:.3f}<extra></extra>",
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=composite.index,
    y=composite["composite"],
    mode="lines",
    name="Composite 90d MA",
    line=dict(color="#1E293B", width=2.0),
    hovertemplate="%{x|%Y-%m-%d}<br>Composite 90d: %{y:.3f}<br>Signal: %{customdata}<extra></extra>",
    customdata=composite["signal"],
), row=3, col=1)

fig.add_hline(y=0, line=dict(color="#9CA3AF", width=0.8, dash="dot"),
              opacity=0.5, row=3, col=1)


"""# 10. Row 4: recession probability
# Show the 12-month recession probability implied by each spread using the logistic regression from file 4.
# We reconstruct the probability time series from the spreads using the same z-score.
# For the dashboard we show current_probs as a bar + a note.

# build a simple time series: for each spread, show 12m prob
# We use the composite signal value as a proxy — scaled to 0-100%
# using the historical relationship between composite and recession probability.

# load the monthly combined data to reconstruct probabilities"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

spreads_monthly = spreads[["spread_2y10y", "spread_3m10y", "spread_2y30y"]].resample("ME").mean()
spreads_monthly.index = spreads_monthly.index.to_period("M").to_timestamp()
recession_m = recession.copy()
recession_m.index = recession_m.index.to_period("M").to_timestamp()

combined_m = spreads_monthly.join(recession_m, how="inner").dropna()

horizon = 12
prob_series = {}

for col in ["spread_2y10y", "spread_3m10y", "spread_2y30y"]:
    target = combined_m["recession"].shift(-horizon)
    df_lr = pd.DataFrame({"x": combined_m[col], "y": target}).dropna()

    sc = StandardScaler()
    x_s = sc.fit_transform(df_lr[["x"]].values)
    y_b = df_lr["y"].values.astype(int)

    if len(np.unique(y_b)) < 2:
        continue

    mdl = LogisticRegression(random_state=42)
    mdl.fit(x_s, y_b)

    # predict probability for each month
    x_all = sc.transform(combined_m[[col]].values)
    probs  = mdl.predict_proba(x_all)[:, 1] * 100
    prob_series[col] = pd.Series(probs, index=combined_m.index)

for col in ["spread_2y10y", "spread_3m10y", "spread_2y30y"]:
    if col not in prob_series:
        continue
    fig.add_trace(go.Scatter(
        x=prob_series[col].index,
        y=prob_series[col].values,
        mode="lines",
        name=spread_labels[col] + " recession prob",
        line=dict(color=spread_colors[col], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>" + spread_labels[col] +
                      " recession prob (12m): %{y:.1f}%<extra></extra>",
    ), row=4, col=1)

# 50% reference line
fig.add_hline(y=50, line=dict(color="#EF4444", width=1, dash="dash"),
              opacity=0.6, row=4, col=1)


# 11. Rows 5 and 6: commodity bars
regimes_ordered = ["Steep", "Flat", "Humped", "Inverted"]
colors_ordered  = [regime_colors[r] for r in regimes_ordered]

for row_n, commodity in [(5, "Gold"), (6, "WTI")]:
    df_c = commodity_perf[commodity_perf["commodity"] == commodity]
    df_c = df_c.set_index("regime").reindex(regimes_ordered).reset_index()

    returns_pct = (df_c["ann_return"] * 100).round(1)

    fig.add_trace(go.Bar(
        x=df_c["regime"],
        y=returns_pct,
        marker_color=colors_ordered,
        name=commodity + " by regime",
        text=[f"{v:.1f}%" for v in returns_pct],
        textposition="outside",
        hovertemplate="Regime: %{x}<br>Ann. Return: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=row_n, col=1)

    fig.add_hline(y=0, line=dict(color="#6B7280", width=0.8),
                  row=row_n, col=1)


# ── 12. Recession shapes across all rows
# Build recession shading shapes for rows 1, 3, 4
# (rows 2 already has regime colors, bars don't need shading)

all_shapes = []

# for rows 1, 3, 4 - per-row y-axis references
for yref in ["y", "y3", "y4"]:
    rec_daily_idx = pd.date_range(
        recession.index.min(), recession.index.max(), freq="D"
    )
    rec_daily = recession.reindex(rec_daily_idx, method="ffill")
    rec_vals = rec_daily["recession"]
    starts = rec_vals[(rec_vals == 1) & (rec_vals.shift(1) == 0)].index
    ends = rec_vals[(rec_vals == 0) & (rec_vals.shift(1) == 1)].index.tolist()

    if len(starts) > len(ends):
        ends.append(rec_daily.index[-1])

    for rs, re in zip(starts, ends):
        all_shapes.append(dict(
            type="rect", xref="x", yref=yref + " domain",
            x0=rs.strftime("%Y-%m-%d"),
            x1=re.strftime("%Y-%m-%d"),
            y0=0, y1=1,
            fillcolor="rgba(200,200,200,0.30)",
            line=dict(width=0),
            layer="below",
        ))


# ── 13. Layout and styling
last_date = composite.index[-1].strftime("%b %d, %Y")
current_sig = composite["signal"].iloc[-1]
current_val = composite["composite"].iloc[-1]
sig_color = signal_colors.get(current_sig, "#6B7280")

fig.update_layout(
    title=dict(
        text=(
            f"<b>Yield Curve Regime & Recession Predictor</b>"
            f"<br><sup>Last updated: {last_date}  |  "
            f"Current signal: "
            f"<span style='color:{sig_color}'><b>{current_sig}</b></span>"
            f" (composite = {current_val:.3f})</sup>"
        ),
        x=0.01,
        font=dict(size=18),
    ),
    height=1400,
    paper_bgcolor="#FAFAFA",
    plot_bgcolor="#FAFAFA",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="right",
        x=1,
        font=dict(size=10),
    ),
    shapes=all_shapes,
    font=dict(family="Inter, Arial, sans-serif", size=11),
    margin=dict(l=60, r=40, t=120, b=60),
)

# clean up axes
for row_n in range(1, 7):
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        showline=False,
        row=row_n, col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        row=row_n, col=1,
    )

# x-axis label only on bottom panel
fig.update_xaxes(title_text="Date", row=6, col=1)

# y-axis labels
fig.update_yaxes(title_text="Spread (pp)", row=1, col=1)
fig.update_yaxes(title_text="Z-score", row=3, col=1)
fig.update_yaxes(title_text="Probability (%)", row=4, col=1)
fig.update_yaxes(title_text="Ann. Return (%)", row=5, col=1)
fig.update_yaxes(title_text="Ann. Return (%)", row=6, col=1)


# 14. Save html
html_path = os.path.join(base_dir, "dashboard.html")
fig.write_html(
    html_path,
    include_plotlyjs="cdn",   
    full_html=True,
    config={
        "scrollZoom" : True,
        "displaylogo" : False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    },
)

print(f"\n dashboard.html saved → {html_path}")
print(f"\n Open this file in any browser:")
print(f" {html_path}")

print("\n" + "=" * 60)
print(f" Current signal : {current_sig}")
print(f" Composite value : {current_val:.3f}")
print(f" Last data point : {last_date}")
print("=" * 60)
print("\n Dashboard complete! All six panels:")
print(" 1. Three yield curve spreads")
print(" 2. Regime timeline")
print(" 3. Composite signal with traffic light")
print(" 4. Recession probability at 12m horizon")
print(" 5. Gold performance by regime")
print(" 6. WTI performance by regime")
print("=" * 60)

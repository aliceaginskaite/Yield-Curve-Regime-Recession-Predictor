# Yield-Curve-Regime-Recession-Predictor

## 📈 Fixed Income — Yield Curve Regime & Recession Predictor 

## Why this project exists

This is a research pipeline that turns raw Treasury yields into an early-warning system. It has regime classification, predictive analytics, commodity overlays, and an interactive dashboard.
 
We know that yield reversion is a red flag. This project quantifies how bad, which spread matters most, how far in advance, and what it means for your commodity positions.

What is in there:
- Three spreads tracked simultaneously
- K-means clustering for data-driven regime detection instead of manual thresholds
- Cross-correlation + ROC-AUC to measure each spread's predictive power
- Expanding-window normalization to eliminate look-ahead bias in backtests
- Commodity performance broken down by yield curve regime (Gold vs WTI)
- A composite signal with a traffic-light output and false alarm rate analysis
- An interactive Plotly dashboard — one HTML file, no server needed

**Current signal (as of June 2026): 🟢 GREEN** (composite z-score = −0.478)

---

## Pipeline overview

```
1_data_collection.py       ← FRED API + yfinance
        ↓
2_spreads_construction.py  ← 3 spreads + rolling MAs
        ↓
3_regime_detection.py      ← K-means clustering (k=4)
        ↓
4_recession_analysis.py    ← Cross-correlation + Logistic regression + ROC-AUC
        ↓
5_commodity_analysis.py    ← Gold & WTI performance by regime
        ↓
6_composite_signal.py      ← Weighted composite + traffic light
        ↓
7_dashboard_plotly.py      ← Interactive HTML dashboard
```

Each script reads from `data/` and writes back to `data/`. They should be ran in order. All outputs persist as CSVs so any step can be re-run independently.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/aliceaginskaite/Yield-Curve-Regime-Recession-Predictor.git
cd yield-curve-regime-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your FRED API key in 1_data_collection.py
API_KEY = "your_key_here"   # free at https://fred.stlouisfed.org/docs/api/api_key.html

# 4. Run the full pipeline
python 1_data_collection.py
python 2_spreads_construction.py
python 3_regime_detection.py
python 4_recession_analysis.py
python 5_commodity_analysis.py
python 6_composite_signal.py
python 7_dashboard_plotly.py

# 5. Open the dashboard
open dashboard.html
```

---

## Data sources

| Dataset | Source | Frequency | Coverage |
|---|---|---|---|
| 3M Treasury Yield (DGS3MO) | FRED | Daily | 1990–present |
| 2Y Treasury Yield (DGS2) | FRED | Daily | 1990–present |
| 10Y Treasury Yield (DGS10) | FRED | Daily | 1990–present |
| 30Y Treasury Yield (DGS30) | FRED | Daily | 1990–present |
| NBER Recession Indicator (USREC) | FRED | Monthly | 1990–present |
| Gold Front-Month Futures (GC=F) | Yahoo Finance | Daily | 2000–present |
| WTI Crude Front-Month Futures (CL=F) | Yahoo Finance | Daily | 2000–present |

**Coverage:** 9,513 daily observations across ~35 years, spanning 4 complete NBER recession cycles.

---

## Why three spreads

### 2Y–10Y (Classic benchmark)
The Fed watches this one. When the Fed hikes aggressively, the 2Y rises faster than the 10Y and the spread inverts. Every US recession since 1980 was preceded by an inversion here.

### 3M–10Y (NY Fed preferred)
The 3M yield is nearly a direct proxy for the Fed Funds Rate. The NY Fed's own research identifies this spread as the *strongest statistical predictor* of recessions at the 12-month horizon. AUC at 12m: **0.777**.

### 2Y–30Y (Long-end slope)
Captures the full slope of the curve. The 30Y is driven by long-term inflation expectations and structural demand from pension funds and insurance companies (ALM). Less sensitive to Fed actions, more to structural factors.

**Historical inversion rates (1990–2026):**

| Spread | Days Inverted | % of Time |
|---|---|---|
| 2Y–10Y | 1,093 | 11.5% |
| 3M–10Y | 1,165 | 12.3% |
| 2Y–30Y | 848 | 8.9% |

---

## Regime detection (Script 3)

K-means clustering on 8 features derived from the three spreads (raw values + 30-day and 90-day moving averages + daily first differences).

**Why k=4 when silhouette score peaks at k=2?**

k=2 gives "steep" vs "everything else" which is statistically clean but economically useless. k=4 maps cleanly onto the four regimes practitioners actually care about: Steep, Flat, Inverted, and Humped. This is a deliberate override for interpretability.

**Regime distribution (1990–2026):**

| Regime | Days | % of Sample |
|---|---|---|
| Flat | 5,204 | 55.2% |
| Inverted | 2,482 | 26.3% |
| Steep | 1,749 | 18.5% |

**Transition matrix** once in Inverted, it stays there:

| From \ To | Flat | Inverted | Steep |
|---|---|---|---|
| Flat | 80.9% | 2.2% | 16.8% |
| Inverted | 4.7% | **95.3%** | 0.0% |
| Steep | 50.1% | 0.0% | 49.9% |

The 95.3% self-persistence of the Inverted regime confirms what practitioners observe: once the curve inverts, it stays inverted for months.

---

## Recession prediction (Script 4)

### Cross-correlation analysis
For each spread, we compute `corr(spread[t], recession[t + lag])` across lags 0 to 24 months.

All three spreads peak in predictive correlation at **~23–24 months** meaning the best leading indicator is the spread from almost 2 years ago.

### ROC-AUC by horizon

| Spread | Best Lag | AUC @ 6m | AUC @ 12m | AUC @ 18m |
|---|---|---|---|---|
| 2Y–10Y | 24 months | 0.577 | 0.745 | **0.834** |
| 3M–10Y | 24 months | 0.622 | 0.777 | **0.828** |
| 2Y–30Y | 23 months | 0.598 | 0.752 | **0.859** |

AUC above 0.8 at the 18-month horizon is a strong result for a single-variable predictor with no additional macro inputs.

### Current recession probabilities (as of April 2026)

| Spread | 6m | 12m | 18m |
|---|---|---|---|
| 2Y–10Y | 9.3% | 8.3% | 7.2% |
| 3M–10Y | 9.9% | 9.1% | 8.6% |
| 2Y–30Y | 8.9% | 6.9% | 4.8% |

All three spreads agree: recession probability is low across all horizons.

---

## Commodity performance by regime (Script 5)

This is where the project becomes actionable.

### Gold

| Regime | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Steep | 0.3% | 19.0% | 0.02 | −56.5% |
| Flat | 9.8% | 18.2% | 0.54 | −38.9% |
| **Inverted** | **20.0%** | **16.4%** | **1.22** | **−21.9%** |

Gold performs best when the curve is inverted: lower volatility, higher return, best Sharpe. This aligns with Gold's role as a safe-haven asset during late-cycle and recessionary environments.

### WTI Crude

| Regime | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| **Steep** | **84.3%** | **39.0%** | **2.16** | **−56.8%** |
| Flat | −11.0% | 44.8% | −0.25 | −99.6% |
| Inverted | −31.2% | 43.2% | −0.72 | −90.0% |

WTI is the mirror image of Gold. It thrives in steep curve environments (early cycle, economic expansion) and collapses in inversions. The regime signal functions as a regime-aware commodity rotation signal.

**The trade:** Long Gold / Short WTI when the curve inverts. Reverse when the curve steepens.

---

## Composite signal (Script 6)

### Construction
1. Each spread is z-scored using an **expanding window** (no look-ahead bias: at each date, only data available up to that date is used for normalization)
2. Spreads are weighted by their expanding-window AUC at the 12-month horizon
3. The weighted sum is smoothed with a 90-day rolling mean

**Most recent weights (June 2026):**
- 2Y–10Y: 32.8%
- 3M–10Y: 34.1%
- 2Y–30Y: 33.1%

Weights are nearly equal because all three spreads have similar predictive power. The composite is effectively a precision-weighted average.

### Traffic light thresholds

| Signal | Condition | Historical Frequency |
|---|---|---|
| 🟢 GREEN | Composite > 33rd percentile | 67.0% of days |
| 🟡 YELLOW | 10th–33rd percentile | 23.0% of days |
| 🔴 RED | Composite ≤ 10th percentile | 10.0% of days |

### False alarm analysis (RED signal, no recession within 18 months)

| Signal | False Alarm Rate |
|---|---|
| Composite | 62.9% |
| 2Y–10Y alone | 58.3% |
| 3M–10Y alone | 56.7% |
| 2Y–30Y alone | 64.1% |

The composite signal does not dramatically outperform individual spreads on false alarm rate. This is a known limitation of yield curve signals in general. Inversion frequently precedes slowdowns that do not qualify as NBER recessions. The composite is best used as a *risk regime indicator* (are conditions consistent with late-cycle?) rather than a binary recession predictor.

---

## Dashboard (Script 7)

`dashboard.html` — a single self-contained file, no server required.

Six interactive panels:
1. **Yield curve spreads** - all three spreads with 90-day MA and NBER recession shading
2. **Regime timeline** - color-coded regime classification across the full history
3. **Composite signal** - z-score with GREEN/YELLOW/RED zone shading
4. **Recession probability** - logistic regression output at 12-month horizon for each spread
5. **Gold by regime** - annualized returns bar chart
6. **WTI by regime** - annualized returns bar chart
   
All panels share a synchronized x-axis, hover anywhere to see all six panels at the same date.

---

## Project structure

```
yield-curve-regime-predictor/
│
├── data/                       
│   ├── treasuries.csv
│   ├── commodities.csv
│   ├── recession.csv
│   ├── spreads.csv
│   ├── spreads_with_regime.csv
│   ├── spread_predictive_power.csv
│   ├── current_recession_probs.csv
│   ├── commodity_regime_performance.csv
│   └── composite_signal.csv
│
├── plots/                        
│   ├── 1_spreads.png
│   ├── 02_regimes.png
│   ├── 03_recession_analysis.png
│   ├── 4_commodity_analysis.png
│   └── 5_composite_signal.png
│
├── 1_data_collection.py
├── 2_spreads_construction.py
├── 3_regime_detection.py
├── 4_recession_analysis.py
├── 5_commodity_analysis.py
├── 6_composite_signal.py
├── 7_dashboard_plotly.py
│
├── dashboard.html              
├── requirements.txt
└── README.md
```

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
scipy
plotly
requests
yfinance
```

Install with:
```bash
pip install -r requirements.txt
```

## Known limitations & honest caveats

**False alarm rate is high (~63%).** Yield curve inversion frequently precedes economic slowdowns that the NBER does not formally classify as recessions. The signal reflects *conditions consistent with late-cycle stress*, not a guaranteed recession call.

**T-tests for commodity returns are non-significant.** The t-test for Gold and WTI return differences across regimes came back with `nan` p-values due to regime-length imbalances. The performance patterns are directionally robust across the full sample but should not be treated as statistically proven.

**Logistic regression is intentionally simple.** Single-variable models are used deliberately to isolate the spread's predictive content. Adding macro controls (unemployment, PMI, credit spreads) would improve AUC but reduce interpretability.

**NBER recession dates are announced with a lag of 6–18 months.** The training labels are accurate historically, but in real-time operation the most recent recession boundary is not yet known. This is a structural limitation of all NBER-based recession models.

---

## Potential extensions

- Add credit spreads (HY–IG, CDX) as additional features
- Incorporate Fed Funds Rate and unemployment rate as macro controls
- Test Hidden Markov Model (HMM) as an alternative to K-means for regime detection
- Add equity sector rotation analysis by regime (similar to commodity analysis)
- Walk-forward backtest of a long/short commodity portfolio using the composite signal
- Deploy as a live dashboard with daily data refresh via GitHub Actions + FRED API

---

## License

Use freely, attribution appreciated.

---

*Built with FRED data, yfinance, scikit-learn, and Plotly. No proprietary data sources required.*

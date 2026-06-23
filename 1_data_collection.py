"""
 1_data_collection.py
 Yield Curve Regime & Recession Predictor

 What this file does:
- Downloads Treasury yields from the FRED API
- Downloads the NBER recession indicator
- Downloads Gold and WTI via yfinance
- Saves everything to the data/ folder for the following files
"""

# 1. Imports
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


print("1_data_collection.py - data collection")
print("=" * 60)


#  2. Settings

API_KEY  = "b3bed8ac61aa8f620ac66318cabce0a7"
BASE_URL = "https://api.stlouisfed.org/fred/"

START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

#Folder for saving data (created automatically if not)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

print(f"\nPeriod: {START_DATE}  →  {END_DATE}")
print(f"Data folder: {DATA_DIR}\n")

"""""
#  3. Helper function Fred API request
# All work with FRED is done through one endpoint: series/observations
# It returns a time series for any ticker.
#
# Request parameters:
#   series_id — series ticker (e.g., "DGS10")
#   api_key — API key
#   file_type — we request JSON (easier to parse than XML)
#   observation_start — beginning date
#   observation_end — end date
#
# FRED returns a list of observations — each is a dictionary:
#   {"date": "2024-01-02", "value": "4.05"}
# The value "." means a missing value (no data for that day)
"""

def fetch_fred_series(series_id: str, label: str) -> pd.DataFrame:
    """
  Downloads a single time series from the FRED API.
  Returns a DataFrame with one column (label) and a Date index.
    """
    url = BASE_URL + "series/observations"

    params = {
        "series_id" : series_id,
        "api_key" : API_KEY,
        "file_type" : "json",
        "observation_start" : START_DATE,
        "observation_end" : END_DATE,
    }

    response = requests.get(url, params=params, timeout=30)

    # If request failed, raise an error
    if response.status_code != 200:
        raise ConnectionError(
            f"FRED API returned {response.status_code} for {series_id}.\n"
            f"Response: {response.text[:200]}"
        )

    data = response.json()

    # Transforming the list observations to a DataFrame
    observations = data["observations"]
    df = pd.DataFrame(observations)[["date", "value"]]

    # Converting data types
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.rename(columns={"date": "Date", "value": label})
    df = df.set_index("Date")

    return df

"""""
# 4. Loading treasury yields
# Load 4 series - they will give us 3 spreads in the following file:
#
# DGS3MO - 3-month yield (Treasury Bill)
# Very sensitive to the Fed's rate decisions.
# When the Fed raises rates sharply, the 3M grows the fastest.
#
# DGS2 - 2-year yield
# Reflects market expectations for the rate two years in advance.
# The most sensitive to monetary policy of the long-term securities.
#
# DGS10 - 10-year yield (global benchmark)
# Reflects long-term growth and inflation expectations.
# The 2Y-10Y spread is the most well-known spread.
#
# DGS30 - 30-year yield
# Purchased by pension funds and insurers for ALM.
# Less sensitive to the Fed, more sensitive to inflation expectations.
"""

TREASURY_TICKERS = {
    "DGS3MO": "3M",
    "DGS2" : "2Y",
    "DGS10" : "10Y",
    "DGS30" : "30Y",
}

print("Loading treasury yields from FRED API...")

treasury_frames = []

for fred_id, label in TREASURY_TICKERS.items():
    try:
        df = fetch_fred_series(fred_id, label)
        treasury_frames.append(df)
        print(f" {label:5s}  ({fred_id}) - {len(df)} lines")
    except Exception as e:
        print(f"! {label:5s}  ({fred_id}) - error: {e}")

# Concatenate all 4 series into one DataFrame.
treasuries = pd.concat(treasury_frames, axis=1, join="outer")
treasuries.index.name = "Date"
treasuries.index = pd.to_datetime(treasuries.index)

treasuries.ffill(inplace=True)

treasuries.dropna(how="all", inplace=True)

print(f"\nTreasury Yields: {treasuries.shape[0]} rows × {treasuries.shape[1]} columns")
print(f"Period: {treasuries.index[0].date()} → {treasuries.index[-1].date()}")
print(treasuries.tail(3).to_string())


"""# ── 5. Loading Recession Indicator NBER ────────────────────
# USREC — binary indicator of recessions from NBER
#   1 = this month was a recession (according to NBER's official date)
#   0 = no recession
# NBER announces recessions with a lag, with a delay of 6-18 months. 
# This means historical data is accurate (no look-ahead bias). 
# This is suitable for backtesting.

# The data is monthly, which is normal. When analyzing predictive power,
# we will resample the spreads to a monthly frequency as well."""

print("\nLoading recession indicator from NBER (USREC)...")

try:
    recession = fetch_fred_series("USREC", "recession")
    recession.index = pd.to_datetime(recession.index)

    n_rec = int(recession["recession"].sum())
    print(f"USREC - {len(recession)} lines (monthly)")
    print(f"Recessionary months in the sample: {n_rec}")
    print(f"This corresponds to ~{n_rec//12} years of recession over the period")

except Exception as e:
    print(f"! USREC - error: {e}")
    recession = pd.DataFrame()


"""# 6. Loading Gold and WTI from Yahoo Finance
# yfinance pulls data from Yahoo Finance—convenient for commodities.
# We only use the Close column.
# Front month futures are a good reflection of the spot price for long periods
# we care about price dynamics"""

COMMODITY_TICKERS = {
    "GC=F": "Gold",
    "CL=F": "WTI",
}

print("\nLoading Gold and WTI from Yahoo Finance...")

commodity_frames = []

for ticker, label in COMMODITY_TICKERS.items():
    try:
        raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

        df = raw[["Close"]].copy()
        df.columns = [label]
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # remove 0 and minus values
        df = df[df[label] > 0]

        commodity_frames.append(df)
        print(f" Done {label:5s} ({ticker}) - {len(df)} lines")

    except Exception as e:
        print(f" ! {label:5s}  ({ticker}) - error: {e}")

commodities = pd.concat(commodity_frames, axis=1, join="outer")
commodities.index.name = "Date"
commodities.ffill(inplace=True)
commodities.dropna(how="all", inplace=True)

print(f"\nCommodities: {commodities.shape[0]} lines × {commodities.shape[1]} columns")
print(f"Period: {commodities.index[0].date()} → {commodities.index[-1].date()}")
print(commodities.tail(3).to_string())


"""# 7. Data quality check 
# Check for gaps before saving.
# DGS30 may have gaps at the beginning—the 30-year bond.
# The US Treasury temporarily stopped issuing them from 2002 to 2006.
# This is a known issue, not a loading error."""

print("\nGaps (% of all lines)")
print("\nTreasuries:")
missing = (treasuries.isna().mean() * 100).round(2)
for col, pct in missing.items():
    flag = "!" if pct > 5 else ""
    print(f"  {col:5s}: {pct:.2f}%{flag}")

print("\nCommodities:")
missing_c = (commodities.isna().mean() * 100).round(2)
for col, pct in missing_c.items():
    flag = "!" if pct > 5 else ""
    print(f"{col:5s}: {pct:.2f}%{flag}")


"""# 8. Downloading to csv
# Each dataset in a separate file.
# The following scripts read from data/"""

print("\nLoading data to CSV files...")

path_treasuries = os.path.join(DATA_DIR, "treasuries.csv")
treasuries.to_csv(path_treasuries)
print(f"treasuries.csv → {path_treasuries}")

path_commodities = os.path.join(DATA_DIR, "commodities.csv")
commodities.to_csv(path_commodities)
print(f"commodities.csv → {path_commodities}")

if not recession.empty:
    path_recession = os.path.join(DATA_DIR, "recession.csv")
    recession.to_csv(path_recession)
    print(f"recession.csv → {path_recession}")

print("\n" + "=" * 60)
print("Data is loaded and saved.")

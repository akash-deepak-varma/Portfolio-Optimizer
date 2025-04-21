import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Step 1: Define Nifty 100 tickers (NSE symbols with '.NS' suffix for Yahoo Finance)
nifty100_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
    "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", 
    "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", 
    "TITAN.NS", "ULTRACEMCO.NS", "TECHM.NS", "POWERGRID.NS", "WIPRO.NS", "NTPC.NS", 
    "JSWSTEEL.NS", "INDUSINDBK.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIGREEN.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "NESTLEIND.NS", "BPCL.NS", "EICHERMOT.NS", 
    "HDFCLIFE.NS", "BRITANNIA.NS", "GRASIM.NS", "HINDALCO.NS", "ONGC.NS", "BAJAJ-AUTO.NS",
    "SBILIFE.NS", "IOC.NS", "SHREECEM.NS", "CIPLA.NS", "DIVISLAB.NS", "ICICIPRULI.NS", 
    "BAJAJFINSV.NS", "GAIL.NS", "DRREDDY.NS", "HEROMOTOCO.NS", "AMBUJACEM.NS", 
    "M&M.NS", "PVR.NS", "ZEE.NS", "INDIACEM.NS", "MCDOWELL-N.NS", "TATACONSUM.NS", 
    "HINDPETRO.NS", "ADANIPOWER.NS", "MUTHOOTFIN.NS", "BERGEPAINT.NS", "BOSCHLTD.NS",
    "CANBK.NS", "LUPIN.NS", "HAVELLS.NS", "MOTHERSUMI.NS", "TATASTEEL.NS", "SAIL.NS", 
    "RBLBANK.NS", "EXIDEIND.NS", "KIRLOSKARIND.NS", "APOLLOHOSP.NS", 
    "CUMMINSIND.NS", "TATAPOWER.NS", "MANAPPURAM.NS", "GODREJCP.NS", "JINDALSTEL.NS", 
    "VOLTAS.NS", "SYNGENE.NS", "IDFCFIRSTB.NS", "NMDC.NS", "PNB.NS", "BANDHANBNK.NS", 
    "ICICIGI.NS", "TATAMETALI.NS", "HDFCAMC.NS", "NATIONALUM.NS", "MINDTREE.NS", 
    "HINDZINC.NS", "UJJIVAN.NS", "COLPAL.NS", "TATAGLOBAL.NS", "MOTILALOFSH.NS", 
    "STERLITE.NS", "FSL.NS", "JUBLFOOD.NS", "UNITEDBREWERIES.NS", "BIOCON.NS"
]


print(f"🎯 {len(nifty100_tickers)} tickers loaded.")

# Step 2: Date range
start_date = "2022-01-01"
end_date = "2023-01-01"

print("Fetching data...")
price_data = yf.download(nifty100_tickers, start=start_date, end=end_date)["Close"]
price_data = price_data.dropna(axis=1)  # Drop stocks with missing data

# Step 4: Calculate daily returns
daily_returns = price_data.pct_change().dropna()

# Step 5: Calculate expected annual returns (mean * 252 trading days)
expected_annual_returns = daily_returns.mean() * 252

# Step 6: Calculate annualized covariance matrix
cov_matrix = daily_returns.cov() * 252

# Save to CSVs
price_data.to_csv("nifty100_prices.csv")
daily_returns.to_csv("nifty100_daily_returns.csv")
expected_annual_returns.to_csv("nifty100_expected_returns.csv")
cov_matrix.to_csv("nifty100_covariance_matrix.csv")

print("✅ Data fetched and saved successfully!")
print(f"📈 Stocks used: {len(price_data.columns)}")

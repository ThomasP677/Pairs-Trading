import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime
from graph import graph_stock_data

# -----------------------------------------------------------------------------
# Zapier settings
# -----------------------------------------------------------------------------
# Replace with your actual Catch‑Hook URL from Zapier.
ZAP_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/123456/abcdef/"  # <-- CHANGE ME


def push_to_zap(timestamp, ticker_a, ticker_b, price_a, price_b, z):
    """Send one JSON row to Zapier so the Zap can insert it in Google Sheets."""
    payload = {
        "timestamp": str(timestamp),  # ISO‑8601 for Sheets & BigQuery friendliness
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "price_a": round(float(price_a), 4),
        "price_b": round(float(price_b), 4),
        "z_score": round(float(z), 4),
    }
    try:
        resp = requests.post(ZAP_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"✓ Pushed to Zapier: {payload}")
    except requests.RequestException as exc:
        print(f"⚠︎ Zapier webhook failed: {exc}")


# -----------------------------------------------------------------------------
# Statistical helper functions
# -----------------------------------------------------------------------------

def compute_range(nums):
    return np.ptp(nums)


def compute_iqr(nums):
    if len(nums) == 0:
        return np.nan
    return np.percentile(nums, 75) - np.percentile(nums, 25)


def compute_correlation(data1, data2):
    if data1.empty or data2.empty:
        return np.nan
    return data1["Close"].corr(data2["Close"])


def compute_cointegration(data1, data2):
    if data1.empty or data2.empty:
        return (np.nan, np.nan, np.nan)
    return sm.tsa.stattools.coint(data1["Close"], data2["Close"])


def fetch_stock_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """Robust download wrapper.

    * First tries `yfinance.download` which works for most tickers.
    * Falls back to the Ticker.history API.
    * Returns *always* a DataFrame (possibly empty) instead of raising.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, threads=False)
        if df.empty:
            # Fallback (rarely needed but defensive)
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
    except Exception as err:
        print(f"Error downloading {ticker}: {err}")
        df = pd.DataFrame()

    if df.empty:
        print(f"⚠︎ No data found for {ticker}.")
    else:
        print(f"{ticker} downloaded ({len(df):,} rows).")
    return df


def analyze_stock_data(data: pd.DataFrame, ticker: str):
    if data.empty:
        print(f"Skipping analysis for {ticker} — no data.")
        return

    range_val = compute_range(data["Close"])
    iqr_val = compute_iqr(data["Close"])

    print(f"\nAnalyzing data for {ticker}:")
    print("-" * 48)
    print(f"Mean Price: {data['Close'].mean():.6f}")
    print(f"Max Price: {data['Close'].max():.6f}")
    print(f"Min Price: {data['Close'].min():.6f}")
    print("*** Spread ***")
    print(f"Variance: {data['Close'].var():.6f}")
    print(f"Range: {range_val:.6f}")
    print(f"Std Dev: {data['Close'].std():.6f}")
    print(f"IQR: {iqr_val:.6f}")


def calculate_z_score(data1: pd.DataFrame, data2: pd.DataFrame, window: int = 30):
    if data1.empty or data2.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    spread = data1['Close'] - data2['Close']
    mean_spread = spread.rolling(window=window).mean()
    std_spread = spread.rolling(window=window).std()
    z_score = (spread - mean_spread) / std_spread
    return spread, z_score


def threshold_based_algorithm(data: pd.DataFrame, z_score: pd.Series, spread: pd.Series):
    if z_score.empty:
        print("No z‑score series to run threshold strategy.")
        return
    data['Z-Score'] = z_score
    data['Spread'] = spread
    upper_threshold, lower_threshold = 2, -2

    data['Position'] = 0
    data.loc[data['Z-Score'] > upper_threshold, 'Position'] = -1
    data.loc[data['Z-Score'] < lower_threshold, 'Position'] = 1

    data['Position'] = data['Position'].shift(1).fillna(0)
    data['Spread Return'] = data['Spread'].pct_change().fillna(0)
    data['Strategy Return'] = data['Position'] * data['Spread Return']
    data['Cumulative Return'] = (1 + data['Strategy Return']).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Cumulative Return'], label='Strategy Return')
    plt.title('Pairs Trading Strategy Based on Z-Score')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()


def greedy_algorithm_sma(data1, data2, window=30):
    return calculate_z_score(data1, data2, window)


def greedy_algorithm_ema(data1, data2, window=30):
    if data1.empty or data2.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    spread = data1['Close'] - data2['Close']
    ema_spread = spread.ewm(span=window, adjust=False).mean()
    std_spread = spread.ewm(span=window, adjust=False).std()
    z_score = (spread - ema_spread) / std_spread
    return spread, z_score


# -----------------------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------------------

def main():
    tickers = ["DAL", "AAL"]
    start_date, end_date = "2022-12-30", "2023-12-30"

    data1 = fetch_stock_data(tickers[0], start_date, end_date)
    data2 = fetch_stock_data(tickers[1], start_date, end_date)

    # Abort early if we have no data (avoids downstream crashes)
    if data1.empty or data2.empty:
        print("❌ One or both tickers returned no data — exiting early.")
        return

    analyze_stock_data(data1, tickers[0])
    analyze_stock_data(data2, tickers[1])

    cointegration = compute_cointegration(data1, data2)
    correlation = compute_correlation(data1, data2)

    print("\nComparing the differences between DAL and AAL:")
    print("-" * 48)
    print(f"Cointegration p-value: {cointegration[1]:.6f}")
    print(f"Correlation: {correlation:.6f}")
    print(f"Std‑Dev Δ: {abs(data1['Close'].std() - data2['Close'].std()):.6f}\n")

    spread_sma, z_score_sma = greedy_algorithm_sma(data1, data2)
    spread_ema, z_score_ema = greedy_algorithm_ema(data1, data2)

    if not z_score_sma.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(z_score_sma.index, z_score_sma, label='SMA Z-Score')
        if not z_score_ema.empty:
            plt.plot(z_score_ema.index, z_score_ema, label='EMA Z-Score', linestyle='--')
        plt.axhline(0, color='black', linestyle='--')
        plt.axhline(2, color='red', linestyle='--', label='Upper Threshold')
        plt.axhline(-2, color='green', linestyle='--', label='Lower Threshold')
        plt.title('Z-Score of the Spread (SMA vs EMA)')
        plt.xlabel('Date')
        plt.ylabel('Z-Score')
        plt.legend()
        plt.grid()
        plt.savefig('z_score_comparison.png')

    combined_data_sma = pd.DataFrame(index=data1.index, data={
        'Close1': data1['Close'],
        'Close2': data2['Close']
    })
    combined_data_ema = combined_data_sma.copy()

    print("\nRunning Threshold-Based Algorithm for SMA Z-Score:")
    threshold_based_algorithm(combined_data_sma, z_score_sma, spread_sma)
    print("\nRunning Threshold-Based Algorithm for EMA Z-Score:")
    threshold_based_algorithm(combined_data_ema, z_score_ema, spread_ema)

    # ---------------------------------------------------------------------
    # Push the latest SMA z‑score & prices to Zapier / Google Sheets
    # ---------------------------------------------------------------------
    if not z_score_sma.empty:
        latest_idx = z_score_sma.last_valid_index()
        push_to_zap(
            timestamp = latest_idx,
            ticker_a  = tickers[0],
            ticker_b  = tickers[1],
            price_a   = data1.loc[latest_idx, 'Close'],
            price_b   = data2.loc[latest_idx, 'Close'],
            z         = z_score_sma.loc[latest_idx],
        )

    # Optional visual compare
    graph_stock_data(tickers, start_date, end_date)


if __name__ == "__main__":
    main()

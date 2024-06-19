import yfinance as yf
import numpy as np
import statsmodels.api as sm
# import pandas as pd
from graph import graph_stock_data


# Function to Compute Range
def compute_range(nums):
    return np.ptp(nums)


# Function to compute Interquartile Range (IQR)
def compute_iqr(nums):
    return np.percentile(nums, 75) - np.percentile(nums, 25)


# Function to compute correlation
def compute_correlation(data1, data2):
    return data1["Close"].corr(data2["Close"])


# Function to compute cointegration
    """ This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in 1st stage regression, i.e. in
    cointegrating equation. """


def compute_cointegration(data1, data2):
    cointegration_result = sm.tsa.stattools.coint(data1['Close'], data2['Close'])
    return cointegration_result


# Function to fetch stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    stock = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d')
    if not stock.empty:
        print(f"{ticker} downloaded successfully.")
    else:
        print(f"Failed to download {ticker}.")
    return stock


# Analyze stock data
def analyze_stock_data(data, ticker):

    # Compute range and IQR
    range_val = compute_range(data["Close"])
    iqr_val = compute_iqr(data["Close"])

    print(f"\n\nAnalyzing data for {ticker}:")
    print("------------------------------------------------")
    print(f"{ticker} Mean Price: {data['Close'].mean():.6f}")
    print(f"{ticker} Maximum Price: {data['Close'].max():.6f}")
    print(f"{ticker} Minimum Price: {data['Close'].min():.6f}\n")
    print("*** Spread ***")
    print(f"{ticker} Variance Price: {data['Close'].var():.6f}")
    print(f"{ticker} Range: {range_val:.6f}")
    print(f"{ticker} Standard Deviation: {data['Close'].std():.6f}")
    print(f"{ticker} Interquartile Range (IQR): {iqr_val:.6f}\n")


# Main function to fetch and analyze data
def main():
    tickers = ["DAL", "AAL"]
    ticker1 = "DAL"  # Delta Airlines
    ticker2 = "AAL"  # American Airlines
    start_date = "2022-12-30"
    end_date = "2023-12-30"

    # Fetch stock data
    data1 = fetch_stock_data(ticker1, start_date, end_date, interval='1d')
    data2 = fetch_stock_data(ticker2, start_date, end_date, interval='1d')

    # Analyze the stock data for each ticker
    analyze_stock_data(data1, ticker1)
    analyze_stock_data(data2, ticker2)

    # Compute cointegration with other ticker
    cointegration = compute_cointegration(data1, data2)

    # Compare the differences between the two stocks
    print("\nComparing the differences between DAL and AAL:")
    print("------------------------------------------------")
    print(f"Cointegration p-value: {cointegration[1]:.6f}")
    correlation = compute_correlation(data1, data2)
    print(f"Correlation: {correlation:.6f}")
    std_diff = abs(data1["Close"].std() - data2["Close"].std())
    print(f"Standard Deviation Difference: {std_diff:.6f}\n")

    # Graph the stock data, Un comment below to include graph data
    graph_data = graph_stock_data(tickers, start_date, end_date)
    for ticker in tickers:
        return (graph_data)


if __name__ == "__main__":
    main()

import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import pandas as pd


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
def compute_cointegration(data1, data2):
    """
    This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in the 1st stage regression, i.e. in
    cointegrating equation.
    """
    cointegration_result = sm.tsa.stattools.coint(data1['Close'], data2['Close'])
    return cointegration_result


# Function to generate AR process
def generate_ar_process(lags, coefs, length):
    # cast coefs to np array
    coefs = np.array(coefs)

    # initial values
    series = [np.random.normal() for _ in range(lags)]

    for _ in range(length):
        # get previous values of the series, reversed
        prev_vals = series[-lags:][::-1]

        # get new value of time series
        new_val = np.sum(np.array(prev_vals) * coefs) + np.random.normal()

        series.append(new_val)

    return np.array(series)


# Function to fetch stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    stock = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
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


# Function to visualize stock data
def visualize_stock_data(tickers, start_date, end_date):
    combined_data = {}
    for ticker in tickers:
        combined_data[ticker] = fetch_stock_data(ticker, start_date, end_date)['Close']


# Main function to fetch and analyze data
def main():
    tickers = ["DAL", "AAL"]
    start_date = "2022-12-30"
    end_date = "2023-12-30"

    # Fetch stock data
    data1 = fetch_stock_data(tickers[0], start_date, end_date, interval='1d')
    data2 = fetch_stock_data(tickers[1], start_date, end_date, interval='1d')

    # Analyze the stock data for each ticker
    analyze_stock_data(data1, tickers[0])
    analyze_stock_data(data2, tickers[1])

    # Compute cointegration and correlation
    cointegration = compute_cointegration(data1, data2)

    # Compare the differences between the two stocks
    print("\nComparing the differences between DAL and AAL:")
    print("------------------------------------------------")
    print(f"Cointegration p-value: {cointegration[1]:.6f}")
    correlation = compute_correlation(data1, data2)
    print(f"Correlation: {correlation:.6f}")
    std_diff = abs(data1["Close"].std() - data2["Close"].std())
    print(f"Standard Deviation Difference: {std_diff:.6f}\n")

    # Graph the stock data
    graph_data = visualize_stock_data(tickers, start_date, end_date)
    for ticker in tickers:
        return (graph_data)


if __name__ == "__main__":
    main()

import yfinance as yf
import numpy as np
import statsmodels.api as sm
import pandas as pd


# Function to Compute Range
def compute_range(nums):
    return np.max(nums) - np.min(nums)


# Function to compute Interquartile Range (IQR)
def compute_iqr(nums):  # To Calculate IQR it is Q3 - Q1
    q75, q25 = np.percentile(nums, [75, 25])
    return q75 - q25


# Function to compute correlation
def compute_correlation(data1, data2):
    correlation = np.corrcoef(data1['Close'], data2['Close'])[0, 1]
    return correlation


# Function to compute cointegration
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
    print(f"\n\nAnalyzing data for {ticker}:")
    print("------------------------------------------------")

    print(f"{ticker} Mean Price: {abs(data['Close'].mean()):.2f}")
    print(f"{ticker} Variance Price: {abs(data['Close'].var()):.2f}")
    print(f"{ticker} Maximum Price: {abs(data['Close'].max()):.2f}")
    print(f"{ticker} Minimum Price: {abs(data['Close'].min()):.2f}")
    print(f"{ticker} Standard Deviation: {abs(data['Close'].std()):.2f}\n")


# Function definitions for statistical calculations
def compute_maximum(nums):
    return nums.max()


def compute_minimum(nums):
    return nums.min()


def compute_mean(nums):
    return nums.mean()


def compute_variance(nums):
    return nums.var()


def compute_std_dev(nums):
    return nums.std()


# Main function to fetch and analyze data
def main():
    ticker1 = "DAL"  # Delta Airlines
    ticker2 = "AAL"  # American Airlines
    start_date = "2022-12-31"
    end_date = "2023-12-30"

    # Fetch stock data
    data1 = fetch_stock_data(ticker1, start_date, end_date)
    data2 = fetch_stock_data(ticker2, start_date, end_date)

    # Analyze the stock data for each ticker
    analyze_stock_data(data1, ticker1)
    analyze_stock_data(data2, ticker2)

    # Compute cointegration with other ticker
    cointegration = compute_cointegration(data1, data2)

    # Compute range and IQR
    close_prices_DAL = data1["Close"]
    range_val1 = compute_range(close_prices_DAL)
    iqr_val1 = compute_iqr(close_prices_DAL)
    close_prices_AAL = data2["Close"]
    range_val2 = compute_range(close_prices_AAL)
    iqr_val2 = compute_iqr(close_prices_AAL)

    # Compare the differences between the two stocks
    print("\nComparing the differences between DAL and AAL:")
    print("------------------------------------------------")
    print(f"{ticker1} Range: {range_val1:.6f}")
    print(f"{ticker2} Range: {range_val2:.2f}")
    print(f"{ticker1} Interquartile Range (IQR): {iqr_val1:.6f}")
    print(f"{ticker2} Interquartile Range (IQR): {iqr_val2:.6f}")
    print(f"Cointegration p-value: {cointegration[1]:.6f}")
    correlation = compute_correlation(data1, data2)
    print(f"Correlation: {correlation:.6f}")
    print(
        f"Standard Deviation Difference: {abs(data1['Close'].std() - data2['Close'].std()):.6f}\n"
    )

    # Graph the stock data, Un comment below to include graph data
    # graph_data = graph_stock_data(ticker, start_date, end_date)
    # return (graph_data)


if __name__ == "__main__":
    main()

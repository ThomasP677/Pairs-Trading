import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from graph import graph_stock_data


# Function to compute range
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
    cointegration_result = sm.tsa.stattools.coint(data1['Close'], data2['Close'])
    return cointegration_result


# Function to fetch stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    stock = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
    if not stock.empty:
        print(f"{ticker} downloaded successfully.")
    else:
        print(f"Failed to download {ticker}.")
    return stock


# Function to analyze stock data
def analyze_stock_data(data, ticker):
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


# Function to calculate the z-score of the spread
def calculate_z_score(data1, data2, window=30):
    # Calculate the spread
    spread = data1['Close'] - data2['Close']

    # Calculate rolling mean and standard deviation of the spread
    mean_spread = spread.rolling(window=window).mean()
    std_spread = spread.rolling(window=window).std()

    # Calculate the z-score
    z_score = (spread - mean_spread) / std_spread

    return spread, z_score


# Threshold-based trading algorithm
def threshold_based_algorithm(data, z_score, spread, threshold=1.5):
    # Add the z_score and spread to the data DataFrame
    data['Z-Score'] = z_score
    data['Spread'] = spread

    # Define thresholds
    upper_threshold = 2
    lower_threshold = -2

    # Define positions
    data['Position'] = 0
    data.loc[data['Z-Score'] > upper_threshold, 'Position'] = -1  # Short position
    data.loc[data['Z-Score'] < lower_threshold, 'Position'] = 1   # Long position

    # Shift positions to simulate actual trading
    data['Position'] = data['Position'].shift(1).fillna(0)

    # Calculate daily returns of the spread
    data['Spread Return'] = data['Spread'].pct_change().fillna(0)

    # Calculate strategy returns
    data['Strategy Return'] = data['Position'] * data['Spread Return']

    # Calculate cumulative returns
    data['Cumulative Return'] = (1 + data['Strategy Return']).cumprod()

    # Plot the cumulative return
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Cumulative Return'], label='Strategy Return')
    plt.title('Pairs Trading Strategy Based on Z-Score')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()


# Greedy Algorithm 1: Simple Moving Average (SMA) based z-score
def greedy_algorithm_sma(data1, data2, window=30):
    spread, z_score = calculate_z_score(data1, data2, window)
    return spread, z_score


# Greedy Algorithm 2: Exponential Moving Average (EMA) based z-score
def greedy_algorithm_ema(data1, data2, window=30):
    # Calculate the spread
    spread = data1['Close'] - data2['Close']

    # Calculate EMA of the spread
    ema_spread = spread.ewm(span=window, adjust=False).mean()
    std_spread = spread.ewm(span=window, adjust=False).std()

    # Calculate the z-score
    z_score = (spread - ema_spread) / std_spread

    return spread, z_score


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

    # Calculate and plot z-score for both greedy algorithms
    spread_sma, z_score_sma = greedy_algorithm_sma(data1, data2)
    spread_ema, z_score_ema = greedy_algorithm_ema(data1, data2)

    plt.figure(figsize=(10, 5))
    plt.plot(z_score_sma.index, z_score_sma, label='SMA Z-Score')
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

    # Combine data for threshold-based algorithm
    combined_data_sma = pd.DataFrame(index=data1.index)
    combined_data_sma['Close1'] = data1['Close']
    combined_data_sma['Close2'] = data2['Close']

    combined_data_ema = combined_data_sma.copy()

    # Run the threshold-based algorithm for both z-scores
    print("\nRunning Threshold-Based Algorithm for SMA Z-Score:")
    threshold_based_algorithm(combined_data_sma, z_score_sma, spread_sma)
    print("\nRunning Threshold-Based Algorithm for EMA Z-Score:")
    threshold_based_algorithm(combined_data_ema, z_score_ema, spread_ema)

    # Graph the stock data
    combined, pct_change, cum_returns = graph_stock_data(tickers, start_date, end_date)
    return combined, pct_change, cum_returns


if __name__ == "__main__":
    main()

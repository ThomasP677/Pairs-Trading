import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Use the Agg backend for Matplotlib
plt.switch_backend('Agg')


def download_stock_data(ticker, start_date, end_date, interval='1d'):
    try:
        return yf.download(ticker, start=start_date, end=end_date, interval=interval)['Adj Close']
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None


def plot_adjusted_closing_prices(combined, tickers):
    plt.figure()
    for col in combined.columns:
        plt.plot(combined.index, combined[col], label=f"{col} Adj Closing Prices")
    plt.legend(loc="upper left")
    plt.title("Adjusted Closing Prices Comparison")
    plt.ylabel("Closing Prices")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig('adjusted_closing_prices.png')


def plot_percent_change(pct_change, tickers):
    plt.figure()
    for col in pct_change.columns:
        plt.plot(pct_change.index, pct_change[col], label=f"{col} Percent Change")
    plt.legend(loc="upper left")
    plt.title("Percent Change Comparison")
    plt.ylabel("Percent Change")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig('percent_change.png')


def plot_cumulative_returns(cum_returns, tickers):
    plt.figure()
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=f"{col} Cumulative Returns")
    plt.legend(loc="upper left")
    plt.title("Cumulative Returns Comparison")
    plt.ylabel("Cumulative Returns")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig('cumulative_returns.png')


# Add more functions for other types of plots...

def graph_stock_data(tickers, start_date, end_date, interval='1d'):
    try:
        combined = pd.DataFrame({ticker: download_stock_data(ticker, start_date, end_date, interval) for ticker in tickers})

        # Adjusted Closing Prices Plot
        plot_adjusted_closing_prices(combined, tickers)

        if len(tickers) == 2:
            pct_change = combined.pct_change().dropna()
            
            # Percent Change Plot
            plot_percent_change(pct_change, tickers)

            # Cumulative Returns Plot
            cum_returns = (1 + pct_change).cumprod() - 1
            plot_cumulative_returns(cum_returns, tickers)

        # Histogram Line Graph for Mean Prices
        plt.figure()
        combined.mean().plot(kind='hist', bins=20, edgecolor='black', histtype='step', linewidth=2)
        plt.title('Histogram of Mean Prices')
        plt.grid(True)
        plt.legend(tickers)
        plt.savefig('histogram_mean.png')

        # Histogram Line Graph for Max and Min Prices
        plt.figure()
        combined.agg(['max', 'min']).T.plot(kind='hist', bins=20, edgecolor='black', histtype='step', linewidth=2)
        plt.title('Histogram of Max and Min Prices')
        plt.grid(True)
        plt.legend(['Max', 'Min'])
        plt.savefig('histogram_max_min.png')

        # Histogram Line Graph for Spread (Max - Min)
        plt.figure()
        (combined.max() - combined.min()).plot(kind='hist', bins=20, edgecolor='black', histtype='step', linewidth=2)
        plt.title('Histogram of Price Spread')
        plt.grid(True)
        plt.legend(tickers)
        plt.savefig('histogram_spread.png')

        # Scatter Plot for Variance
        plt.figure()
        variance = combined.var()
        plt.scatter(variance.index, variance.values)
        plt.title('Scatter Plot of Price Variance')
        plt.ylabel('Variance')
        plt.xlabel('Stocks')
        plt.grid(True)
        plt.savefig('scatter_variance.png')

        # Histogram Line Graph for Interquartile Range (IQR)
        plt.figure()
        (combined.quantile(0.75) - combined.quantile(0.25)).plot(kind='hist', bins=20, edgecolor='black', histtype='step', linewidth=2)
        plt.title('Histogram of Interquartile Range (IQR)')
        plt.grid(True)
        plt.legend(tickers)
        plt.savefig('histogram_iqr.png')

        # Histogram for Standard Deviation
        plt.figure()
        combined.std().plot(kind='hist', bins=20, edgecolor='black', histtype='step', linewidth=2)
        plt.title('Histogram of Standard Deviation')
        plt.grid(True)
        plt.legend(tickers)
        plt.savefig('hist_std_dev.png')

        return combined, pct_change if 'pct_change' in locals() else None, cum_returns if 'cum_returns' in locals() else None

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None

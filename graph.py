import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def graph_stock_data(tickers, start_date, end_date, interval='1d'):
    start_date = "2023-01-1"
    end_date = "2023-12-31"

    tickers = ["DAL", "AAL"]

    combined = pd.DataFrame()

    try:
        for ticker in tickers:
            data = yf.download(ticker, start=start_date, end=end_date)
            combined[ticker] = data[['Adj Close']]

        for ticker in tickers:
            plt.plot(combined[ticker], label=ticker)

        plt.legend(loc="upper right")
        plt.show()

    except Exception as e:
        print(f"Error occured: {e}")

    print(combined)

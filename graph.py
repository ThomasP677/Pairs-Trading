import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def graph_stock_data(tickers, start_date, end_date, interval='1d'):
    def download_stock_data(tickers):
        return yf.download(tickers, start=start_date, end=end_date, interval=interval)['Adj Close']

    try:
        combined = pd.DataFrame({ticker: download_stock_data(ticker) for ticker in tickers})

        plt.figure()
        for col in combined.columns:
            plt.plot(combined.index, combined[col], label=f"{col} Adj Closing Prices")

        plt.legend(loc="upper left")
        plt.title("Adjusted Closing Prices Comparison")
        plt.ylabel("Closing Prices")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()

        if len(tickers) == 2:
            pct_change = combined.pct_change().dropna()

            plt.figure()
            for col in pct_change.columns:
                plt.plot(pct_change.index, pct_change[col], label=f"{col} Percent Change")

            plt.legend(loc="upper left")
            plt.title("Percent Change Comparison ")
            plt.ylabel("Percent Change")
            plt.xlabel("Date")
            plt.grid(True)
            plt.show()

            cum_returns = (1 + pct_change).cumprod() - 1

            plt.figure()
            for col in cum_returns.columns:
                plt.plot(cum_returns.index, cum_returns[col], label=f"{col} Cumulative Returns")

            plt.legend(loc="upper left")
            plt.title("Cumulative Returns Comparison")
            plt.ylabel("Cumulative Returns")
            plt.xlabel("Date")
            plt.grid(True)
            plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")

    print(combined)

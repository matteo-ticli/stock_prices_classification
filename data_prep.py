"""
We are going to focus at first on the Nasdaq100. We are going to scrape each trading day in the past 20 years.
    The scope is to create a strategy which is able to beat the market. We are going to base our analisys on:
        1. Technical indicators
        2. Differrent stocks for the correlation
        3. Dilated timeframe for each day

    PROCEDURE:
        1. Get data using Yahoo Finance
        2. Store data in a df
        3. Build technical indicators
        4. Calculate the "images"
        5. For each day label the image: SELL (0), BUY (1), HOLD (2)
        6. Data scaling
        7. Connect AWS
        8. Create the CNN
        9. Connects it to LSTM
        10. Test multiple models
        11. Conclusions
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
import pandas_datareader.data as web
import requests
import time


tickers = ['^NDX', '^GSPC', '^DJI', '^RUT', '^NYA', '^GDAXI', '^N225', '^FCHI', '^HSI', '000001.SS']
## figure out ^FTSE


def get_data(tickers, start_date='1999-01-01', end_date=dt.today()):
    ticker_dict = dict()
    for idx, ticker in enumerate(tickers):
        try:
            df_ticker = web.DataReader(ticker, 'yahoo', start_date, end_date)
            ticker_dict[ticker] = df_ticker['Adj Close']
        except:
            pass
    df = pd.DataFrame(ticker_dict)
    print(df)
    return df


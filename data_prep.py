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

import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import technical_indicators as ti


tickers = ['^NDX', '^GSPC', '^DJI', '^RUT', '^NYA', '^GDAXI', '^N225', '^FCHI', '^HSI', '000001.SS']
tickers_name = ['NASDAQ', 'SP500', 'DJI', 'RUSSEL', 'NYSE', 'DAX', 'NIKKEI 225', 'CAC 40', 'HANG SENG', 'SSE']
main_asset = 'NASDAQ'
time_delta = 10
tech_indicators = 10

df_ticker['Return'] = df_ticker[:, 'Close'].diff()/df_ticker[:, 'Close']

def get_data(tickers, start_date='2000-01-01', end_date='2021-1-1'):
    path = os.getcwd() + '/data/'
    print(path)
    for idx, ticker in enumerate(tickers):
        try:
            df_ticker = web.DataReader(ticker, 'yahoo', start_date, end_date).drop(labels=['Open', 'Volume', 'Adj Close'], axis=1)
            df_ticker = df_ticker.reset_index().dropna()
            df_ticker.to_csv(path+tickers_name[idx]+'.csv', index=False)
        except:
            continue


def create_dfs(directory=(os.getcwd()+'/data/')):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        df = pd.read_csv(path)

        ti.simple_moving_average(df)
        ti.weighted_moving_average(df)
        ti.momentum(df)
        ti.stochastic_k(df)
        ti.stochastic_d(df)
        ti.moving_average_convergence_divergence(df)
        ti.relative_strength_index(df)
        ti.williams_r(df)
        ti.commodity_channel_index(df)
        ti.accumulation_distribution_oscillator(df)

        df.to_csv(path, index=False)

create_dfs()


def load_assets_dfs(directory=(os.getcwd()+'/data/'), ma=main_asset):
    dfs_dict = dict()
    dfs_dict[ma] = pd.read_csv(os.path.join(directory, ma+'.csv'))

    for filename in os.listdir(directory):
        if filename == ma+'.csv':
            continue
        filename_split = filename.split('.', 1)
        dfs_dict[filename_split[0]] = pd.read_csv(os.path.join(directory, filename))

    return dfs_dict

dfs_dict = load_assets_dfs()


def calculate_returns(dfs_dict):
    for i, asset in enumerate(dfs_dict):
        dfs_dict[asset]['Return'] = dfs_dict[asset]['Close'].diff()/dfs_dict[asset]['Close']


def order_correlated_assets(dfs_dict, day, ma=main_asset):
    arr = np.zeros((time_delta+1, len(dfs_dict)))
    list_asset = list()
    for i, asset in enumerate(list(dfs_dict.keys())):
        arr[:, i] = dfs_dict[asset].loc[day-time_delta:day, 'Return']
        list_asset.append(asset)

    df = pd.DataFrame(data=arr, columns=list_asset)

    corr_matrix = df.corr()

    corr_matrix_ordered = corr_matrix.sort_values(by=['NASDAQ'])
    indexes = list(corr_matrix_ordered.index)
    indexes.reverse()
    return indexes


def create_tensor(dfs_dict, ma=main_asset):
    tensor = dict()
    for day in range(20, len(dfs_dict[ma])):
        indexes = order_correlated_assets(dfs_dict, day)
        z, y, x = time_delta, tech_indicators, len(dfs_dict)
        pivot = np.empty((z, y, x))
        for sub_day in range(time_delta, 0, -1):
            for item in ordered_corr:
                pass


def label_tensor(df):
    pass

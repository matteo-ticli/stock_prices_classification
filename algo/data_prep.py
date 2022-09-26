import os
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import date, timedelta
import technical_indicators as ti
import time
from ta import momentum, trend
import ta

def get_data(directory, tickers, tickers_name, start_date=date(2000,1,1), end_date=(2016,1,1), time_range = 'd'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for idx, ticker in enumerate(tickers):
        try:
            df_ticker = pdr.get_data_yahoo(ticker, start_date, end_date, interval=time_range)
            df_ticker = df_ticker.reset_index().dropna()
            df_ticker.to_csv(directory + '/' + tickers_name[idx] + '.csv', index=False)
            print(directory + '/' + tickers_name[idx])
        except:
            print(idx)
            continue


def simple_moving_average(df, time_delta=9):
    df['SMA'] = np.zeros((len(df), 1))
    df['SMA'] = df['Close'].rolling(time_delta).mean()

def weighted_moving_average(df, time_delta=9):
    wma = trend.WMAIndicator(close=df['Close'], window=time_delta)
    df['WMA'] = wma.wma()


def momentum(df, time_delta=9):
    df['MOM'] = np.zeros((len(df), 1))
    df.loc[:time_delta-2, 'MOM'] = np.nan
    for i in range(time_delta-1, len(df)):
        mom = df.iloc[i]['Close'] - df.iloc[i-time_delta+1]['Close']
        df.at[i, 'MOM'] = mom

def stochastic_k(df, time_delta=9):
    df['K %'] = np.zeros((len(df), 1))
    df.loc[:time_delta - 2, 'K %'] = np.nan
    for i in range(time_delta-1, len(df)):
        low_min = min(df.loc[i-time_delta+1:i, 'Low'])
        high_max = max(df.loc[i-time_delta+1:i, 'High'])
        k = (df.iloc[i]['Close'] - low_min)/(high_max - low_min)*100
        df.at[i, 'K %'] = k

def stochastic_d(df, time_delta=9):
    df['D %'] = np.zeros((len(df), 1))
    df['D %'] = df['K %'].rolling(time_delta).mean()

def exponential_moving_average(df, time_delta=9):
    EMA = 'EMA' + str(time_delta)
    df[EMA] = np.zeros((len(df), 1))
    for i in range(1, len(df)):
        df.at[i, EMA] = df.iloc[i-1][EMA] + ((2/(time_delta+1)) * (df.iloc[i]['Close'] - df.iloc[i-1][EMA]))

def moving_average_convergence_divergence(df, time_delta=9, time_delta_ema_1=12, time_delta_ema_2=26):
    exponential_moving_average(df, time_delta_ema_1)
    exponential_moving_average(df, time_delta_ema_2)
    df['MACD'] = np.zeros((len(df), 1))
    for i in range(1, len(df)):
        ema_diff = df.iloc[i]['EMA'+str(time_delta_ema_1)] - df.iloc[i]['EMA'+str(time_delta_ema_2)]
        df.at[i, 'MACD'] = df.iloc[i-1]['MACD'] + ((2/(time_delta+1)) * (ema_diff - df.iloc[i-1]['MACD']))

def relative_strength_index(df, time_delta=9):
    rsi = ta.momentum.RSIIndicator(close=df['Close'], window=time_delta)
    df['RSI'] = rsi.rsi()

def williams_r(df, time_delta=9):
    df['W %R'] = ta.momentum.williams_r(high=df['High'], low=df['Low'], close=df['Close'], lbp=time_delta)

def commodity_channel_index(df, time_delta=9):
    cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=time_delta)
    df['CCI'] = cci.cci()

def accumulation_distribution_oscillator(df):
    df['AD'] = np.zeros((len(df), 1))
    for i in range(1, len(df)):
        try:
            df.at[i, 'AD'] = (df.iloc[i]['High'] - df.iloc[i-1]['Close'])/(df.iloc[i]['High'] - df.iloc[i]['Low'])
        except:
            df.at[i, 'AD'] = 0.0


def create_csv(directory):
    for filename in os.listdir(directory):
        if filename[-4:] != '.csv':
            continue
        path = os.path.join(directory, filename)
        df = pd.read_csv(path)

        simple_moving_average(df)
        weighted_moving_average(df)
        momentum(df)
        stochastic_k(df)
        stochastic_d(df)
        moving_average_convergence_divergence(df)
        relative_strength_index(df)
        williams_r(df)
        commodity_channel_index(df)
        accumulation_distribution_oscillator(df)

        df.to_csv(path, index=False)


def load_assets_dfs(directory, main_asset):
    dfs_dict = dict()
    dfs_dict[main_asset] = pd.read_csv(os.path.join(directory, main_asset + '.csv'))

    for filename in os.listdir(directory):
        if filename == main_asset + '.csv':
            continue
        if filename[-4:] != '.csv':
            continue
        filename_split = filename.split('.', 1)
        dfs_dict[filename_split[0]] = pd.read_csv(os.path.join(directory, filename))

    ## clean rows that do not share same date
    start_date = date(year=2000, month=1, day=1)
    end_date = date(year=2021, month=1, day=1)
    current_date = start_date

    while current_date <= end_date:

        idx_list = list()
        for asset in dfs_dict.keys():
            df = dfs_dict[asset]
            idx = df.index[df['Date'] == current_date.isoformat()].to_list()
            if len(idx) != 0:
                idx_list.append(idx)

        if len(idx_list) != len(dfs_dict) and len(idx_list) != 0:
            for asset in dfs_dict.keys():
                df = dfs_dict[asset]
                idx = df.index[df['Date'] == current_date.isoformat()].to_list()
                if len(idx) != 0:
                    dfs_dict[asset].drop(labels=idx[0], axis=0, inplace=True)
                    dfs_dict[asset].reset_index(drop=True, inplace=True)

        current_date += timedelta(days=1)

    return dfs_dict


def calculate_returns(dfs_dict):
    for i, asset in enumerate(dfs_dict):
        dfs_dict[asset]['Return'] = dfs_dict[asset]['Close'].diff() / dfs_dict[asset]['Close']
    return dfs_dict


def order_correlated_assets(dfs_dict, day, time_delta):
    arr = np.zeros((time_delta, len(dfs_dict)))
    list_asset = list()
    for i, asset in enumerate(list(dfs_dict.keys())):
        arr[:, i] = dfs_dict[asset].loc[day-time_delta+1:day, 'Return']
        list_asset.append(asset)
    df = pd.DataFrame(data=arr, columns=list_asset)
    corr_matrix = df.corr()
    corr_matrix_ordered = corr_matrix.sort_values(by=[main_asset])
    ordered_indexes = list(corr_matrix_ordered.index)
    ordered_indexes.reverse()
    return ordered_indexes


def label_tensor(dfs_dict, main_asset, day):
    if dfs_dict[main_asset].loc[day, 'Close'] < dfs_dict[main_asset].loc[day + 1, 'Close']:
        label = 1
    else:
        label = 0
    return label


def create_tensor(dfs_dict, main_asset, time_delta, tech_indicators, start_date_num=50, end_date_num=100):

    t, z, y, x = end_date_num-start_date_num, time_delta, tech_indicators, len(dfs_dict)
    tensor = np.zeros((t, z, y, x))
    labels = np.zeros((t, ))

    for idx, day in enumerate(range(start_date_num, end_date_num)):
        label = label_tensor(dfs_dict, main_asset, day)
        ordered_indexes = order_correlated_assets(dfs_dict, day, time_delta)
        if day == 50:
            print(dfs_dict[main_asset].loc[day, 'Date'])
        for i, subday in enumerate(range(day - time_delta, day)):

            for j, asset in enumerate(ordered_indexes):

                # SMA
                if dfs_dict[asset].loc[subday, 'Close'] > dfs_dict[asset].loc[subday, 'SMA']:
                    tensor[idx, i, 0, j] = 1
                if dfs_dict[asset].loc[subday, 'Close'] <= dfs_dict[asset].loc[subday, 'SMA']:
                    tensor[idx, i, 0, j] = 0

                # WMA
                if dfs_dict[asset].loc[subday, 'Close'] > dfs_dict[asset].loc[subday, 'WMA']:
                    tensor[idx, i, 1, j] = 1
                if dfs_dict[asset].loc[subday, 'Close'] <= dfs_dict[asset].loc[subday, 'WMA']:
                    tensor[idx, i, 1, j] = 0

                # Mom
                if dfs_dict[asset].loc[subday, 'MOM'] > 0:
                    tensor[idx, i, 2, j] = 1
                if dfs_dict[asset].loc[subday, 'MOM'] <= 0:
                    tensor[idx, i, 2, j] = 0

                # K%
                if dfs_dict[asset].loc[subday, 'K %'] > dfs_dict[asset].loc[subday - 1, 'K %']:
                    tensor[idx, i, 3, j] = 1
                if dfs_dict[asset].loc[subday, 'K %'] <= dfs_dict[asset].loc[subday - 1, 'K %']:
                    tensor[idx, i, 3, j] = 0

                # D%
                if dfs_dict[asset].loc[subday, 'D %'] > dfs_dict[asset].loc[subday - 1, 'D %']:
                    tensor[idx, i, 4, j] = 1
                if dfs_dict[asset].loc[subday, 'D %'] <= dfs_dict[asset].loc[subday - 1, 'D %']:
                    tensor[idx, i, 4, j] = 0

                # MACD
                if dfs_dict[asset].loc[subday, 'MACD'] > dfs_dict[asset].loc[subday - 1, 'MACD']:
                    tensor[idx, i, 5, j] = 1
                if dfs_dict[asset].loc[subday, 'MACD'] <= dfs_dict[asset].loc[subday - 1, 'MACD']:
                    tensor[idx, i, 5, j] = 0

                # RSI
                if dfs_dict[asset].loc[subday, 'RSI'] <= 30 or dfs_dict[asset].loc[subday, 'RSI'] > dfs_dict[asset].loc[
                    subday - 1, 'RSI']:
                    tensor[idx, i, 6, j] = 1
                if dfs_dict[asset].loc[subday, 'RSI'] >= 70 or dfs_dict[asset].loc[subday, 'RSI'] <= \
                        dfs_dict[asset].loc[subday - 1, 'RSI']:
                    tensor[idx, i, 6, j] = 0

                # W %R
                if dfs_dict[asset].loc[subday, 'W %R'] > dfs_dict[asset].loc[subday - 1, 'W %R']:
                    tensor[idx, i, 7, j] = 1
                if dfs_dict[asset].loc[subday, 'W %R'] <= dfs_dict[asset].loc[subday - 1, 'W %R']:
                    tensor[idx, i, 7, j] = 0

                # CCI
                if dfs_dict[asset].loc[subday, 'CCI'] < -200 or dfs_dict[asset].loc[subday, 'CCI'] > \
                        dfs_dict[asset].loc[subday - 1, 'CCI']:
                    tensor[idx, i, 8, j] = 1
                if dfs_dict[asset].loc[subday, 'CCI'] > 200 or dfs_dict[asset].loc[subday, 'RSI'] <= \
                        dfs_dict[asset].loc[subday - 1, 'RSI']:
                    tensor[idx, i, 8, j] = 0

                # AD
                if dfs_dict[asset].loc[subday, 'AD'] > dfs_dict[asset].loc[subday - 1, 'AD']:
                    tensor[idx, i, 9, j] = 1
                if dfs_dict[asset].loc[subday, 'AD'] <= dfs_dict[asset].loc[subday - 1, 'AD']:
                    tensor[idx, i, 9, j] = 0

        labels[idx] = label

    return tensor, labels


def execute_data_prep(directory, time_delta, tech_indicators, tickers, tickers_name, main_asset,
                      start_date, end_date, timerange, start_date_num, tensor_name, labels_name):

    t0 = time.time()

    get_data(directory, tickers, tickers_name, start_date=start_date, end_date=end_date, time_range=timerange)
    create_csv(directory)
    dfs_dict = load_assets_dfs(directory, main_asset)
    dfs_dict = calculate_returns(dfs_dict)
    TOT_LEN = len(dfs_dict[main_asset])-1
    tensor, labels = create_tensor(dfs_dict, main_asset, time_delta, tech_indicators, start_date_num=start_date_num, end_date_num=TOT_LEN)

    ## save a copy in directory of each asset of dfs_dict
    for asset in dfs_dict.keys():
        dfs_dict[asset].to_csv(directory + '/' + asset + '.csv', index=False)

    np.save(tensor_name, tensor)
    np.save(labels_name, labels)

    t1 = time.time()
    delta_t = t1 - t0
    print(f"The data preparation process lasted: {delta_t} seconds")

    return tensor, labels, dfs_dict


## execute this code with these parameters ###

directory = os.getcwd() + '/data'



tickers = ['^GSPC', '^NDX', '^DJI', '^RUT', '^NYA', '^GDAXI', '^N225', '^FCHI', '^HSI', '000001.SS']
tickers_name = ['SP500', 'NASDAQ', 'DJI', 'RUSSEL', 'NYSE', 'DAX', 'NIKKEI 225', 'CAC 40', 'HANG SENG', 'SSE']
main_asset = 'SP500'
time_delta = 10
tech_indicators = 10

start_date = '2000-01-01'
end_date = '2021-01-01'
start_date_num = 50

tensor_name = directory + '/tensor_SP500.npy'
labels_name = directory + '/labels_SP500.npy'

timerange = 'd'

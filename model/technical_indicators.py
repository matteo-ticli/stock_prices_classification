import numpy as np
import pandas as pd
from ta import momentum, trend
import ta

"""
We are going to define ten technical indicators that refer to day trading strategies.
We are going to rely, for some of these indicators, on the python library ta.
We are not using the result of the technical indicators but rather a deterministic trend signal (0 or 1)
The value of time_delta = 9 due to the fact that wer want to exploit a daily strategy
"""

# df = pd.read_csv('/Users/mticli/Documents/BOCCONI/FINAL PROJECT/CNN_LSTM-stock-prices-prediction/data/NASDAQ.csv')


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


# simple_moving_average(df)
# weighted_moving_average(df)
# momentum(df)
# stochastic_k(df)
# stochastic_d(df)
# exponential_moving_average(df)
# moving_average_convergence_divergence(df)
# relative_strength_index(df)
# williams_r(df)
# commodity_channel_index(df)
# accumulation_distribution_oscillator(df)
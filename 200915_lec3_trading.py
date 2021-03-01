#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:12:32 2020

@author: mamsds
"""

import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

from backtesting import Backtest , Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

class simple_moving_average_Cross(Strategy):
    def init(self):
        Closel = self.data.Close
        self.ma1 = self.I(SMA, Closel, 10)
        self.ma2 = self.I(SMA, Closel, 100)

    def next(self):
        # If mal crosses above ma2 , buy the asset
        if crossover(self.ma1 , self.ma2):
            self.buy()
        # Else , if mal crosses below ma2 , sell it
        elif crossover(self.ma2, self.ma1):
            self.sell()


def draw_moving_average(df, ticker: str,company_name: str):
    

    short_moving_avg = df.Close.rolling(window=20).mean()
    long_moving_avg = df.Close.rolling(window=100).mean()
    
    plt.subplots(figsize=(16, 9))
    plt.plot(df.Close.index, df.Close, label='{} ({})'.format(company_name, ticker), alpha = 0.8)
    plt.plot(short_moving_avg.index, short_moving_avg, label = '20-day MA')
    plt.plot(long_moving_avg.index, long_moving_avg, label = '100-day MA')
    
    import numpy as np
    indicator = np.where(short_moving_avg > long_moving_avg, 55, 5)
    plt.plot(df.Close.index, indicator, alpha = 0.3)
    #plt.plot(tsla)
    plt.xlabel('Date')
    plt.ylabel('Closing price ($)')
    plt.legend(loc = 'lower right')
#    ax =
    plt.show()

def main():
    

    df = web.DataReader('0005.hk', 'yahoo', dt.date(2010, 1, 1), dt.date(2020, 8, 31))
    print(df.head(1))
    print(df.tail(1))
    print('end')
  #  draw_moving_average(df, '0001.HK', 'CK Hutchison')
  #  df = web.DataReader(['0005.hk'], 'yahoo', dt.date(2010, 1, 1), dt.date(2020, 8, 31))
  #  draw_moving_average(df, '0005.HK', 'HSBC')
   # print(df.head(20))
    bt_simple_moving_average_Cross = Backtest(df, simple_moving_average_Cross, cash = 10000, commission = 0.002)

    print(bt_simple_moving_average_Cross.run())
    bt_simple_moving_average_Cross.plot() # the plot cannot be shown if run in Spyder.

if __name__ == '__main__':
    main()
    
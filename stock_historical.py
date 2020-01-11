#!/usr/bin/env python3

import os
import sys
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from tqdm import tqdm

def main(symbols):
    """
    Collect stock price data for arbitrary number of available companies from Yahooo! finance. Save them in stock_data directory.
    """

    path=r'./stock_data'

    os.mkdir(path)

    start = dt.datetime(2005,1,1)
    end = dt.datetime.today()
    for symbol in tqdm(symbols):
        df = web.DataReader(symbol,'yahoo',start,end)
        df['Company'] = str(symbol)
        df.to_csv(os.path.join(path,str(symbol)+'.csv'))
    

if __name__ == '__main__':
    symbols = sys.argv[1:]
    main(symbols)
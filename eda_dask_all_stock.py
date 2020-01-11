#!/usr/bin/env python3

import os
import sys
import numpy as np
import time, pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

from dask.distributed import Client
#client = Client(n_workers=4, threads_per_worker=8, processes=False, memory_limit='2GB')
#client

def make_plt(result, tickers,ylabel,filename,pth):
    fig = plt.gcf()
    fig.set_size_inches(11,8)
    for ticker in tickers:
        result[ticker].plot.line(label=str(ticker))
    plt.xlabel('time', fontsize=18)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(loc="upper left",prop={'size': 16})
    plt.savefig(os.path.join(pth,filename+'.png'), dpi=100)


def make_bar_plt(tickers, bar1,bar2,label1,label2,ylabel,filename, pth):
    x = np.arange(len(tickers))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bar1, width, label=label1)
    rects2 = ax.bar(x + width/2, bar2, width, label=label2)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    plt.legend(loc="upper left",prop={'size': 16})
    plt.savefig(os.path.join(pth,filename+'.png'), dpi=100)


def main(input):
    """
    performs etl on collected data for all companies present in stock_data directory. EDA, comparison of stock prices of various companies.create stock_plots directory and stores the plots there.
    """
    path=r'./stock_plots'
    os.mkdir(path)

    # Read from csv files 
    dd_stock = dd.read_csv(input,dtype={'Volume':'float64'})
    dd_stock['Date']=dd.to_datetime(dd_stock.Date,unit='ns')

    max_open = dd_stock.groupby(dd_stock.Company).Open.max().compute()
    min_open = dd_stock.groupby(dd_stock.Company).Open.min().compute()


    mean_Volume = dd_stock.groupby(dd_stock.Company).Volume.mean().compute()
    min_Volume = dd_stock.groupby(dd_stock.Company).Volume.min().compute()

    # Just to create dask visualize, we can comment out then
    # Boolean series where company is Nike
    #is_NKE = (dd_stock['Company'] == 'NKE')
    # Filter the DataFrame using
    #NKE_Volume = dd_stock.loc[is_NKE, 'Volume']


    #total_volume = NKE_Volume.sum()
    #total_volume.visualize(filename='dask.png', rankdir='LR')


    dd_stock_df = dd_stock.compute()
    tickers = dd_stock_df.Company.unique()

    #add year, months and year with month columns
    dd_stock_df['year'] = dd_stock_df['Date'].dt.year

    dd_stock_df['month'] = dd_stock_df['Date'].dt.month

    dd_stock_df['month_year'] = pd.to_datetime(dd_stock_df['Date']).dt.to_period('M')

    # create dask dataframe again for lazy evaluations in etl
    dd_stock = dd.from_pandas(dd_stock_df, npartitions=3)

    # generate monthly mean of open grouped by company nad month_year columns
    monthly_mean_open = dd_stock.groupby([dd_stock.Company,dd_stock.month_year]).Open.mean()

    result1 = monthly_mean_open.compute()

    # create the plot and save it to directory    
    make_plt(result=result1,tickers=tickers, ylabel='monthly_mean_open',filename='monthly_mean_open',pth=path)



    # create more plots
    make_bar_plt(tickers, bar1=max_open, bar2=min_open,label1='max_open',label2='min_open',ylabel='open',filename='barplot1',pth=path)

    make_bar_plt(tickers, bar1=mean_Volume, bar2=min_Volume, label1='mean_volume',label2='min_volume',ylabel='volume',filename='barplot2',pth=path)


    # Just to create dask visualize, we can comment out then
    # Computational graph of a single output aggregation (for a small number of groups, like 1000)
    #dd_stock.groupby([dd_stock.Company,dd_stock.month_year]).Volume.mean().visualize(filename='dask2.png',node_attr={'penwidth': '6'})

    # Computational graph of an aggregation to four outputs (for a larger number of groups, like 1000000)
    #dd_stock.groupby([dd_stock.Company,dd_stock.month_year]).Volume.mean(split_out=4).visualize(filename='dask3.png',node_attr={'penwidth': '6'})

    

if __name__ == '__main__':
    input= sys.argv[1:]
    main(input)
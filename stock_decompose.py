#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

def make_plt(arg1,arg2, xlab,ylab,filename):
    fig = plt.gcf()
    fig.set_size_inches(11,8)
    plt.scatter(arg1,arg2)
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.savefig(filename+'.png', dpi=100)



def main(filename):
    """
    Generate trend, seasonality, residual, and lag of stock price (Open price). 
    """
    path=r'./stock_plots'

    # read file
    data = pd.read_csv(filename, header = 0, parse_dates = [0], index_col = 0)

    
    target = data['Open']
    log_target = np.log(target)

    log_target.interpolate(inplace = True)

    # decompose data to trend seasonality and residual, later we compare residual with sentiment
    decomposition = seasonal_decompose(target, freq=28, model = 'multiplicative')
    decomposition.plot()
    plt.savefig(os.path.join(path,'decompose.png'), dpi=100)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # compute difference, later we also compare residual with sentiment
    target_lagged = target.shift()
    target_diff = target - target_lagged

    d = {'trend': trend, 'seasonality': seasonal, 'residual':residual, 'difference':target_diff}
    df = pd.DataFrame(data=d)
    df = df.dropna()

    df['standardized_residual'] = StandardScaler().fit_transform(df[['residual']])

    df.to_csv(str(filename)+'stock_decompose.csv')
    

if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)
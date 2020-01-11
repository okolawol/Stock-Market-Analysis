#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler


# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
    



def main(filename):
    """
    Integrate decompose stock data with sentiment data and compute the correlation between the difference, residual and standardized average sentiment of all news by day. 
    """

    path = r'./stock_plots'
    # read decomposed data

    df = pd.read_csv(str(filename)+'stock_decompose.csv', header = 0, parse_dates = [0], index_col = 0)

    # read sentiment data
    sent = pd.read_csv('sentiment_ts.csv', header = 0, parse_dates = [0], index_col = 0)
    sent = sent.sort_values('Date')
    
    sent['standardized_sentiment_all_news'] = StandardScaler().fit_transform(sent[['avg_sentiment_all_news']])
    #sent['standardized_sentiment_top_news'] = StandardScaler().fit_transform(sent[['avg_sentiment_top_news']])


    merged = pd.merge(df, sent, on='Date')



    # Create a pair grid instance
    grid = sns.PairGrid(data= merged,
                        vars = ['difference', 'standardized_residual', 'standardized_sentiment_all_news'], height = 4)

    # Map the plots to the locations
    grid = grid.map_upper(plt.scatter, color = 'darkred')
    grid = grid.map_upper(corr)
    grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
    grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'darkred')
    plt.savefig(os.path.join(path,'corr.png'), dpi=100)

    #merged.to_csv('merged.csv')
    

if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)
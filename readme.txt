IF RUNNING ON CLUSTER MAKE SURE ALL INPUT FILES & FOLDERS ARE COPIED INTO HDFS.


sentiment_ts.csv is an aggregation of our all our news data. Which is needed for the stock and news integration in stock_sentim_corr.py. IT NEEDS TO BE PRESENT IN YOUR HDFS DIRECTORY for it to be found and read.

Our original news data is too large to place here. It also takes about 2 hours to clean and transform. So for testing purposes you can use a small subset of the raw headlines when running the first step of the headline etl process (news_headline_etl.py). But after that there is a cleaned full headline dataset on the compute cluster located in hdfs dfs -ls /user/okolawol/news_headlines_cleaned.

All the plots and trained models created from the original headlines data is present in the plots, models folder as well for reference. Stocks analysis can be done completely with data gotten from stock_historical.py.

Generated files can be overwritten when running on the cluster with test data but the originals will be in the repository if you need them. Trained models for example wont be accurate if it is not trained on the full dataset.
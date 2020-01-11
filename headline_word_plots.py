import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('Word Plots').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

# add more functions as necessary

def main(input_dir,output_dir):
    # main logic starts here
    df_schema = types.StructType([
        types.StructField('title_clean', types.StringType()),
        types.StructField('title', types.StringType()),
        types.StructField('created_utc_iso', types.DateType()),
        types.StructField('polarity_subjectivity', types.ArrayType(types.FloatType()))
    ])

    headlines_df = spark.read.json(input_dir,encoding='utf-8',schema=df_schema).repartition(80)
    split_sentiment_df = headlines_df.withColumn(
        'polarity', functions.element_at(headlines_df['polarity_subjectivity'],1)
    ).withColumn(
        'subjectivity', functions.element_at(headlines_df['polarity_subjectivity'],2)
    ).cache()

    for year_int in range(2008,2020):
        print('Plotting for '+str(year_int))
        headlines_year = split_sentiment_df.where(
            functions.year(split_sentiment_df['created_utc_iso']) == year_int
        ).withColumn('year',functions.year(split_sentiment_df['created_utc_iso']))

        headlines_grouped = headlines_year.groupBy(headlines_year['year']).agg(
            functions.collect_set(headlines_year['title_clean']).alias('titles_group')
        )
        headlines_joined = headlines_grouped.select( functions.array_join(headlines_grouped['titles_group'],' ').alias('joined') )
        string_to_plot = headlines_joined.collect()[0]['joined'] #only one row remaining of concatenated headlines

        wordcloud = WordCloud(background_color='white', stopwords=stopwords, width=1000, height=500).generate(string_to_plot)
        wordcloud.to_file(output_dir + '/'+str(year_int)+'_words.png')

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    stopwords = set(STOPWORDS)
    stopwords.update(['new','world','says','say','year','video','day','may','make','top','amp','report','two','best','home','government','online','time'])
    main(input_dir,output_dir)
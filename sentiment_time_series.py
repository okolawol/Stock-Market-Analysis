import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn as sns
sns.set(rc={'figure.figsize':(12, 4)})

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('Sentiment Graphs').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# add more functions as necessary
#spark sql doesnt allow vector access through query
@functions.udf(returnType=types.FloatType())
def first_element(vector):
    return float(vector[0])

@functions.udf(returnType=types.ArrayType(types.LongType()))
def top_scores(score_list):
    score_desc = sorted(score_list,reverse=True)
    #top 100 scores only
    threshold = 100
    if(len(score_desc) <= threshold):
        return score_desc
    else:
        return score_desc[:threshold]

#spark sql doesnt the column with the array is not iterable
@functions.udf(returnType=types.BooleanType())
def arr_contains(list,value):
    return value in list

def main(input_dir,output_dir):
    # main logic starts here
    df_schema = types.StructType([
        types.StructField('title_clean', types.StringType()),
        types.StructField('created_utc_iso', types.DateType()),
        types.StructField('polarity_subjectivity', types.ArrayType(types.FloatType())),
        types.StructField('score',types.LongType())
    ])
    headlines_df = spark.read.json(input_dir,encoding='utf-8',schema=df_schema).repartition(80).cache()

    agg_scores = headlines_df.groupBy(headlines_df['created_utc_iso']).agg(
        functions.collect_set(headlines_df['score']).alias('scores_per_day')
    )

    top_scores_df = agg_scores.withColumn('sorted_scores',top_scores(agg_scores['scores_per_day']))

    #getting top headlines with the highest scores
    top_headlines = headlines_df.join(functions.broadcast(top_scores_df),on=['created_utc_iso']).where(
        arr_contains(top_scores_df['sorted_scores'],headlines_df['score'])
    ).select(
        headlines_df['title_clean'],
        headlines_df['created_utc_iso'],
        headlines_df['polarity_subjectivity'],
        headlines_df['score']
    ).withColumn(
        'polarity', functions.element_at(headlines_df['polarity_subjectivity'],1)
    ).withColumn(
        'subjectivity', functions.element_at(headlines_df['polarity_subjectivity'],2)
    )

    agg_sentiment_by_day = top_headlines.groupBy(top_headlines['created_utc_iso']).agg(
        functions.avg(top_headlines['polarity']).alias('avg_sentiment')
    ).cache()

    assembler = VectorAssembler(inputCols=['avg_sentiment'], outputCol='features')
    scaler = MinMaxScaler(inputCol='features',outputCol='normalized_avg_vector')

    pipeline = Pipeline(stages=[assembler, scaler])
    scaler_model = pipeline.fit(agg_sentiment_by_day)

    scaled_avg = scaler_model.transform(agg_sentiment_by_day)
    scaled_avg = scaled_avg.withColumn('normalized_avg',first_element(scaled_avg['normalized_avg_vector']))

    #save scaled_avg to file need to coelesce aggregates into 1 file
    #because this will be read by pandas later on which doesnt support multi file
    #scaled_avg.select(
        #scaled_avg['created_utc_iso'].alias('date'),
        #scaled_avg['normalized_avg'].alias('avg_sentiment_top_news')
    #).coalesce(1).write.csv(output_dir, mode='overwrite', compression='gzip')


    #this wil always be each day of year since from 2008 to 2019
    #all data is aggregated into 365 days * 11 years = at around 4000 records
    aggregate_pandas = scaled_avg.select(
        scaled_avg['created_utc_iso'].alias('date'),
        scaled_avg['normalized_avg'].alias('sentiment')
    ).toPandas()

    aggregate_pandas = aggregate_pandas.set_index('date');
    plt.plot(aggregate_pandas['sentiment'],marker='.',alpha=0.5,linestyle='None')
    plt.savefig(output_dir+ '/sentiment_series.png')

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir,output_dir)
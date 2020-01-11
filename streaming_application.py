import sys, os, re
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('Streaming Application').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

from pyspark.ml import PipelineModel
from textblob import TextBlob
import nltk

# add more functions as necessary

@functions.udf(returnType=types.FloatType())
def headline_subjectivity(headline):
    words = TextBlob(headline)
    return words.sentiment.subjectivity;

@functions.udf(returnType=types.StringType())
def clean_sentence(headline):
    sentence = cleaning_regex.sub('', headline).lower()
    sentence = sentence.split(' ')

    for word in list(sentence):
        if word in stop_words:
            sentence.remove(word)

    sentence = ' '.join(sentence)
    return sentence

def prepare_for_process(raw_df):
    print('preping raw inputs...')
    clean_title_df = raw_df.withColumn(
        'title_clean',clean_sentence(raw_df['title'])
    )
    with_subjectivity = clean_title_df.withColumn(
        'subjectivity',headline_subjectivity(clean_title_df['title_clean'])
    ).withColumn(
        'label', functions.lit(0.0) #we need to match the ml model schema
    )
    print('cleaning complete...')
    return with_subjectivity


def main(input_stream,sentiment_model_file):
    # main logic starts here
    headline_schema = types.StructType([
        types.StructField('title', types.StringType()),
        types.StructField('score',types.LongType()),
        types.StructField('num_comments',types.LongType()),
    ])

    #load the sentiment model
    sentiment_model = PipelineModel.load(sentiment_model_file)

    #Load the headline stream
    headlines_stream = spark.readStream.format('json').schema(headline_schema).load(input_stream)

    #match the schema our sentiment model needs
    headlines_stream = prepare_for_process(headlines_stream)

    #make the prediction 0 = lowest,1 = neutral, 2 = sentiment
    predictions_df = sentiment_model.transform(headlines_stream)
    predictions_df = predictions_df.select(predictions_df['title'],predictions_df['prediction'])

    predictions_df.writeStream.format('console').outputMode('append').option('truncate',False).start().awaitTermination(600)

if __name__ == '__main__':
    input_stream = sys.argv[1]
    sentiment_model_file = sys.argv[2]
    cleaning_regex = re.compile('([^\s\w]|_)+')
    stop_words = nltk.corpus.stopwords.words()
    stop_words.append('say')
    main(input_stream,sentiment_model_file)
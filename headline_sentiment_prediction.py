import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from wordcloud import STOPWORDS

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('Sentiment prediction model').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Word2Vec, VectorAssembler, VectorSizeHint
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# add more functions as necessary
#change to 0.05
@functions.udf(returnType=types.FloatType())
def get_label(polarity):
    if (polarity > 0.05):
        val = 2
    elif (polarity < -0.05):
        val = 0
    else:
        val = 1
    return float(val)

def main(input_dir,output_dir):
    # main logic starts here
    df_schema = types.StructType([
        types.StructField('title_clean', types.StringType()),
        types.StructField('polarity_subjectivity', types.ArrayType(types.FloatType())),
        types.StructField('score',types.LongType()),
        types.StructField('num_comments',types.LongType()),
    ])

    headlines_df = spark.read.json(input_dir,encoding='utf-8',schema=df_schema).repartition(80)
    split_sentiment_df = headlines_df.withColumn(
        'polarity', functions.element_at(headlines_df['polarity_subjectivity'],1)
    ).withColumn(
        'subjectivity', functions.element_at(headlines_df['polarity_subjectivity'],2)
    )

    df_sentiment = split_sentiment_df.withColumn('label',get_label(split_sentiment_df['polarity']))

    training_set, validation_set = df_sentiment.randomSplit([0.75, 0.25])

    headline_vector_size = 3
    word_freq_vector_size = 100

    tokenizer = Tokenizer(inputCol='title_clean',outputCol='words')
    headline2Vector = Word2Vec(vectorSize=headline_vector_size,minCount=0,inputCol='words',outputCol='headline_vector')
    hashingTF = HashingTF(inputCol='words',outputCol='word_counts',numFeatures=word_freq_vector_size)
    idf = IDF(inputCol='word_counts',outputCol='word_frequecy',minDocFreq=5)
    headline_vector_size_hint = VectorSizeHint(inputCol='headline_vector',size=headline_vector_size) #need this for streaming
    word_freq_vector_size_hint = VectorSizeHint(inputCol='word_frequecy',size=word_freq_vector_size) #need this for streaming
    feature_assembler = VectorAssembler(inputCols=['headline_vector','score','num_comments','subjectivity','word_frequecy'], outputCol='features')
    dt_classifier = DecisionTreeClassifier(featuresCol='features',labelCol='label',predictionCol='prediction',maxDepth=9)

    pipeline = Pipeline(stages=[tokenizer,  headline2Vector, hashingTF, idf, headline_vector_size_hint, word_freq_vector_size_hint, feature_assembler, dt_classifier])
    sentiment_model = pipeline.fit(training_set)

    validation_predictions = sentiment_model.transform(validation_set)

    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='label')
    validation_score = evaluator.evaluate(validation_predictions)
    print('Validation score for Sentiment model F1: %g' % (validation_score, ))

    validation_score_accuracy = evaluator.evaluate(validation_predictions,{evaluator.metricName: "accuracy"})
    print('Validation score for Sentiment model Accuracy: %g' % (validation_score_accuracy, ))

    sentiment_model.write().overwrite().save(output_dir)
    
    

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir,output_dir)
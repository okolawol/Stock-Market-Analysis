import sys, re
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
import nltk
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('News Headlines ETL').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

# add more functions as necessary
@functions.udf(returnType=types.StringType())
def posix_time_to_iso_string(timestamp):
    return datetime.fromtimestamp(timestamp).isoformat()

@functions.udf(returnType=types.FloatType())
def headline_sentiment_polarity(headline):
    return sentiment_analyser.polarity_scores(headline)['compound'];

@functions.udf(returnType=types.StringType())
def clean_sentence(headline):
    sentence = cleaning_regex.sub('', headline).lower()
    sentence = sentence.split(' ')

    for word in list(sentence):
        if word in stop_words:
            sentence.remove(word)

    sentence = ' '.join(sentence)
    return sentence

@functions.udf(returnType=types.BooleanType())
def english_only(sentence):
    try: 
        return detect(sentence) == 'en'
    except:
        return False

def main(input_dir, output_dir):
    # main logic starts here
    json_df_schema = types.StructType([
        types.StructField('allow_live_comments', types.BooleanType()),
        types.StructField('author', types.StringType()),
        types.StructField('author_flair_css_class', types.StringType()),
        types.StructField('author_flair_text', types.StringType()),
        types.StructField('author_flair_type', types.StringType()),
        types.StructField('author_fullname', types.StringType()),
        types.StructField('author_patreon_flair', types.BooleanType()),
        types.StructField('can_mod_post', types.BooleanType()),
        types.StructField('contest_mode', types.BooleanType()),
        types.StructField('created_utc', types.LongType()),
        types.StructField('domain', types.StringType()),
        types.StructField('full_link', types.StringType()),
        types.StructField('id', types.StringType()),
        types.StructField('is_crosspostable', types.BooleanType()),
        types.StructField('is_meta', types.BooleanType()),
        types.StructField('is_original_content', types.BooleanType()),
        types.StructField('is_reddit_media_domain', types.BooleanType()),
        types.StructField('is_robot_indexable', types.BooleanType()),
        types.StructField('is_self', types.BooleanType()),
        types.StructField('is_video', types.BooleanType()),
        types.StructField('link_flair_background_color', types.StringType()),
        types.StructField('link_flair_text_color', types.StringType()),
        types.StructField('link_flair_type', types.StringType()),
        types.StructField('locked', types.BooleanType()),
        types.StructField('media_only', types.BooleanType()),
        types.StructField('no_follow', types.BooleanType()),
        types.StructField('num_comments', types.IntegerType()),
        types.StructField('num_crossposts', types.IntegerType()),
        types.StructField('og_description', types.StringType()),
        types.StructField('og_title', types.StringType()),
        types.StructField('over_18', types.BooleanType()),
        types.StructField('parent_whitelist_status', types.StringType()),
        types.StructField('permalink', types.StringType()),
        types.StructField('pinned', types.BooleanType()),
        types.StructField('pwls', types.IntegerType()),
        types.StructField('retrieved_on', types.LongType()),
        types.StructField('score', types.IntegerType()),
        types.StructField('selftext', types.StringType()),
        types.StructField('send_replies', types.BooleanType()),
        types.StructField('spoiler', types.BooleanType()),
        types.StructField('stickied', types.BooleanType()),
        types.StructField('subreddit', types.StringType()),
        types.StructField('subreddit_id', types.StringType()),
        types.StructField('subreddit_subscribers', types.LongType()),
        types.StructField('subreddit_type', types.StringType()),
        types.StructField('thumbnail', types.StringType()),
        types.StructField('title', types.StringType()),
        types.StructField('total_awards_received', types.IntegerType()),
        types.StructField('url', types.StringType()),
        types.StructField('whitelist_status', types.StringType()),
        types.StructField('wls', types.IntegerType()),
    ])
    raw_df = spark.read.json(input_dir,schema=json_df_schema,encoding='utf-8',multiLine=True).repartition(80)
    #important fields mmight change but for now these are what we are thinking of using
    no_nulls= raw_df.where(
        raw_df['score'].isNotNull() 
        & raw_df['title'].isNotNull() 
        & raw_df['url'].isNotNull()
        & raw_df['domain'].isNotNull()
        & raw_df['created_utc'].isNotNull()
    ).withColumn('title_clean',clean_sentence(raw_df['title']))

    no_empty_strings = no_nulls.filter(
        (no_nulls['title_clean'] != '') & (english_only(no_nulls['title_clean']))
        & ( (no_nulls['url'] != '') | (no_nulls['domain'] != '') )
    )
    with_iso_time_format = no_empty_strings.withColumn('created_utc_iso',posix_time_to_iso_string(no_empty_strings['created_utc']))

    with_polarity = with_iso_time_format.withColumn('sentiment_polarity',headline_sentiment_polarity(with_iso_time_format['title_clean']))

    with_polarity.write.csv(output_dir, mode='overwrite', compression='gzip')

#run like this: ${SPARK_HOME}/bin/spark-submit news_headline_etl.py [input directory] [output directory]
#this will clean filter the data and fields we dont need, and convert the epoch time to iso timestamp string.
#save compressed json and calculate the polarity of each news headline with textblob
if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    sentiment_analyser = SentimentIntensityAnalyzer()
    cleaning_regex = re.compile('([^\s\w]|_)+')
    stop_words = nltk.corpus.stopwords.words()
    stop_words.append('say')
    main(input_dir, output_dir)
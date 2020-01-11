import sys
import requests
import json
import time
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

def get_url_data(params):
    response = requests.get(url=URL, params=params)
    data = response.json()
    file_name = '/subreddits-'+'after-'+str(params['after'])+'.json'
    with open(output_dir+file_name,'w', encoding='utf-8') as writer:
        json.dump(data['data'], writer, ensure_ascii=False) #remove indentation
    return data['data']

#run like this: python3 news_fetch_api.py [output directory]
# this will gather all headlines from start time: after param till present then stop
# all data will be stored in the output directory specified
if __name__ == '__main__':
    output_dir = sys.argv[1]
    
    subreddit = sys.argv[2]
    start_time = sys.argv[3]
    page_size = sys.argv[4]


    URL = 'https://api.pushshift.io/reddit/search/submission/'
    #1199174400 means jan 1 2008
    PARAMS = {
        'q':'',
        'after': int(start_time) if start_time is not '0' else 1199174400,
        'subreddit': subreddit if subreddit is not '0' else 'worldnews',
        'author':'',
        'aggs':'',
        'metadata': True,
        'frequency':'hour',
        'advanced': False,
        'sort':'asc',
        'domain':'',
        'sort_type':'created_utc',
        'size': int(page_size) if page_size is not '0' else 500
    }
    print(PARAMS)
    while True:
        print('Fetching next series after '+str(PARAMS['after']))
        subreddits = get_url_data(PARAMS)
        if(len(subreddits) > 0):
            PARAMS['after'] = subreddits[-1]['created_utc']
        else:
            break
        time.sleep(1)

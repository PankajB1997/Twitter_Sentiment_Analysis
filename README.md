# Twitter Sentiment Analysis  
  
CS4225 Project  
  
Total tweets added:  31,130,257 (3.01 GB)  
  
This is consolidated from multiple sources, including [an online twitter archive](https://archive.org/details/archiveteam-twitter-stream-2018-10), [Sentiment140 dataset on Kaggle](https://www.kaggle.com/kazanova/sentiment140) and around 40,000 tweets that we downloaded manually using the Twitter API.  
  
To consolidate the entire dataset into one large CSV file, simply clone this repo and run `python prepare_tweets_csv.py`  
  
In order to load the large CSV file in another script after running the python script:  
`tweets_csv = pandas.read_csv(open("tweets_large_32M.csv", "rU", encoding="utf-8"), header=None, index_col=None)`  

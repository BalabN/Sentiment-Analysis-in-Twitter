import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Tweet import Tweet


def get_negAndpos(tweets):
    all_pos = " ".join([tweet.message_without_punctuation for tweet in tweets if tweet.sentiment == Tweet.POSITIVE])
    all_neg = " ".join([tweet.message_without_punctuation for tweet in tweets if tweet.sentiment == Tweet.NEGATIVE])
    all_neutral = " ".join([tweet.message_without_punctuation for tweet in tweets if tweet.sentiment == Tweet.NEUTRAL])
    print(type(all_pos))
    return [all_pos, all_neg, all_neutral]


def get_tweets(data_file):
    tweets = []
    matrix = pd.read_csv(data_file, sep='\t').as_matrix()
    for tweet in matrix:
        tweets.append(Tweet(tweet[0], tweet[1], tweet[2]))
    return tweets


trainTweets = get_tweets('data/train-A.tsv')
testTweets = get_tweets('data/test-A.tsv')

vect = TfidfVectorizer()
tfidf = vect.fit_transform(get_negAndpos(trainTweets))
print(type(tfidf))

weights = vect.transform(Tweet.get_all_messages(testTweets))
print(weights)

res = [Tweet.classes[x] for x in np.argmax(tfidf.A.dot(weights.T.A), axis=0)]
print("klasifikacijska tocnost " + str(
    np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets))]) / len(res)))

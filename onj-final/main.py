from sklearn import linear_model

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Tweet import Tweet


def get_negAndpos(tweets):
    lem_pos = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.POSITIVE]
    lem_neg = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.NEGATIVE]
    lem_neu = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.NEUTRAL]
    all_pos = ' '.join(str(x) for x in lem_pos)
    all_neg = ' '.join(str(x) for x in lem_neg)
    all_neutral = ' '.join(str(x) for x in lem_neu)
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
m = vect.transform(Tweet.get_all_messages(trainTweets))
# print("TF-IDF vectors (each column is a document):\n{}\nRows:\n{}".format(tfidf.T.A, vect.get_feature_names()))

weights = vect.transform(Tweet.get_all_messages(testTweets))
# print("TF-IDF vectors (each column is a document):\n{}\nRows:\n{}".format(weights.T.A, vect.get_feature_names()))

res = [Tweet.classes[x] for x in np.argmax(tfidf.A.dot(weights.T.A), axis=0)]
print("klasifikacijska tocnost " + str(
    np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets))]) / len(res)))

logistic = linear_model.LogisticRegression()
logistic.fit(m, Tweet.get_all_sentiment(trainTweets))
res = logistic.predict(weights)
print("klasifikacijska tocnost regresija " + str(
    np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets))]) / len(res)))

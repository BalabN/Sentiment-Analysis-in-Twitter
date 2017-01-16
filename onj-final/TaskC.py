from sklearn import linear_model

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Tweet import Tweet
from sklearn.ensemble import RandomForestClassifier
import Evaluate


def get_negAndpos(tweets):
    lem_pos = Tweet.get_all_messages_sentiment(tweets, Tweet.POSITIVE)
    lem_neg = Tweet.get_all_messages_sentiment(tweets, Tweet.NEGATIVE)
    lem_neu = Tweet.get_all_messages_sentiment(tweets, Tweet.NEUTRAL)
    lem_vneg = Tweet.get_all_messages_sentiment(tweets, Tweet.VERYNEGATIVE)
    lem_vpos = Tweet.get_all_messages_sentiment(tweets, Tweet.VERYPOSITIVE)

    all_pos = ' '.join(lem_pos)
    all_neg = ' '.join(lem_neg)
    all_neutral = ' '.join(lem_neu)
    all_vneg = ' '.join(lem_vneg)
    all_vpos = ' '.join(lem_vpos)
    return [all_pos, all_neg, all_neutral, all_vpos, all_vneg]


def get_tweets(data_file):
    tweets = []
    matrix = pd.read_csv(data_file, sep='\t').as_matrix()
    for tweet in matrix:
        tweets.append(Tweet(tweet[0], tweet[1], tweet[2], tweet[3]))
    return tweets


trainTweets = get_tweets('data/train-CE.tsv')
testTweets = get_tweets('data/test-CE.tsv')

topics = np.unique(pd.read_csv('data/test-CE.tsv', sep='\t').as_matrix()[:, 1])

vect = TfidfVectorizer()
tfidf = vect.fit_transform(get_negAndpos(trainTweets))

eval_res = []
eval_acc = []
for topic in topics:
    weights_all = vect.transform([' '.join(Tweet.get_all_messages(testTweets, topic))])
    pred_all = tfidf.A.dot(weights_all.T.A)

    weights = vect.transform(Tweet.get_all_messages(testTweets, topic))
    res = [Tweet.sentiments[x] for x in np.argmax(tfidf.A.dot(weights.T.A) * pred_all, axis=0)]

    eval_res += [Evaluate.evaluateC(res, Tweet.get_all_sentiment(testTweets, topic))]
    eval_acc += [
        np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets, topic))]) / len(res)]
print("Evaluation " + str(np.average(eval_res)))
print("Evaluation Acc" + str(np.average(eval_acc)))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Tweet import Tweet
import Evaluate


def get_negAndpos(tweets):
    lem_pos = Tweet.get_all_messages_sentiment(tweets, Tweet.POSITIVE)
    lem_neg = Tweet.get_all_messages_sentiment(tweets, Tweet.NEGATIVE)
    all_pos = ' '.join(lem_pos)
    all_neg = ' '.join(lem_neg)
    return [all_pos, all_neg]


def get_tweets(data_file):
    tweets = []
    matrix = pd.read_csv(data_file, sep='\t').as_matrix()
    for tweet in matrix:
        tweets.append(Tweet(tweet[0], tweet[1], tweet[2], tweet[3]))
    return tweets


trainTweets = get_tweets('data/train-BD.tsv')
testTweets = get_tweets('data/test-BD.tsv')

topics = np.unique(pd.read_csv('data/test-BD.tsv', sep='\t').as_matrix()[:, 1])

vect = TfidfVectorizer()
tfidf = vect.fit_transform(get_negAndpos(trainTweets))

eval_res = []
eval_acc = []

for topic in topics:
    weights_all = vect.transform([' '.join(Tweet.get_all_messages(testTweets, topic))])
    pred_all = tfidf.A.dot(weights_all.T.A)

    weights = vect.transform(Tweet.get_all_messages(testTweets, topic))
    res = [Tweet.sentiments[x] for x in np.argmax(tfidf.A.dot(weights.T.A) * pred_all, axis=0)]

    eval_res += [Evaluate.evaluateB(res, Tweet.get_all_sentiment(testTweets, topic))]
    eval_acc += [
        np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets, topic))]) / len(res)]
print("Evaluation " + str(np.average(eval_res)))
print("Evaluation Acc" + str(np.average(eval_acc)))
